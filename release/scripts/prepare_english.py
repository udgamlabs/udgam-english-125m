#!/usr/bin/env python3
"""Bounded, revision-verified English corpus preflight; never loads model weights.

Example (from workspace root):
  .venv/bin/python scratch_20260922/scripts/prepare_english.py \
    --output-dir scratch_20260922/data/english-preflight-001 \
    --report scratch_20260922/records/data-preflight.json

The viewer API is tied to a repository revision, not an arbitrary old revision.
Every response MUST carry x-revision matching the pinned Hub SHA. A mismatch is
an error, never permission to silently follow main or replace a source.

This samples an equal, bounded number of candidate documents per source to audit
adapters, then selects a 60/30/10 DOCUMENT-count mixture without replacement.
Token proportions are explicitly unmeasured until our tokenizer has been fitted
on training documents. This tiny preflight is not the full pretraining corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata
from urllib.parse import urljoin, urlparse


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
SOURCE_SPECS = {
    "fineweb_edu": {
        "repo": "HuggingFaceFW/fineweb-edu", "configs": ["sample-10BT"],
        "text_field": "text", "initial_weight": 0.6,
        "license": "ODC-By-1.0; Common Crawl Terms of Use",
        "license_notes": "Database terms do not transfer ownership of underlying web text. Retain source URLs and attribution.",
        "teacher": None,
        "curation_dependency": "Educational classification informed by Llama-3-70B-Instruct annotations; natural web text, not generated prose.",
    },
    "finephrase": {
        "repo": "HuggingFaceFW/finephrase", "configs": ["table", "faq", "tutorial"],
        "text_field": "rollout_results[0].text", "initial_weight": 0.3,
        "license": "ODC-By-1.0; upstream FineWeb-Edu / Common Crawl terms",
        "license_notes": "Generated rewrites retain source provenance; author license is not blanket ownership of upstream pages.",
        "teacher": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "curation_dependency": "Synthetic rewrites of FineWeb-Edu sample-350BT. The top-level text is the original, not the rewrite.",
    },
    "nemotron_fact": {
        "repo": "nvidia/Nemotron-Pretraining-Specialized-v1.2",
        "configs": ["Nemotron-Pretraining-Fact-Seeking"],
        "text_field": "text", "initial_weight": 0.1,
        "license": "CC-BY-4.0",
        "license_notes": "Only Fact-Seeking is selected. Preserve per-row license, FineWiki provenance and teacher identity; other repository subsets are excluded.",
        "teacher": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "curation_dependency": "Synthetic fact-seeking text grounded in FineWiki.",
    },
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class BudgetExceeded(RuntimeError):
    pass


class RevisionMismatch(RuntimeError):
    pass


class BoundedHTTP:
    """Limit ALL response-body bytes, including metadata, error and probe bodies."""

    def __init__(self, maximum_bytes: int):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False  # no implicit .netrc credentials
        self.session.headers.update({"Accept-Encoding": "identity", "User-Agent": "udgam-data-preflight/1"})
        self.session.max_redirects = 3
        self.maximum_bytes = maximum_bytes
        self.received_bytes = 0
        self.requests = []

    def get(self, url: str, params=None):
        for _ in range(4):
            body, headers, status, resolved = self._get_one(url, params)
            if status not in (301, 302, 303, 307, 308):
                return body, headers
            if "location" not in headers:
                raise ValueError("Redirect without Location")
            url = urljoin(resolved, headers["location"])
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.username or parsed.password:
                raise ValueError("Refusing non-HTTPS or credential-bearing redirect")
            params = None
        raise ValueError("Too many redirects")

    def _get_one(self, url: str, params=None):
        remaining = self.maximum_bytes - self.received_bytes
        if remaining <= 0:
            raise BudgetExceeded("Total response-body download budget exhausted")
        with self.session.get(url, params=params, stream=True, timeout=(10, 40), allow_redirects=False) as response:
            headers = {k.lower(): v for k, v in response.headers.items()}
            # A known oversized response is not read. For unannounced/chunked
            # responses we stop at the remaining budget without a retry.
            announced = headers.get("content-length")
            if announced and int(announced) > remaining:
                raise BudgetExceeded("Response Content-Length exceeds remaining download budget")
            chunks = []
            while self.received_bytes < self.maximum_bytes:
                size = min(65536, self.maximum_bytes - self.received_bytes)
                chunk = response.raw.read(size, decode_content=True)
                if not chunk:
                    break
                self.received_bytes += len(chunk)
                chunks.append(chunk)
            else:
                # Do not read an extra byte merely to detect EOF: reject this
                # boundary response conservatively while respecting the cap.
                raise BudgetExceeded("Response reached total download budget")
            body = b"".join(chunks)
            self.requests.append({"url": response.url, "status": response.status_code,
                                  "body_bytes": len(body), "x_revision": headers.get("x-revision")})
            response.raise_for_status()
            return body, headers, response.status_code, response.url

    def get_json(self, url: str, params=None):
        body, headers = self.get(url, params)
        return json.loads(body), headers


def pin_sources(http, cutoff: str) -> dict:
    sources = {}
    for name, spec in SOURCE_SPECS.items():
        meta, _ = http.get_json("https://huggingface.co/api/datasets/" + spec["repo"])
        sha = meta.get("sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"Invalid revision returned for {name}")
        if meta.get("lastModified", "9999")[:10] > cutoff:
            raise ValueError(f"{name} latest revision is newer than research cutoff {cutoff}; select an older revision explicitly")
        card_url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{sha}/README.md"
        body, headers = http.get(card_url)
        if headers.get("x-repo-commit", sha) != sha:
            raise RevisionMismatch(f"{name}: pinned card revision mismatch")
        sources[name] = {**spec, "revision": sha, "last_modified": meta.get("lastModified"),
                         "gated": meta.get("gated"), "card_url": card_url,
                         "card_sha256": hashlib.sha256(body).hexdigest(), "card_text": body.decode("utf-8"),
                         "hub_url": "https://huggingface.co/datasets/" + spec["repo"]}
    return {"schema_version": SCHEMA_VERSION, "research_cutoff": cutoff,
            "pinned_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
            "initial_mixture": {n: s["initial_weight"] for n, s in SOURCE_SPECS.items()},
            "mixture_unit": "intended training tokens; preflight uses document counts only",
            "source_selection_policy": "No explicit noncommercial or competition restrictions in selected components; retain upstream obligations. This is not a blanket rights-clearance certificate.",
            "excluded": ["ClimbMix (noncommercial)", "BhashaKritika (commercial and competition restrictions)",
                         "Other Nemotron v1.2 subsets (different licenses/teachers)"]}


def validate_lock(lock: dict, cutoff: str):
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported data source lock schema")
    if lock.get("research_cutoff", "9999") > cutoff:
        raise ValueError("Lock is newer than research cutoff")
    for name, spec in SOURCE_SPECS.items():
        pinned = lock["sources"][name]
        if pinned["repo"] != spec["repo"] or pinned["configs"] != spec["configs"]:
            raise ValueError(f"Unexpected source/config substitution for {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", pinned["revision"]):
            raise ValueError(f"Unpinned revision for {name}")
        if pinned.get("last_modified", "9999")[:10] > cutoff:
            raise ValueError(f"{name} revision postdates cutoff")
        if digest(pinned["card_text"]) != pinned["card_sha256"]:
            raise ValueError(f"{name} saved source card hash mismatch")


def viewer_page(http, source: dict, config: str, offset: int, length: int):
    data, headers = http.get_json("https://datasets-server.huggingface.co/rows", {
        "dataset": source["repo"], "config": config, "split": "train", "offset": offset, "length": length})
    observed = headers.get("x-revision")
    if observed != source["revision"]:
        raise RevisionMismatch(f"{source['repo']}/{config}: viewer x-revision {observed!r} != pinned {source['revision']}")
    return data


def sample_candidates(http, name: str, source: dict, count: int, seed: int, errors=None):
    """Deterministic stratified pages; never requests the full source corpus."""
    configs = source["configs"]
    for index, config in enumerate(configs):
        wanted = count // len(configs) + (index < count % len(configs))
        if not wanted:
            continue
        try:
            yield from sample_config_candidates(http, name, source, config, wanted, seed)
        except Exception as exc:
            if errors is None:
                raise
            errors.append({"source": name, "config": config, "type": type(exc).__name__, "error": str(exc)})


def sample_config_candidates(http, name, source, config, wanted, seed):
    """A failed config must not silently prevent other configs being audited."""
    if wanted:
        probe = viewer_page(http, source, config, 0, 1)
        total = probe.get("num_rows_total")
        if not isinstance(total, int) or total <= 0:
            raise ValueError(f"{name}/{config}: no indexed row count available")
        # Spread pages through the viewer's indexed population, not necessarily
        # the entire dataset if the viewer reports partial=True.
        wanted = min(wanted, total)
        pages = (wanted + 19) // 20
        rng = random.Random(digest(f"{seed}:{name}:{config}"))
        used = set()
        for page_index in range(pages):
            length = min(20, wanted - page_index * 20)
            lo, hi = total * page_index // pages, total * (page_index + 1) // pages
            start = rng.randint(lo, max(lo, hi - length))
            data = viewer_page(http, source, config, start, length)
            for item in data.get("rows", []):
                row_index = item.get("row_idx")
                if row_index in used:
                    continue
                used.add(row_index)
                yield {"source": name, "config": config, "row_index": row_index,
                       "viewer_partial": bool(data.get("partial")),
                       "viewer_indexed_rows": total, "row": item.get("row", {}),
                       "truncated_cells": item.get("truncated_cells", [])}


class Rejected(ValueError):
    pass


def quality_check(text: str, min_words: int, max_chars: int):
    if not isinstance(text, str) or not text.strip():
        raise Rejected("empty_text")
    text = unicodedata.normalize("NFC", text).strip()
    if len(text) > max_chars:
        raise Rejected("overlength_text")
    words = re.findall(r"\w+", text)
    if len(words) < min_words:
        raise Rejected("too_short")
    if normalized(text) in {"no answer", "document:", "no output", "n/a"}:
        raise Rejected("placeholder_output")
    if len(words) >= 30:
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        if 1 - len(set(grams)) / len(grams) > 0.4:
            raise Rejected("repetitive_text")
    if "\ufffd" in text or any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise Rejected("invalid_control_or_replacement_character")
    return text


def adapt(candidate: dict, pinned: dict, min_words=30, max_chars=50000) -> dict:
    if candidate.get("truncated_cells"):
        raise Rejected("api_truncated_cells")
    name, row = candidate["source"], candidate["row"]
    original = None
    if name == "finephrase":
        original = row.get("text")
        if not isinstance(original, str) or not original.strip() or not row.get("id"):
            raise Rejected("missing_finephrase_source_provenance")
        results = row.get("rollout_results")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise Rejected("missing_generated_rollout")
        result = results[0]
        if result.get("finish_reason") not in ("stop", "eos_token"):
            raise Rejected("incomplete_or_failed_rollout")
        text = result.get("text")
        if isinstance(text, str) and normalized(text) == normalized(original):
            raise Rejected("rewrite_identical_to_input")
        source_id = str(row["id"])
        group_keys = ["fineweb-id:" + source_id, "natural-text:" + digest(normalized(original))]
    elif name == "fineweb_edu":
        text = row.get("text")
        if not row.get("id") or not row.get("url"):
            raise Rejected("missing_fineweb_source_provenance")
        source_id = str(row["id"])
        group_keys = ["fineweb-id:" + source_id]
        if isinstance(text, str):
            group_keys.append("natural-text:" + digest(normalized(text)))
        if row.get("language") not in (None, "en"):
            raise Rejected("non_english_source")
    elif name == "nemotron_fact":
        text = row.get("text")
        source_id = row.get("uuid")
        if not source_id or not row.get("license"):
            raise Rejected("missing_nemotron_source_provenance")
        licenses = row["license"] if isinstance(row["license"], list) else [row["license"]]
        if {str(value).casefold().replace("_", "-") for value in licenses} != {"cc-by-4.0"}:
            raise Rejected("unexpected_nemotron_row_license")
        group_keys = ["nemotron-uuid:" + str(source_id)]
        # Explicit source identity when supplied; absence remains visible.
        metadata = row.get("metadata") or {}
        if isinstance(metadata, dict):
            for key in ("source_url", "url", "wiki_id", "document_id", "source_id"):
                if metadata.get(key):
                    group_keys.append("finewiki:" + key + ":" + str(metadata[key]))
    else:
        raise ValueError(f"No adapter for source {name}")
    text = quality_check(text, min_words, max_chars)
    fingerprint = digest(normalized(text))
    group_keys.append("output-text:" + fingerprint)
    metadata = row.get("metadata")
    return {"text": text, "source": name, "dataset": pinned["repo"], "revision": pinned["revision"],
            "config": candidate["config"], "row_index": candidate["row_index"],
            "source_id": str(source_id), "source_url": row.get("url"),
            "source_metadata": metadata, "source_row_sha256": digest(canonical_json(row)),
            "group_keys": sorted(set(group_keys)), "normalized_text_sha256": fingerprint,
            "text_sha256": digest(text), "license": pinned["license"],
            "row_license": row.get("license"), "teacher": pinned["teacher"],
            "curation_dependency": pinned["curation_dependency"],
            "original_text": original, "viewer_partial": candidate["viewer_partial"],
            "viewer_indexed_rows": candidate["viewer_indexed_rows"],
            "source_document_identity_available": name != "nemotron_fact" or len(group_keys) > 2}


class BenchmarkDeny:
    def __init__(self, texts=(), ngram=13):
        self.ngram = ngram
        self.ngrams = set()
        self.full = set()
        self.input_count = 0
        for text in texts:
            normalized_text = normalized(text)
            if not normalized_text:
                continue
            self.input_count += 1
            self.full.add(normalized_text)
            words = normalized_text.split()
            self.ngrams.update(tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1))

    def matches(self, text):
        value = normalized(text or "")
        if value in self.full:
            return True
        words = value.split()
        return any(tuple(words[i:i + self.ngram]) in self.ngrams for i in range(len(words) - self.ngram + 1))


def load_deny(paths):
    texts = []
    for path in paths:
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    text = row.get("text", row.get("prompt"))
                    if not isinstance(text, str):
                        raise ValueError("Deny JSONL must contain a text or prompt string")
                    texts.append(text)
        else:
            texts.extend(path.read_text().split("\n\n"))
    return BenchmarkDeny(texts)


def group_deduplicate_split(records, seed, validation_fraction):
    """Union source/output identities before split; deterministic under reorder."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    for record in records:
        keys = record["group_keys"]
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parents[max(a, b)] = min(a, b)
    result, seen = [], set()
    for record in sorted(records, key=lambda r: (r["source"], r["source_id"], r["config"], r["text_sha256"])):
        fingerprint = record["normalized_text_sha256"]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        group = find(record["group_keys"][0])
        bucket = int(digest(f"{seed}:{group}")[:16], 16) / 2**64
        result.append({**record, "group_id": digest(group), "split": "val" if bucket < validation_fraction else "train"})
    return result, len(records) - len(result)


def select_document_mixture(records, weights, seed):
    """Bounded without-replacement selection; token ratios are NOT asserted."""
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    if any(not by_source[name] for name in weights):
        return []
    maximum = int(min(len(by_source[name]) / weight for name, weight in weights.items()))
    selected = []
    for name, weight in weights.items():
        quota = int(maximum * weight)
        ordered = sorted(by_source[name], key=lambda r: digest(f"{seed}:{r['source_id']}:{r['config']}:{r['text_sha256']}"))
        selected.extend(ordered[:quota])
    return sorted(selected, key=lambda r: digest(f"{seed}:order:{r['text_sha256']}"))


def write_json_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def write_corpus(output_dir, all_records, selected):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for filename, rows in [("accepted.jsonl", all_records), ("mixture.jsonl", selected)]:
        path = output_dir / filename
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Split directories for downstream adapters AND a flat nanochat view whose
    # final lexicographic shard is validation, matching nanochat.dataset.py.
    flat = output_dir / "nanochat"
    flat.mkdir()
    for split in ("train", "val"):
        rows = [r for r in selected if r["split"] == split]
        directory = output_dir / split
        directory.mkdir()
        table = pa.table({"text": [r["text"] for r in rows], "group_id": [r["group_id"] for r in rows],
                          "source": [r["source"] for r in rows]}, schema=pa.schema([
                              ("text", pa.string()), ("group_id", pa.string()), ("source", pa.string())]))
        file = directory / "shard_00000.parquet"
        pq.write_table(table, file, compression="zstd", row_group_size=32)
        flat_name = "shard_00000.parquet" if split == "train" else "shard_00001.parquet"
        (flat / flat_name).write_bytes(file.read_bytes())
        hashes[str(file.relative_to(output_dir))] = hashlib.sha256(file.read_bytes()).hexdigest()
        hashes["nanochat/" + flat_name] = hashlib.sha256((flat / flat_name).read_bytes()).hexdigest()
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lockfile", type=Path, default=PROJECT / "research/data_sources.json")
    parser.add_argument("--report", type=Path, default=PROJECT / "records/data-preflight.json")
    parser.add_argument("--cutoff", default="2026-09-22")
    parser.add_argument("--max-download-bytes", type=int, default=5_000_000)
    parser.add_argument("--documents-per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--benchmark-deny", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.report.exists():
        parser.error("Output directory and report must not already exist; use new paths")
    if not 0 < args.validation_fraction < 1 or args.documents_per_source <= 0 or args.max_download_bytes <= 0:
        parser.error("Invalid bounds or validation fraction")
    http = BoundedHTTP(args.max_download_bytes)
    errors, rejected, accepted, sampled_counts = [], Counter(), [], Counter()
    report = {"schema_version": SCHEMA_VERSION, "experiment": "English source adapter preflight, not full training corpus",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "research_cutoff": args.cutoff,
              "seed": args.seed, "max_download_bytes": args.max_download_bytes, "documents_per_source": args.documents_per_source,
              "initial_document_mixture": {n: s["initial_weight"] for n, s in SOURCE_SPECS.items()},
              "token_mixture_measured": False, "pretrained_weights_downloaded": False,
              "benchmark_deny_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.benchmark_deny]}
    try:
        if args.lockfile.exists():
            lock = json.loads(args.lockfile.read_text())
        else:
            lock = pin_sources(http, args.cutoff)
            write_json_new(args.lockfile, lock)
        validate_lock(lock, args.cutoff)
        deny = load_deny(args.benchmark_deny)
        report["benchmark_deny_entries"] = deny.input_count
        report["source_revisions"] = {n: s["revision"] for n, s in lock["sources"].items()}
        for name, pinned in lock["sources"].items():
            try:
                for candidate in sample_candidates(http, name, pinned, args.documents_per_source, args.seed, errors):
                    sampled_counts[name] += 1
                    try:
                        record = adapt(candidate, pinned, args.min_words, args.max_chars)
                        if deny.matches(record["text"]) or deny.matches(record["original_text"]):
                            raise Rejected("benchmark_text_overlap")
                        accepted.append(record)
                    except Rejected as exc:
                        rejected[f"{name}:{exc}"] += 1
            except Exception as exc:
                errors.append({"source": name, "type": type(exc).__name__, "error": str(exc)})
        records, duplicates = group_deduplicate_split(accepted, args.seed, args.validation_fraction)
        selected = select_document_mixture(records, report["initial_document_mixture"], args.seed)
        report["hashes"] = write_corpus(args.output_dir, records, selected)
        report["source_candidates"] = dict(sampled_counts)
        report["accepted_by_source"] = dict(Counter(r["source"] for r in records))
        report["selected_by_source"] = dict(Counter(r["source"] for r in selected))
        report["selected_text_bytes_by_source"] = {n: sum(len(r["text"].encode("utf-8")) for r in selected if r["source"] == n) for n in SOURCE_SPECS}
        report["selected_by_split"] = dict(Counter(r["split"] for r in selected))
        report["normalized_exact_duplicates_removed"] = duplicates
        report["rejections"] = dict(rejected)
        train_groups = {r["group_id"] for r in selected if r["split"] == "train"}
        val_groups = {r["group_id"] for r in selected if r["split"] == "val"}
        report["train_validation_group_overlap"] = len(train_groups & val_groups)
        report["records_without_original_document_identity"] = sum(not r["source_document_identity_available"] for r in records)
        report["viewer_partial_sources"] = sorted({r["source"] for r in records if r["viewer_partial"]})
        if not train_groups or not val_groups:
            errors.append({"type": "InsufficientSplit", "error": "No nonempty train/validation pair after filtering and mixture selection"})
        if any(not report["accepted_by_source"].get(name) for name in SOURCE_SPECS):
            errors.append({"type": "MissingSource", "error": "At least one required source yielded zero accepted documents; no substitution made"})
        report["output_dir"] = str(args.output_dir.resolve())
    except Exception as exc:
        errors.append({"type": type(exc).__name__, "error": str(exc)})
    report["status"] = "passed_adapter_preflight" if not errors else "blocked_or_partial"
    report["errors"] = errors
    report["downloaded_response_body_bytes"] = http.received_bytes
    report["requests"] = http.requests
    report["ready_for_main_training"] = False
    report["limitations"] = ["Small source-adapter preflight; sampling is over the viewer's indexed population.",
        "Mixture matches document-count targets only; actual token fractions must be measured with our train-only tokenizer.",
        "Only normalized exact deduplication and supplied benchmark exact/13-word overlaps are checked; paraphrases and hidden leakage remain possible.",
        "Fact-Seeking may lack original FineWiki document IDs, preventing complete source-document grouping; row UUID grouping alone cannot establish disjoint fact provenance.",
        "No benchmark overlap clearance is claimed without a frozen benchmark deny file.",
        "No trained checkpoint, model quality result or fully cleared commercial release is produced by this script."]
    write_json_new(args.report, report)
    print(json.dumps({k: report.get(k) for k in ["status", "downloaded_response_body_bytes", "source_candidates", "accepted_by_source", "selected_by_source", "selected_by_split", "errors"]}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
