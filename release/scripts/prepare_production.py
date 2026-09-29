#!/usr/bin/env python3
"""Freeze benchmark exclusions, then prepare bounded pinned English Parquet inputs.

No model or tokenizer weights, paid API, source repetition or implicit fallback.
The output is an un-tokenized corpus; only measured tokenizer/packer supply can
establish pilot readiness. This script never marks a GPU run ready.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote, urljoin, urlparse
import zlib

import prepare_english as base

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "hellaswag": ("Rowan/hellaswag", "default", "validation"),
    "piqa": ("baber/piqa", "default", "validation"),
    "boolq": ("aps/super_glue", "boolq", "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
}
COMPONENTS = {
    "fineweb_edu": ("fineweb_edu", "sample-10BT", "sample/10BT", .60),
    "finephrase_table": ("finephrase", "table", "table", .15),
    "finephrase_tutorial": ("finephrase", "tutorial", "tutorial", .15),
    "nemotron_fact": ("nemotron_fact", "Nemotron-Pretraining-Fact-Seeking", "Nemotron-Pretraining-Fact-Seeking", .10),
}


class HTTP:
    """Count every received body, including redirects/errors/range reads."""
    def __init__(self, budget):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"Accept-Encoding": "identity", "User-Agent": "udgam-production-prep/1"})
        self.budget, self.received, self.requests = budget, 0, []

    def get(self, url, params=None, byte_range=None):
        original = url
        headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
        for _ in range(5):
            remaining = self.budget - self.received
            if remaining <= 0:
                raise base.BudgetExceeded("Download byte budget exhausted")
            with self.session.get(url, params=params, headers=headers, stream=True, allow_redirects=False, timeout=(15, 90)) as response:
                h = {k.lower(): v for k, v in response.headers.items()}
                redirect = response.status_code in (301, 302, 303, 307, 308)
                if byte_range is not None and not redirect and response.status_code != 206:
                    raise ValueError(f"Server did not honor bounded Range request: HTTP {response.status_code}")
                announced = int(h.get("content-length", 0))
                expected = byte_range[1] - byte_range[0] + 1 if byte_range is not None and not redirect else None
                if announced > remaining or (expected is not None and announced > expected):
                    raise base.BudgetExceeded("Response exceeds remaining or requested byte bound")
                chunks, count = [], 0
                limit = min(remaining, expected) if expected is not None else remaining
                while count < limit:
                    chunk = response.raw.read(min(131072, limit - count), decode_content=True)
                    if not chunk:
                        break
                    chunks.append(chunk); count += len(chunk); self.received += len(chunk)
                body = b"".join(chunks)
                self.requests.append({"url": original, "status": response.status_code, "body_bytes": count,
                                      "range": byte_range, "etag": h.get("etag")})
                if redirect:
                    url = urljoin(response.url, h["location"])
                    parsed = urlparse(url)
                    if parsed.scheme != "https" or parsed.username or parsed.password:
                        raise ValueError("Unsafe redirect")
                    params = None
                    continue
                response.raise_for_status()
                if expected is not None:
                    observed = h.get("content-range", "")
                    if not observed.startswith(f"bytes {byte_range[0]}-{byte_range[1]}/") or count != expected:
                        raise ValueError("Incomplete or mismatched range response")
                elif count == limit:
                    raise base.BudgetExceeded("Unbounded response reached remaining budget")
                return body, h
        raise ValueError("Too many redirects")

    def json(self, url, params=None):
        b, h = self.get(url, params)
        return json.loads(b), h


class RangeFile(io.RawIOBase):
    """Seekable immutable file backed by explicit bounded HTTP Range reads."""
    def __init__(self, http, url, size):
        super().__init__()
        self.http, self.url, self.size, self.position = http, url, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        new = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if new < 0 or new > self.size: raise ValueError("Seek outside file")
        self.position = new
        return new
    def read(self, size=-1):
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if not count: return b""
        if count > 64 * 1024 * 1024: raise base.BudgetExceeded("Single parquet read exceeds64MiB; no whole-file fallback")
        data, _ = self.http.get(self.url, byte_range=(self.position, self.position + count - 1))
        self.position += len(data)
        return data


def pinned_metadata(http, repo, cutoff):
    meta, _ = http.json("https://huggingface.co/api/datasets/" + repo,
                        [("expand[]", field) for field in ["sha", "lastModified", "gated", "cardData"]])
    if not re.fullmatch("[0-9a-f]{40}", meta.get("sha", "")) or meta["lastModified"][:10] > cutoff:
        raise ValueError("Invalid or post-cutoff dataset revision")
    if meta.get("gated"):
        raise ValueError("Dataset is gated; no authenticated fallback permitted")
    return meta


def parquet_inventory(http, repo, revision, prefix="", maximum_pages=8):
    endpoint = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}"
    if prefix: endpoint += "/" + quote(prefix, safe="/")
    params, entries, pages = {"recursive": "true", "limit": 1000}, [], 0
    while endpoint:
        items, headers = http.json(endpoint, params)
        if not isinstance(items, list): raise ValueError("Unexpected Hub tree response")
        entries.extend({"path": x["path"], "size": x["size"], "oid": x.get("oid"), "lfs": x.get("lfs")}
                       for x in items if x.get("type") == "file" and x["path"].endswith(".parquet"))
        pages += 1
        link = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
        endpoint = link.group(1) if link and pages < maximum_pages else None
        params = None
    if not entries: raise ValueError(f"No parquet source files found for {repo}/{prefix}")
    return sorted(entries, key=lambda e: e["path"]), bool(link)


def open_parquet(http, repo, revision, entry):
    import pyarrow.parquet as pq
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(entry['path'], safe='/')}"
    remote = RangeFile(http, url, entry["size"])
    # A small benchmark/source file is cheaper to acquire once than through
    # dozens of column requests; large files remain row-group range streams.
    if entry["size"] <= 16 * 1024 * 1024:
        return pq.ParquetFile(io.BytesIO(remote.read()), pre_buffer=False)
    return pq.ParquetFile(remote, pre_buffer=True)


def benchmark_fields(task, row):
    if task == "arc_challenge":
        return [row["question"], *row["choices"]["text"], row["question"] + " " + " ".join(row["choices"]["text"])]
    if task == "hellaswag": return [row["ctx"], *row["endings"], *[row["ctx"] + " " + ending for ending in row["endings"]]]
    if task == "piqa": return [row["goal"], row["sol1"], row["sol2"], row["goal"] + " " + row["sol1"], row["goal"] + " " + row["sol2"]]
    if task == "boolq": return [row["passage"], row["question"], row["passage"] + " " + row["question"]]
    if task == "winogrande": return [row["sentence"], row["sentence"].replace("_", row["option1"]), row["sentence"].replace("_", row["option2"])]
    raise ValueError(task)


def select_suite(task, rows, count=500):
    seen, chosen = set(), []
    for index, row in enumerate(rows):
        checksum = base.digest(base.canonical_json(row))
        if checksum in seen: continue
        seen.add(checksum)
        key = base.digest(f"english-small-v1|20260922|{task}|{index}|{checksum}")
        chosen.append({"original_split_row_index": index, "document_sha256": checksum, "selection_hash": key,
                       "original_id": row.get("id", row.get("ind", row.get("qID")))})
    chosen.sort(key=lambda x: x["selection_hash"])
    if len(chosen) < count: raise ValueError("Benchmark split too small for registered selection")
    return chosen[:count]


def benchmark_files(files, config, split):
    pattern = re.compile(r"(?:^|[/_])" + re.escape(split) + r"(?:[-_.]|/)")
    return [e for e in files if pattern.search(e["path"]) and (config == "default" or e["path"].startswith(config + "/"))]


def freeze_benchmarks(http, directory, cutoff):
    """Fetch full five evaluation splits before any training-body acquisition."""
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "cutoff": cutoff, "selection_seed": 20260922,
                "selection_rule": "english-small-v1|20260922|task_name|original_split_row_index|canonical_document_sha256",
                "tasks": {}, "status": "freezing"}
    texts = []
    for task, (repo, config, split) in BENCHMARKS.items():
        meta = pinned_metadata(http, repo, cutoff)
        files, partial = parquet_inventory(http, repo, meta["sha"])
        matched = benchmark_files(files, config, split)
        if not matched or partial: raise ValueError(f"Incomplete or ambiguous benchmark file inventory for {task}")
        rows = []
        for entry in matched:
            parquet = open_parquet(http, repo, meta["sha"], entry)
            rows.extend(parquet.read().to_pylist())
        path = directory / f"{task}.{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for row in rows: f.write(base.canonical_json(row) + "\n")
        for row in rows: texts.extend(benchmark_fields(task, row))
        manifest["tasks"][task] = {"repo": repo, "revision": meta["sha"], "last_modified": meta["lastModified"],
            "config": config, "split": split, "license_metadata": (meta.get("cardData") or {}).get("license"),
            "files": matched, "total_rows": len(rows), "local_file": str(path), "sha256": file_hash(path),
            "selected": select_suite(task, rows)}
        print(json.dumps({"stage": "benchmark_frozen", "task": task, "rows": len(rows), "downloaded_bytes": http.received}), flush=True)
    manifest.update(status="frozen_before_training_acquisition", download_bytes=http.received,
                    frozen_at_utc=datetime.now(timezone.utc).isoformat())
    base.write_json_new(directory / "manifest.json", manifest)
    return manifest, texts


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def words(text): return re.findall(r"\w+", base.normalized(text))


def rolling_hashes(tokens, n):
    if len(tokens) < n: return
    mask, multiplier, value = (1 << 64) - 1, 1_000_003, 0
    factor = pow(multiplier, n - 1, 1 << 64)
    values = [zlib.crc32(t.encode("utf-8")) + 1 for t in tokens]
    for index, word in enumerate(values):
        if index >= n: value = (value - values[index - n] * factor) & mask
        value = (value * multiplier + word) & mask
        if index >= n - 1: yield value


class Deny:
    """Compact deterministic 64-bit13-word fingerprints plus short8–12-word fields.

    Fingerprint collisions can only cause conservative rejection. This is direct
    overlap screening, never proof against paraphrase or teacher contamination.
    """
    def __init__(self, texts):
        self.full, self.grams = set(), defaultdict(set)
        for text in texts:
            t = words(text)
            if not t: continue
            self.full.add(" ".join(t))
            n = min(13, len(t))
            if n >= 8: self.grams[n].update(rolling_hashes(t, n))
    def matches(self, text):
        t = words(text or "")
        return " ".join(t) in self.full or any(any(h in corpus for h in rolling_hashes(t, n)) for n, corpus in self.grams.items())


def source_candidates(http, name, pinned, config, prefix, seed, inventories):
    files, partial = parquet_inventory(http, pinned["repo"], pinned["revision"], prefix)
    inventories[name] = {"files": files, "inventory_partial": partial}
    ordered = sorted(files, key=lambda e: base.digest(f"{seed}:{name}:{e['path']}"))
    for entry in ordered:
        parquet = open_parquet(http, pinned["repo"], pinned["revision"], entry)
        wanted = ("text", "id", "url", "language") if name == "fineweb_edu" else ("text", "id", "url", "rollout_results") if name.startswith("finephrase") else ("text", "uuid", "license", "metadata")
        columns = [c for c in wanted if c in parquet.schema_arrow.names]
        offsets, total = [], 0
        for index in range(parquet.num_row_groups): offsets.append(total); total += parquet.metadata.row_group(index).num_rows
        groups = sorted(range(parquet.num_row_groups), key=lambda i: base.digest(f"{seed}:{name}:{entry['path']}:{i}"))
        # Spread through source files instead of consuming a single large shard.
        for group in groups[:4]:
            batch = parquet.read_row_group(group, columns=columns, use_threads=False)
            for relative, row in enumerate(batch.to_pylist()):
                yield {"source": "finephrase" if name.startswith("finephrase") else name,
                       "config": config, "row_index": offsets[group] + relative, "row": row,
                       "viewer_partial": False, "viewer_indexed_rows": total, "truncated_cells": [],
                       "source_file": entry, "source_row_group": group, "source_row_in_group": relative}


def export_corpus(directory, records):
    import pyarrow as pa
    import pyarrow.parquet as pq
    hashes = {}
    for split in ("train", "val"):
        rows = [r for r in records if r["split"] == split]
        destination = directory / split; destination.mkdir()
        parquet = destination / "shard_00000.parquet"
        schema = pa.schema([(c, pa.string()) for c in ("text", "group_id", "source", "source_id", "config")])
        table = pa.Table.from_pylist([{c: r[c] for c in schema.names} for r in rows], schema=schema)
        pq.write_table(table, parquet, compression="zstd", row_group_size=1024)
        provenance = destination / "provenance.jsonl.gz"
        with gzip.open(provenance, "wt", encoding="utf-8") as f:
            for r in rows: f.write(base.canonical_json(r) + "\n")
        for path in (parquet, provenance): hashes[str(path.relative_to(directory))] = file_hash(path)
    return hashes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--lockfile", type=Path, default=ROOT / "research/data_sources.json")
    p.add_argument("--benchmark-dir", type=Path, default=ROOT / "data/benchmark-exclusions-001")
    p.add_argument("--target-text-bytes", type=int, default=200_000_000)
    p.add_argument("--training-download-cap", type=int, default=2_000_000_000)
    p.add_argument("--benchmark-download-cap", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--cutoff", default="2026-09-22")
    p.add_argument("--validation-fraction", type=float, default=.03)
    a = p.parse_args(argv)
    if a.output_dir.exists() or a.report.exists(): p.error("Output/report exists; choose fresh immutable paths")
    if a.target_text_bytes <= 0 or not 0 < a.validation_fraction < .5: p.error("Invalid text/split bound")
    if shutil.disk_usage(ROOT).free < 4 * 1024**3: p.error("Need at least4GiB local free space")
    lock = json.loads(a.lockfile.read_text()); base.validate_lock(lock, a.cutoff)
    bench_http, train_http = HTTP(a.benchmark_download_cap), HTTP(a.training_download_cap)
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
              "seed": a.seed, "cutoff": a.cutoff, "target_text_bytes": a.target_text_bytes,
              "source_lock_sha256": file_hash(a.lockfile), "script_sha256": file_hash(Path(__file__)),
              "byte_targets_not_token_fractions": True, "ready_for_training": False, "errors": []}
    try:
        manifest_path = a.benchmark_dir / "manifest.json"
        if manifest_path.exists():
            benchmark = json.loads(manifest_path.read_text()); texts = []
            if benchmark.get("status") != "frozen_before_training_acquisition": raise ValueError("Incomplete benchmark manifest")
            for task, info in benchmark["tasks"].items():
                path = Path(info["local_file"])
                if file_hash(path) != info["sha256"]: raise ValueError("Benchmark file hash mismatch")
                with gzip.open(path, "rt") as f:
                    for line in f: texts.extend(benchmark_fields(task, json.loads(line)))
        else: benchmark, texts = freeze_benchmarks(bench_http, a.benchmark_dir, a.cutoff)
        report["benchmark_manifest_sha256"] = file_hash(manifest_path)
        report["benchmark_rows"] = {k: v["total_rows"] for k, v in benchmark["tasks"].items()}
        deny = Deny(texts); del texts
        report["benchmark_ngram_counts"] = {str(k): len(v) for k, v in deny.grams.items()}
        accepted, rejected, counts, byte_counts, inventories = [], Counter(), Counter(), Counter(), {}
        for component, (source, config, prefix, weight) in COMPONENTS.items():
            target = int(a.target_text_bytes * weight)
            try:
                for candidate in source_candidates(train_http, component, lock["sources"][source], config, prefix, a.seed, inventories):
                    counts[component] += 1
                    try:
                        record = base.adapt(candidate, lock["sources"][source], min_words=20 if source == "nemotron_fact" else 30)
                        if deny.matches(record["text"]) or deny.matches(record["original_text"]): raise base.Rejected("benchmark_overlap")
                        record.update(component=component, source_file=candidate["source_file"],
                                      source_row_group=candidate["source_row_group"], source_row_in_group=candidate["source_row_in_group"])
                        accepted.append(record); byte_counts[component] += len(record["text"].encode("utf-8"))
                    except base.Rejected as e: rejected[f"{component}:{e}"] += 1
                    if byte_counts[component] >= target: break
                if byte_counts[component] < target: report["errors"].append({"component": component, "error": "Finite selected source inventory exhausted below target"})
            except Exception as e: report["errors"].append({"component": component, "error": f"{type(e).__name__}: {e}"})
            print(json.dumps({"stage": "component_complete", "component": component, "accepted_text_bytes": byte_counts[component],
                              "candidates": counts[component], "downloaded_bytes": train_http.received}), flush=True)
        records, duplicates = base.group_deduplicate_split(accepted, a.seed, a.validation_fraction)
        # One record/source group in validation; all related originals/rewrites
        # were assigned as a group first, so dropping siblings cannot leak.
        val_seen, final, removed_val = set(), [], 0
        for r in records:
            if r["split"] == "val":
                if r["group_id"] in val_seen: removed_val += 1; continue
                val_seen.add(r["group_id"])
            final.append(r)
        final.sort(key=lambda r: base.digest(f"{a.seed}:production-order:{r['text_sha256']}"))
        a.output_dir.mkdir(parents=True, exist_ok=False)
        report["hashes"] = export_corpus(a.output_dir, final)
        report.update(output_dir=str(a.output_dir.resolve()), candidate_rows=dict(counts),
                      accepted_before_deduplication=len(accepted), exact_duplicates_removed=duplicates,
                      validation_group_siblings_removed=removed_val, rejected=dict(rejected), inventories=inventories,
                      source_revisions={k: v["revision"] for k,v in lock["sources"].items()},
                      rows_by_split=dict(Counter(r["split"] for r in final)),
                      text_bytes_by_split={s: sum(len(r["text"].encode("utf-8")) for r in final if r["split"] == s) for s in ["train", "val"]},
                      text_bytes_by_component={c: sum(len(r["text"].encode("utf-8")) for r in final if r["component"] == c) for c in COMPONENTS},
                      group_overlap=len({r["group_id"] for r in final if r["split"] == "train"} & val_seen),
                      records_without_original_document_identity=sum(not r["source_document_identity_available"] for r in final))
        report["status"] = "corpus_prepared_token_supply_unmeasured" if not report["errors"] else "partial_corpus_token_supply_unmeasured"
    except Exception as e:
        report["status"] = "blocked"; report["errors"].append({"error": f"{type(e).__name__}: {e}"})
    report.update(benchmark_download_bytes=bench_http.received, training_download_bytes=train_http.received,
                  benchmark_requests=bench_http.requests, training_requests=train_http.requests,
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=["Token counts/fractions not measured; fit scratch tokenizer on train only before deciding supply.",
                  "Direct benchmark8–13word overlap checks do not prove absence of paraphrase/teacher contamination.",
                  "Nemotron rows lack original FineWiki identity; UUID grouping cannot guarantee source-document independence.",
                  "Raw web source rights remain upstream; no blanket relicensing or public redistribution of benchmark text."])
    base.write_json_new(a.report, report)
    print(json.dumps({k: report.get(k) for k in ["status", "rows_by_split", "text_bytes_by_split", "training_download_bytes", "benchmark_download_bytes", "errors"]}, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__": sys.exit(main())
