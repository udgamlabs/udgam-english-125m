#!/usr/bin/env python3
"""Fit an owned BPE on training text only, then pack each split exactly once.

The output is immutable. Rows contain sequence_len+1 distinct token positions;
there is no cycling, padding, document duplication, or shared boundary position.
The final incomplete row is discarded and counted. BOS delimits documents; the
model can attend across document boundaries, as in ordinary concatenated LM data.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
VENDOR = ROOT / "vendor" / ("nanochat-" + REVISION)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def normalized_hash(text):
    text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["text", "group_id", "source"]):
        yield from batch.to_pylist()


def inspect_splits(corpus):
    """Validate IDs and normalized exact leakage before any tokenizer fitting."""
    identities, counts = {}, {}
    for split in ("train", "val"):
        groups, texts, seen_groups = set(), set(), set()
        count = 0
        for row in rows(corpus / split / "shard_00000.parquet"):
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Empty/nontext document")
            if not isinstance(row["group_id"], str) or not row["group_id"]:
                raise ValueError("Every document needs a source group identity")
            h = normalized_hash(row["text"])
            if h in texts:
                raise ValueError("Normalized exact duplicate within " + split)
            if split == "val" and row["group_id"] in seen_groups:
                raise ValueError("Validation requires one document per source group")
            groups.add(row["group_id"]); seen_groups.add(row["group_id"]); texts.add(h)
            count += 1
        if not count:
            raise ValueError("Empty split: " + split)
        identities[split] = (groups, texts)
        counts[split] = {"documents": count, "source_groups": len(groups)}
    if identities["train"][0] & identities["val"][0]:
        raise ValueError("Training/validation source group overlap")
    if identities["train"][1] & identities["val"][1]:
        raise ValueError("Training/validation normalized exact text overlap")
    return counts


def import_tokenizer():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from nanochat.tokenizer import RustBPETokenizer
    return RustBPETokenizer


def save_portable_tokenizer(tokenizer, directory):
    # The upstream pickle is owned output for upstream compatibility. Our trainer
    # uses this plain JSON encoding, so it never has to deserialize a pickle.
    tokenizer.save(str(directory))
    enc = tokenizer.enc
    write_json(directory / "tokenizer.json", {
        "format": "udgam-owned-rustbpe-v1", "pattern": enc._pat_str,
        "mergeable_ranks": [[base64.b64encode(k).decode(), v]
                            for k, v in sorted(enc._mergeable_ranks.items(), key=lambda x: x[1])],
        "special_tokens": enc._special_tokens, "bos_token": "<|bos|>",
    })


def load_portable_tokenizer(directory):
    """Caller must first verify the preparation manifest/file hashes."""
    import tiktoken
    cls = import_tokenizer()
    data = json.loads((Path(directory) / "tokenizer.json").read_text())
    if data.get("format") != "udgam-owned-rustbpe-v1":
        raise ValueError("Unsupported tokenizer provenance")
    enc = tiktoken.Encoding(name="udgam-owned-rustbpe", pat_str=data["pattern"],
                            mergeable_ranks={base64.b64decode(k): v for k, v in data["mergeable_ranks"]},
                            special_tokens=data["special_tokens"])
    return cls(enc, data["bos_token"])


def pack_split(parquet, output, tokenizer, sequence_len):
    import numpy as np
    width = sequence_len + 1
    temporary = output.with_suffix(".tokens.tmp")
    source_counts, source_tokens, source_bytes = Counter(), Counter(), Counter()
    total = 0
    with temporary.open("xb") as f:
        for row in rows(parquet):
            ids = tokenizer.encode(row["text"], prepend=tokenizer.get_bos_token_id())
            if not ids or min(ids) < 0 or max(ids) >= tokenizer.get_vocab_size():
                raise ValueError("Invalid tokenizer result")
            np.asarray(ids, dtype="<u2").tofile(f)
            total += len(ids)
            source_counts[row["source"]] += 1
            source_tokens[row["source"]] += len(ids)
            source_bytes[row["source"]] += len(row["text"].encode())
    nrows = total // width
    if not nrows:
        temporary.unlink()
        raise ValueError("Not enough distinct positions to fill one row")
    raw = np.memmap(temporary, mode="r", dtype="<u2", shape=(total,))
    result = np.lib.format.open_memmap(output, mode="w+", dtype="<u2", shape=(nrows, width))
    for start in range(0, nrows, 2048):
        end = min(nrows, start + 2048)
        result[start:end] = raw[start * width:end * width].reshape(-1, width)
    result.flush(); del result; del raw
    temporary.unlink()
    return {"rows": nrows, "row_width": width, "encoded_positions": total,
            "packed_distinct_positions": nrows * width, "prediction_targets": nrows * sequence_len,
            "discarded_tail_positions": total % width, "wrapping": False,
            "source_documents": dict(source_counts), "source_encoded_positions": dict(source_tokens),
            "source_utf8_bytes": dict(source_bytes)}


REQUIRED_ARTIFACTS = {"train.npy", "val.npy", "token_bytes.npy", "tokenizer.json", "tokenizer.pkl",
                      "source_inputs.json", "source-report.json", "source-train-provenance.jsonl.gz",
                      "source-val-provenance.jsonl.gz"}


def verify_prepared(directory, *, production=False):
    import numpy as np
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    digest = (directory / "manifest.sha256").read_text().strip()
    if sha256(manifest_path) != digest:
        raise ValueError("Preparation manifest hash mismatch")
    data = json.loads(manifest_path.read_text())
    if data.get("format") != "udgam-dense-finite-data-v1" or data.get("status") != "ready":
        raise ValueError("Preparation is not complete")
    if production and data.get("profile") != "production":
        raise ValueError("Preflight data cannot be used for a production pilot")
    if data.get("source_revision") != REVISION or data.get("tokenizer_fit_split") != "train":
        raise ValueError("Tokenizer/source provenance mismatch")
    if not REQUIRED_ARTIFACTS <= set(data["hashes"]):
        raise ValueError("Missing required hashed artifact")
    for name, expected in data["hashes"].items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError("Unsafe artifact path")
        if sha256(path) != expected:
            raise ValueError("Prepared artifact hash mismatch: " + name)
    for split in ("train", "val"):
        arr = np.load(directory / (split + ".npy"), mmap_mode="r", allow_pickle=False)
        info = data["splits"][split]
        if arr.dtype != np.dtype("<u2") or list(arr.shape) != [info["rows"], data["sequence_len"] + 1]:
            raise ValueError("Packed array shape/dtype mismatch")
        if info["wrapping"] is not False or info["packed_distinct_positions"] != arr.size:
            raise ValueError("Finite token supply mismatch")
        if int(arr.max()) >= data["vocab_size"]:
            raise ValueError("Out of vocabulary token")
    token_bytes = np.load(directory / "token_bytes.npy", allow_pickle=False)
    if token_bytes.shape != (data["vocab_size"],) or token_bytes.dtype != np.int32 or (token_bytes < 0).any():
        raise ValueError("Invalid raw token byte table")
    return data, digest


def prepare(corpus, report_path, output, *, vocab_size=16384, sequence_len=2048,
            tokenizer_max_chars=200_000_000, tokenizer_doc_cap=10_000, profile="production"):
    import numpy as np
    corpus, report_path, output = map(Path, (corpus, report_path, output))
    if output.exists():
        raise ValueError("Output exists; use a new immutable path")
    if not 265 <= vocab_size <= 65535 or sequence_len < 2 or tokenizer_max_chars <= 0 or tokenizer_doc_cap <= 0:
        raise ValueError("Invalid tokenizer/packing bounds")
    if profile not in ("production", "tiny_cpu_test", "preflight"):
        raise ValueError("Invalid preparation profile")
    if profile == "production" and (vocab_size != 16384 or sequence_len != 2048):
        raise ValueError("Production candidate requires 16K vocabulary and context 2048")
    report = json.loads(report_path.read_text())
    if profile == "production" and (report.get("status") != "corpus_prepared_token_supply_unmeasured" or report.get("errors")):
        raise ValueError("Production source preparation is incomplete")
    expected_files = {f"{split}/{name}" for split in ("train", "val") for name in ("shard_00000.parquet", "provenance.jsonl.gz")}
    if not expected_files <= set(report.get("hashes", {})):
        raise ValueError("Source report lacks parquet/provenance hashes")
    for name in expected_files:
        if sha256(corpus / name) != report["hashes"][name]:
            raise ValueError("Source hash mismatch: " + name)
    if profile == "production" and not report.get("benchmark_manifest_sha256"):
        raise ValueError("Frozen benchmark evidence missing")
    split_counts = inspect_splits(corpus)
    cls = import_tokenizer()
    fit = {"documents": 0, "characters": 0}

    def train_text():
        for row in rows(corpus / "train/shard_00000.parquet"):
            remaining = tokenizer_max_chars - fit["characters"]
            if remaining <= 0:
                break
            text = row["text"][:min(tokenizer_doc_cap, remaining)]
            fit["documents"] += 1; fit["characters"] += len(text)
            yield text

    tokenizer = cls.train_from_iterator(train_text(), vocab_size)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("Insufficient training diversity to fit requested vocabulary")
    for text in ("Hello world!", "नमस्ते भारत", "0 12 345\nUnicode: café 🎉"):
        if tokenizer.decode(tokenizer.encode(text)) != text:
            raise ValueError("Tokenizer byte roundtrip failed")
    output.mkdir(parents=True)
    save_portable_tokenizer(tokenizer, output)
    special = {tokenizer.encode_special(t) for t in tokenizer.get_special_tokens()}
    token_bytes = np.array([0 if i in special else len(tokenizer.enc.decode_single_token_bytes(i))
                            for i in range(vocab_size)], dtype=np.int32)
    np.save(output / "token_bytes.npy", token_bytes, allow_pickle=False)
    shutil.copyfile(report_path, output / "source-report.json")
    for split in ("train", "val"):
        shutil.copyfile(corpus / split / "provenance.jsonl.gz", output / f"source-{split}-provenance.jsonl.gz")
    write_json(output / "source_inputs.json", {"source_report_sha256": sha256(report_path),
               "source_file_sha256": {name: report["hashes"][name] for name in sorted(expected_files)},
               "split_identity_checks": split_counts, "group_overlap": 0, "normalized_exact_overlap": 0})
    splits = {split: pack_split(corpus / split / "shard_00000.parquet", output / (split + ".npy"),
                                 tokenizer, sequence_len) for split in ("train", "val")}
    import importlib.metadata
    manifest = {"format": "udgam-dense-finite-data-v1", "status": "ready", "profile": profile,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_revision": REVISION,
                "script_sha256": sha256(Path(__file__)), "vocab_size": vocab_size, "sequence_len": sequence_len,
                "tokenizer_fit_split": "train", "tokenizer_fit": fit, "tokenizer_max_chars": tokenizer_max_chars,
                "tokenizer_doc_cap": tokenizer_doc_cap, "packing": "concatenation_BOS_nonoverlapping_rows_tail_discarded",
                "splits": splits, "hashes": {name: sha256(output / name) for name in sorted(REQUIRED_ARTIFACTS)},
                "versions": {p: importlib.metadata.version(p) for p in ("rustbpe", "tiktoken", "numpy", "pyarrow")},
                "quality_measured": False}
    write_json(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(sha256(output / "manifest.json") + "\n")
    verify_prepared(output, production=profile == "production")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--sequence-len", type=int, default=2048)
    p.add_argument("--tokenizer-max-chars", type=int, default=200_000_000)
    p.add_argument("--tokenizer-doc-cap", type=int, default=10_000)
    p.add_argument("--profile", choices=["production", "preflight", "tiny_cpu_test"], default="production")
    a = p.parse_args()
    result = prepare(a.corpus, a.source_report, a.output, vocab_size=a.vocab_size, sequence_len=a.sequence_len,
                     tokenizer_max_chars=a.tokenizer_max_chars, tokenizer_doc_cap=a.tokenizer_doc_cap, profile=a.profile)
    print(json.dumps({"status": result["status"], "profile": result["profile"], "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
