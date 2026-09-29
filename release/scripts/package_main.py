#!/usr/bin/env python3
"""Allowlisted main upload: small hash manifest, streaming tar, remote verification.

Run manifest only AFTER acquisition closes. It takes the builder's process.lock
and checkpoints SQLite WAL in place; no second database/archive copy is made.
Streaming also holds that lock and verifies every byte against the manifest.
No provider calls, subprocesses, credentials, controller files or budget files.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import sys
import tarfile
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
VENDOR = "vendor/nanochat-" + REVISION
STAGE = "data/english-main-001"
PILOT = "data/dense-tokens-production-001"
RUN = "runs/pilot-001"
SMOKE = "data/benchmark-smoke-001"
TASK_ROOT = "research/harness-task-definitions"
FORMAT = "udgam-main-upload-v1"
TASKS = ("arc_challenge", "hellaswag", "piqa", "boolq", "winogrande")
RUNTIME = tuple("scripts/" + name for name in (
    "train_dense.py", "train_main.py", "prepare_dense_tokens.py", "bootstrap_gpu.py",
    "prepare_main.py", "prepare_english.py", "prepare_production.py", "acquire_main_cached.py",
    "main_range_cache.py", "acquire_main_batched.py", "main_batch_engine.py", "pack_main_bounded.py",
    "acquire_main_scanned.py", "main_scanned_store.py", "pack_main_scanned.py",
    "produce_main.py", "evaluate_dense.py", "infer_dense.py", "package_main.py", "backup_main.py"))
CONFIGS = ("configs/gpu-requirements.txt", "configs/nanochat_dense_reference.json", "configs/dense_pilot.json",
    "configs/main_run.json", "configs/inference/requirements.txt", "configs/inference/README.md",
    "configs/inference/LICENSE.inference")
VENDOR_FILES = ("LICENSE", "README.md", "pyproject.toml") + tuple("nanochat/" + name + ".py" for name in (
    "__init__", "checkpoint_manager", "common", "core_eval", "dataloader", "dataset", "engine",
    "execution", "flash_attention", "fp8", "gpt", "loss_eval", "optim", "tokenizer"))
PILOT_FILES = ("manifest.json", "manifest.sha256", "train.npy", "val.npy", "token_bytes.npy",
    "tokenizer.json", "tokenizer.pkl", "source_inputs.json", "source-report.json",
    "source-train-provenance.jsonl.gz", "source-val-provenance.jsonl.gz")
TASK_FILES = ("manifest.json", "commit.json", "lm_eval/api/task.py", "lm_eval/api/model.py",
    "lm_eval/config/task.py", "lm_eval/models/huggingface.py", "lm_eval/tasks/arc/arc_challenge.yaml",
    "lm_eval/tasks/arc/arc_easy.yaml", "lm_eval/tasks/hellaswag/hellaswag.yaml", "lm_eval/tasks/hellaswag/utils.py",
    "lm_eval/tasks/piqa/piqa.yaml", "lm_eval/tasks/super_glue/boolq/default.yaml",
    "lm_eval/tasks/winogrande/default.yaml", "lm_eval/tasks/winogrande/preprocess_winogrande.py")
TRANSPORT_FILES = ("plan.json", "prepare_main.py", "prepare_english.py", "prepare_production.py",
                   "acquire_main_cached.py", "main_range_cache.py")
TRANSPORT_V2_FILES = TRANSPORT_FILES + ("acquire_main_batched.py", "main_batch_engine.py", "pack_main_bounded.py")
TRANSPORT_V3_FILES = TRANSPORT_V2_FILES + ("acquire_main_scanned.py", "main_scanned_store.py", "pack_main_scanned.py")
STAGE_FIXED = ("state.sqlite", "manifest.json", "manifest.sha256", "tokenizer.json", "source-lock.json",
    "source-files.jsonl.gz", "benchmark-selection.json", "network.jsonl", "frozen/val.npy",
    "frozen/token_bytes.npy", "frozen/benchmarks/manifest.json")
STATIC = set(RUNTIME + CONFIGS + ("LICENSE", "LICENSE_SCOPE.md", "research/data_sources.json"))
STATIC.update(VENDOR + "/" + name for name in VENDOR_FILES)
STATIC.update(PILOT + "/" + name for name in PILOT_FILES)
STATIC.update(RUN + "/" + name for name in ("checkpoint.pt", "checkpoint.sha256.json", "result.json"))
STATIC.update(TASK_ROOT + "/" + name for name in TASK_FILES)
STATIC.update(SMOKE + "/" + name for name in ("manifest.json", "manifest.sha256"))
STATIC.update(SMOKE + "/" + name + ".train.jsonl.gz" for name in TASKS)
STATIC.update(STAGE + "/" + name for name in STAGE_FIXED)
STATIC.update(STAGE + "/transport/" + name for name in TRANSPORT_FILES)


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or any(ord(c) < 32 for c in relative):
        raise ValueError("Unsafe upload path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or rel.as_posix() != relative or any(p in {"", ".", ".."} or p.startswith(".") for p in rel.parts):
        raise ValueError("Unsafe upload path")
    path = Path(root) / relative
    current = Path(root)
    for part in rel.parts:
        current = current / part
        if current.is_symlink(): raise ValueError("Symlink in upload path")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_nlink != 1:
        raise ValueError("Upload input must be an unlinked regular file")
    return path


def allowed(relative):
    if relative in STATIC: return True
    if relative in {STAGE + "/transport_v2/" + name for name in TRANSPORT_V2_FILES}: return True
    if relative in {STAGE + "/transport_v3/" + name for name in TRANSPORT_V3_FILES}: return True
    patterns = (STAGE + r"/shards/train_[0-9]{5}\.(?:npy|provenance\.jsonl\.gz)",
        STAGE + r"/transport/runs/[0-9]{5}/(?:start|finish)\.json",
        STAGE + r"/frozen/benchmarks/(?:" + "|".join(TASKS) + r")(?:\.(?:test|validation))?\.jsonl\.gz")
    return any(re.fullmatch(pattern, relative) for pattern in patterns)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def read_json(path, limit=8 * 1024 * 1024):
    if path.stat().st_size > limit: raise ValueError("Metadata file exceeds reviewed size")
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError("Expected JSON object")
    return value


@contextmanager
def stage_lock(root):
    path = Path(root) / STAGE / "process.lock"
    current = Path(root)
    for part in PurePosixPath(STAGE).parts:
        current = current / part
        if current.is_symlink() or not current.is_dir(): raise ValueError("Unsafe stage directory")
    if path.is_symlink(): raise ValueError("Unsafe stage lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "a") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise ValueError("Unsafe stage lock")
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("Acquisition is still active; close it before packaging") from None
        yield


def inspect_stage(root, *, checkpoint=False):
    directory = Path(root) / STAGE
    database = safe_path(root, STAGE + "/state.sqlite")
    if (directory / "packing.json").exists(): raise ValueError("Unfinished packing requires recovery before packaging")
    if checkpoint:
        # Only invoked after taking the same exclusive process lock as the builder.
        connection = sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=0)
        try:
            result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result[0] != 0: raise ValueError("SQLite WAL checkpoint is busy")
        finally: connection.close()
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size: raise ValueError("Uncheckpointed WAL must not be omitted")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",): raise ValueError("SQLite integrity check failed")
        meta = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM meta")}
        if meta.get("status") != "initialized" or meta.get("network_inflight") is not False:
            raise ValueError("Main stage is incomplete or has unreconciled network activity")
        if meta.get("benchmark_dir") != "frozen/benchmarks" or meta["lineage"].get("frozen_validation_path") != "frozen/val.npy" or meta["lineage"].get("token_bytes_path") != "frozen/token_bytes.npy":
            raise ValueError("Main frozen inputs must use portable internal paths")
        stage = read_json(safe_path(root, STAGE + "/manifest.json"))
        if file_hash(directory / "manifest.json") != (directory / "manifest.sha256").read_text().strip():
            raise ValueError("Main manifest checksum mismatch")
        shards = [json.loads(row[0]) for row in connection.execute("SELECT manifest FROM shards ORDER BY id")]
        if not shards or stage.get("shards") != shards or stage.get("lineage") != meta["lineage"] or stage.get("config") != meta["config"]:
            raise ValueError("Missing committed supply or stale main manifest")
        if stage.get("status") != "initialized" or stage.get("network_request_unreconciled") is not False:
            raise ValueError("Stage manifest is not clean")
        if stage.get("downloaded_response_bytes") != meta.get("downloaded_bytes", 0): raise ValueError("Stale download accounting in stage manifest")
        for number, shard in enumerate(shards):
            if shard.get("id") != number or set(shard["hashes"]) != {f"shards/train_{number:05d}.npy", f"shards/train_{number:05d}.provenance.jsonl.gz"}:
                raise ValueError("Noncontiguous or unrecognized committed shards")
        cursor = hashlib.sha256(); job_count = 0
        for row in connection.execute("SELECT file_id,rg,cursor,done FROM jobs ORDER BY file_id,rg"):
            cursor.update(json.dumps(list(row), separators=(",", ":")).encode() + b"\n"); job_count += 1
        receipts, epochs = {}, {}
        for kind, detail in connection.execute("SELECT kind,detail FROM events WHERE kind IN ('cached_acquisition_started','cached_acquisition_finished') ORDER BY id"):
            event = json.loads(detail); phase = "start" if kind.endswith("started") else "finish"
            number = event["id"]
            if type(number) is not int or not 1 <= number <= 99999: raise ValueError("Invalid transport run number")
            expected = f"transport/runs/{number:05d}/{phase}.json"
            if event["record"] != expected or (number, phase) in epochs: raise ValueError("Invalid transport receipt identity")
            epochs[(number, phase)] = True; receipts[expected] = event["sha256"]
        sequence = meta.get("cached_transport_run_sequence", 0)
        if type(sequence) is not int or sequence < 1 or set(epochs) != {(n, p) for n in range(1, sequence + 1) for p in ("start", "finish")}:
            raise ValueError("Every cached acquisition start needs a durable finish receipt")
        last_finish = read_json(safe_path(root, STAGE + f"/transport/runs/{sequence:05d}/finish.json"))
        if last_finish.get("outcome") not in {"graceful_stop", "tranche_stopped_at_target_or_document_boundary"}:
            raise ValueError("Last acquisition failed; review and close it cleanly before packaging")
        return stage, receipts, {"sqlite_quick_check": "ok", "wal_checkpointed": True,
            "jobs": job_count, "job_cursors_sha256": cursor.hexdigest(), "transport_runs": sequence,
            "committed_shards": len(shards), "pending_documents_preserved_in_sqlite": True,
            "packing_incomplete": meta.get("packing_incomplete", False),
            "remote_initial_pack_required": meta.get("packing_incomplete", False)}
    finally: connection.close()


def select(root, stage, receipts):
    selected, expected = set(STATIC), {}
    lineage = stage["lineage"]
    expected.update({STAGE + "/" + p: h for p, h in lineage["metadata_artifact_hashes"].items()})
    expected[STAGE + "/tokenizer.json"] = lineage["tokenizer_sha256"]
    for shard in stage["shards"]:
        expected.update({STAGE + "/" + p: h for p, h in shard["hashes"].items()})
    expected.update({STAGE + "/" + p: h for p, h in receipts.items()})
    expected["scripts/prepare_main.py"] = lineage["builder_sha256"]
    expected.update({"scripts/" + p: h for p, h in lineage["helper_sha256"].items()})
    active = lineage["active_transport"]
    if active.get("driver") in {"cached-batched-main-v2", "cached-scanned-main-v3"}:
        v3 = active["driver"] == "cached-scanned-main-v3"
        expected.update({"scripts/acquire_main_batched.py": active["v2_driver_sha256"] if v3 else active["driver_sha256"],
            "scripts/main_batch_engine.py": active["engine_sha256"],
            "scripts/pack_main_bounded.py": active["packer_sha256"],
            "scripts/acquire_main_cached.py": active["transport_driver_sha256"]})
        plan_path = "transport_v3/plan.json" if v3 else "transport_v2/plan.json"
        previous = lineage.get("previous_active_transports", [])
        expected_history = ["cached-main-acquisition-v1", "cached-batched-main-v2"] if v3 else ["cached-main-acquisition-v1"]
        if [x["active_transport"].get("driver") for x in previous] != expected_history:
            raise ValueError("Missing original transport history")
        original = previous[0]["active_transport"]
        if original["plan_path"] != "transport/plan.json" or original["plan_sha256"] != expected.get(STAGE + "/transport/plan.json"):
            raise ValueError("Original transport plan identity mismatch")
        if v3:
            prior = previous[1]["active_transport"]
            if prior["plan_path"] != "transport_v2/plan.json" or prior["plan_sha256"] != expected.get(STAGE + "/transport_v2/plan.json"):
                raise ValueError("V2 transport plan identity mismatch")
            expected.update({"scripts/acquire_main_scanned.py": active["driver_sha256"],
                "scripts/main_scanned_store.py": active["scanned_store_sha256"],
                "scripts/pack_main_scanned.py": active["scanned_packer_sha256"]})
    elif active.get("driver") == "cached-main-acquisition-v1":
        expected["scripts/acquire_main_cached.py"] = active["driver_sha256"]
        plan_path = "transport/plan.json"
    else:
        raise ValueError("Unrecognized active transport")
    expected["scripts/main_range_cache.py"] = active["cache_sha256"]
    if active["plan_path"] != plan_path or active["plan_sha256"] != expected.get(STAGE + "/" + plan_path):
        raise ValueError("Transport plan identity mismatch")
    for transport in [x["active_transport"] for x in lineage.get("previous_active_transports", [])] + [active]:
        for path, checksum in transport["source_snapshot_hashes"].items():
            if expected.get(STAGE + "/" + path) != checksum: raise ValueError("Transport snapshot identity mismatch")
    pilot = read_json(safe_path(root, PILOT + "/manifest.json"))
    if file_hash(Path(root) / PILOT / "manifest.json") != (Path(root) / PILOT / "manifest.sha256").read_text().strip(): raise ValueError("Pilot manifest checksum mismatch")
    if pilot.get("source_revision") != REVISION or pilot.get("profile") != "production" or pilot.get("status") != "ready": raise ValueError("Owned production pilot token data required")
    expected.update({PILOT + "/" + p: h for p, h in pilot["hashes"].items()})
    if lineage["pilot_manifest_sha256"] != file_hash(Path(root) / PILOT / "manifest.json"): raise ValueError("Pilot/main lineage mismatch")
    side = read_json(safe_path(root, RUN + "/checkpoint.sha256.json"))
    if side.get("format") != "udgam-owned-dense-scratch-v1" or side.get("data_manifest_sha256") != lineage["pilot_manifest_sha256"]: raise ValueError("Owned pilot checkpoint lineage mismatch")
    expected[RUN + "/checkpoint.pt"] = side["sha256"]
    smoke = read_json(safe_path(root, SMOKE + "/manifest.json"))
    if smoke.get("format") != "udgam-benchmark-development-smoke-v1" or set(smoke["tasks"]) != set(TASKS): raise ValueError("Reviewed development smoke required")
    if file_hash(Path(root) / SMOKE / "manifest.json") != (Path(root) / SMOKE / "manifest.sha256").read_text().strip(): raise ValueError("Smoke checksum mismatch")
    for task, item in smoke["tasks"].items():
        if item["file"] != task + ".train.jsonl.gz": raise ValueError("Unexpected development file")
        expected[SMOKE + "/" + item["file"]] = item["sha256"]
    task_manifest = read_json(safe_path(root, TASK_ROOT + "/manifest.json"))
    if task_manifest.get("revision") != "d6de81643928d653435c431bae19945d41d32520": raise ValueError("Wrong task definition revision")
    expected.update({TASK_ROOT + "/" + p: h for p, h in task_manifest["files"].items()})
    selected.update(expected)
    if any(not allowed(path) for path in selected): raise ValueError("Metadata requested a file outside the explicit upload allowlist")
    committed = {Path(path).name for path in selected if path.startswith(STAGE + "/shards/")}
    if {p.name for p in (Path(root) / STAGE / "shards").iterdir()} != committed: raise ValueError("Uncommitted or unexpected shard files require review")
    return selected, expected


def check_network_log(root):
    # The acquisition code records original public source URLs, never headers.
    # Refuse accidental credential-bearing URLs/fields rather than sanitizing evidence.
    with safe_path(root, STAGE + "/network.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if any(key.lower() in {"authorization", "api_key", "access_token", "password", "cookie", "headers"} for key in record): raise ValueError("Credential field in source receipt")
            url = urlsplit(record.get("url", ""))
            if url.scheme != "https" or url.hostname != "huggingface.co" or url.username or url.password or url.fragment or set(parse_qs(url.query, keep_blank_values=True)) - {"limit", "recursive", "expand", "cursor"}:
                raise ValueError("Unexpected or credential-bearing source receipt URL")


def inventory(root, stage, receipts):
    selected, expected = select(root, stage, receipts)
    check_network_log(root)
    result = []
    for relative in sorted(selected):
        path = safe_path(root, relative)
        before = path.stat()
        checksum = file_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino): raise ValueError("Input changed while hashing")
        if relative in expected and checksum != expected[relative]: raise ValueError("Artifact hash mismatch: " + relative)
        result.append({"path": relative, "size_bytes": after.st_size, "sha256": checksum})
    return result


def build_manifest(root, output_relative):
    root = Path(root).resolve()
    if not re.fullmatch(r"records/main-upload-manifest-[A-Za-z0-9_-]+\.json", output_relative): raise ValueError("Explicit new main manifest path required")
    output = root / output_relative
    if output.exists() or output.is_symlink() or output.parent.is_symlink(): raise ValueError("Refusing manifest overwrite")
    with stage_lock(root):
        stage, receipts, state = inspect_stage(root, checkpoint=True)
        files = inventory(root, stage, receipts)
        # Bind original private closeout evidence without copying it to the Pod.
        original = read_json(safe_path(root, "records/pilot-upload-manifest.json"))
        for item in files:
            if item["path"].startswith(VENDOR + "/") and original.get(item["path"], {}).get("sha256") != item["sha256"]:
                raise ValueError("Pinned vendor differs from the original pilot upload")
        backup = read_json(safe_path(root, "records/pilot-backup-001.json"))
        owned = [item for item in backup.get("artifacts", []) if item.get("role") == "checkpoint"]
        checkpoint = next(item for item in files if item["path"] == RUN + "/checkpoint.pt")
        if backup.get("backup_verified") is not True or len(owned) != 1 or owned[0]["sha256"] != checkpoint["sha256"] or owned[0]["size_bytes"] != checkpoint["size_bytes"]:
            raise ValueError("Checkpoint differs from verified pilot backup")
        value = {"format": FORMAT, "manifest_path": output_relative, "vendor_revision": REVISION,
            "layout": {"stage": STAGE, "pilot_data": PILOT, "pilot_checkpoint": RUN + "/checkpoint.pt", "development_smoke": SMOKE},
            "sqlite_snapshot": state, "files": files, "total_bytes": sum(item["size_bytes"] for item in files),
            "pilot_upload_manifest_sha256": file_hash(root / "records/pilot-upload-manifest.json"),
            "pilot_backup_manifest_sha256": file_hash(root / "records/pilot-backup-001.json"),
            "contains_provider_credentials": False, "contains_provider_controller": False,
            "transport": "uncompressed streaming tar; no local archive"}
        with output.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return value


def verify_manifest(root, manifest):
    if manifest.get("format") != FORMAT or manifest.get("vendor_revision") != REVISION: raise ValueError("Unrecognized main upload manifest")
    if not re.fullmatch(r"records/main-upload-manifest-[A-Za-z0-9_-]+\.json", manifest.get("manifest_path", "")): raise ValueError("Unsafe manifest path")
    stage, receipts, state = inspect_stage(root)
    if state != manifest["sqlite_snapshot"] or inventory(root, stage, receipts) != manifest["files"]:
        raise ValueError("Upload snapshot changed; create a new reviewed manifest")
    if sum(item["size_bytes"] for item in manifest["files"]) != manifest["total_bytes"]: raise ValueError("Upload size mismatch")
    return True


class HashReader:
    def __init__(self, handle): self.handle, self.digest = handle, hashlib.sha256()
    def read(self, size=-1):
        data = self.handle.read(size); self.digest.update(data); return data


def stream(root, manifest_path, destination):
    root = Path(root).resolve()
    relative = Path(manifest_path).absolute().relative_to(root).as_posix()
    if not re.fullmatch(r"records/main-upload-manifest-[A-Za-z0-9_-]+\.json", relative): raise ValueError("Unsafe manifest path")
    manifest = read_json(safe_path(root, relative))
    if (root / manifest["manifest_path"]).resolve() != Path(manifest_path).resolve(): raise ValueError("Manifest outside selected root")
    with stage_lock(root):
        verify_manifest(root, manifest)  # Emit no tar bytes if any preflight check fails.
        items = manifest["files"] + [{"path": manifest["manifest_path"], "size_bytes": Path(manifest_path).stat().st_size, "sha256": file_hash(Path(manifest_path))}]
        with tarfile.open(fileobj=destination, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for item in items:
                path = safe_path(root, item["path"])
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as handle:
                    before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != item["size_bytes"]: raise ValueError("Input replaced before streaming")
                    reader = HashReader(handle)
                    info = tarfile.TarInfo(item["path"]); info.size = item["size_bytes"]; info.mode = 0o644; info.mtime = 0
                    archive.addfile(info, reader)
                    after = os.fstat(handle.fileno())
                    if reader.digest.hexdigest() != item["sha256"] or before.st_mtime_ns != after.st_mtime_ns: raise ValueError("Input changed during streaming; discard remote partial extraction")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["manifest", "stream", "verify"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", default="records/main-upload-manifest-001.json")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"records/main-upload-manifest-[A-Za-z0-9_-]+\.json", args.manifest): raise ValueError("Unsafe manifest path")
    if args.action == "manifest":
        result = build_manifest(args.root, args.manifest)
        print(json.dumps({"manifest": result["manifest_path"], "files": len(result["files"]), "total_bytes": result["total_bytes"]}))
    elif args.action == "stream":
        stream(args.root, args.root / args.manifest, sys.stdout.buffer)
    else:
        manifest = read_json(safe_path(args.root, args.manifest))
        with stage_lock(args.root): verify_manifest(args.root, manifest)
        print(json.dumps({"verified": True, "files": len(manifest["files"]), "total_bytes": manifest["total_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
