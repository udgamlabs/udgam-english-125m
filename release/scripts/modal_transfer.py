#!/usr/bin/env python3
"""Offline, exact-manifest Modal transfer helpers; no SDK or provider calls.

The Volume is immutable transfer storage. Materialize into a fresh, local POSIX
filesystem before using SQLite/locks. A finite run has no producer; its stopped
trainer and unchanged sealed input form the recovery contract.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys

BASE_PATH = "records/main-upload-manifest-002.json"
BASE_SHA = "9fa21fa63df6ffa2364d22f97d9d2a95b107f86e7a60118bbd1ca9a3bd3f23ba"
ADAPTER = "scripts/train_main_modal.py"
TRANSFER = "scripts/modal_transfer.py"
DELTA_FORMAT = "udgam-modal-finite-delta-v1"
RUN_FILES = frozenset({"checkpoint.pt", "checkpoint.sha256.json", "result.json",
    "run-contract.json", "loss-history.jsonl", "failure.json", "execution.json",
    "stdout.log", "train.log", "checkpoint.recovery.pt", "checkpoint.recovery.sha256.json"})
REQUIRED_RUN = frozenset({"checkpoint.pt", "checkpoint.sha256.json", "result.json", "run-contract.json"})
CHECKPOINT_FILES = frozenset({"checkpoint.pt", "checkpoint.sha256.json", "checkpoint.recovery.pt", "checkpoint.recovery.sha256.json"})
TRANSIENT_FILES = frozenset({"checkpoint.tmp", ".checkpoint-recovery.pt.pending", ".checkpoint-recovery.sha256.json.pending"})
SECRET_KEYS = frozenset({"authorization", "apikey", "accesstoken", "password", "cookie",
    "modaltokenid", "modaltokensecret", "runpodapikey", "secretaccesskey"})
SECRET_BYTES = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:rp_|sk-)[A-Za-z0-9_-]{20,}")


@dataclass(frozen=True)
class Seal:
    path: str = BASE_PATH
    sha256: str = BASE_SHA
    files: int = 203
    total_bytes: int = 3300571068


def relative(name):
    if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 for c in name):
        raise ValueError("Unsafe relative path")
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(x in {"", ".", ".."} or x.startswith(".") for x in p.parts):
        raise ValueError("Unsafe relative path")
    if p.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
        raise ValueError("Credential file forbidden")
    return name


def root_path(root):
    p = Path(root).absolute()
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("Symlink root/ancestor forbidden")
    if not p.is_dir(): raise ValueError("Root directory missing")
    return p


def regular(root, name):
    relative(name)
    p = root
    for part in PurePosixPath(name).parts:
        p /= part
        if p.is_symlink(): raise ValueError("Symlink input forbidden")
    info = p.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Input must be a non-hardlinked regular file")
    return p


def digest(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def describe(root, name):
    p = regular(root, name); before = p.stat(); checksum = digest(p); after = p.stat()
    if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during hashing")
    return {"path": name, "size_bytes": after.st_size, "sha256": checksum}


def read_json(path):
    if path.stat().st_size > 8 * 1024 * 1024: raise ValueError("Metadata too large")
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError("Expected JSON object")
    return value


def check_items(items):
    if not isinstance(items, list) or not 1 <= len(items) <= 1000: raise ValueError("Bad inventory")
    names = set()
    for item in items:
        if set(item) != {"path", "size_bytes", "sha256"}: raise ValueError("Bad inventory fields")
        name = relative(item["path"])
        if name in names: raise ValueError("Duplicate inventory path")
        names.add(name)
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Bad size/hash")
    return names


def exact_tree(root, names):
    found = set()
    for parent, dirs, files in os.walk(root, followlinks=False):
        for part in dirs:
            p = Path(parent) / part
            if p.is_symlink(): raise ValueError("Symlink directory forbidden")
        for part in files:
            name = (Path(parent) / part).relative_to(root).as_posix()
            if name not in names: raise ValueError("Unexpected transfer file: " + name)
            regular(root, name); found.add(name)
    if found != names: raise ValueError("Missing transfer file")


def input_inventory(root, *, strict=False, seal=Seal()):
    root = root_path(root); manifest_path = regular(root, seal.path)
    if digest(manifest_path) != seal.sha256: raise ValueError("Sealed input manifest hash mismatch")
    m = read_json(manifest_path); names = check_items(m.get("files"))
    if (m.get("format") != "udgam-main-upload-v1" or m.get("manifest_path") != seal.path or
            len(names) != seal.files or m.get("total_bytes") != seal.total_bytes or
            sum(x["size_bytes"] for x in m["files"]) != seal.total_bytes or
            seal.path in names or m.get("contains_provider_credentials") is not False or
            m.get("contains_provider_controller") is not False):
        raise ValueError("Sealed input contract mismatch")
    for item in m["files"]:
        if describe(root, item["path"]) != item: raise ValueError("Sealed input file mismatch: " + item["path"])
    files = sorted(m["files"] + [describe(root, seal.path)], key=lambda x: x["path"])
    if strict: exact_tree(root, names | {seal.path})
    return {"format": "udgam-modal-upload-inventory-v1", "base_manifest_path": seal.path,
            "base_manifest_sha256": seal.sha256, "files": files,
            "total_bytes": sum(x["size_bytes"] for x in files)}


def copy_inventory(source, destination, items, *, reserve_bytes, check_free):
    source = root_path(source); destination = Path(destination).absolute()
    parent = root_path(destination.parent)
    if destination.exists() or destination.is_symlink(): raise FileExistsError("Fresh destination required")
    check_items(items)
    if type(reserve_bytes) is not int or reserve_bytes < 0: raise ValueError("Invalid reserve")
    remaining = sum(i["size_bytes"] for i in items)
    if check_free and shutil.disk_usage(parent).free < reserve_bytes + remaining:
        raise ValueError("Insufficient local filesystem headroom")
    destination.mkdir()
    # On failure preserve only this fresh partial directory for inspection. It
    # lacks a ready receipt and must never be treated as a verified transfer.
    for item in items:
        src = regular(source, item["path"]); before = src.stat()
        if before.st_size != item["size_bytes"]: raise ValueError("Source size changed before copying")
        out = destination / item["path"]; out.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(); size = 0
        with src.open("rb") as f, out.open("xb") as g:
            while block := f.read(4 * 1024 * 1024):
                if size + len(block) > item["size_bytes"]: raise ValueError("Source grew during copying")
                if check_free and shutil.disk_usage(parent).free < reserve_bytes + len(block):
                    raise ValueError("Local filesystem reserve reached")
                g.write(block); h.update(block); size += len(block)
            g.flush(); os.fsync(g.fileno())
        after = src.stat()
        if (size != item["size_bytes"] or h.hexdigest() != item["sha256"] or
                (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("Input changed or copied file mismatch")
        if describe(destination, item["path"]) != item: raise ValueError("Destination hash mismatch")
    return destination


def materialize(source, destination, *, reserve_bytes=8 * 1024**3, seal=Seal()):
    inventory = input_inventory(source, strict=True, seal=seal)
    result = copy_inventory(source, destination, inventory["files"], reserve_bytes=reserve_bytes, check_free=True)
    input_inventory(result, strict=True, seal=seal)
    return {"materialized": True, "destination": str(result), "base_manifest_sha256": seal.sha256,
            "files": len(inventory["files"]), "total_bytes": inventory["total_bytes"]}


def check_json_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if re.sub(r"[^a-z0-9]", "", key.lower()) in SECRET_KEYS: raise ValueError("Credential field forbidden")
            check_json_secrets(item)
    elif isinstance(value, list):
        for item in value: check_json_secrets(item)


def scan_record(path):
    if path.suffix in {".json", ".jsonl"} and path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Structured output too large")
    tail = b""
    with path.open("rb") as f:
        while block := f.read(65536):
            if SECRET_BYTES.search(tail + block): raise ValueError("Credential-looking output forbidden")
            tail = block[-512:]
    if path.suffix == ".json": check_json_secrets(read_json(path))
    elif path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                if line.strip(): check_json_secrets(json.loads(line))


def stopped(pid):
    if type(pid) is not int or pid <= 1 or pid == os.getpid(): raise ValueError("Actual stopped trainer PID required")
    try: os.kill(pid, 0)
    except ProcessLookupError: return
    except PermissionError: raise ValueError("Cannot verify trainer stopped") from None
    raise ValueError("Trainer still present; wait/reap it before export")


def delta_paths(run, execution):
    if not re.fullmatch(r"runs/main-modal-[A-Za-z0-9_-]+", run): raise ValueError("Reviewed Modal run directory required")
    if not re.fullmatch(r"records/modal-execution-[A-Za-z0-9_-]+\.json", execution): raise ValueError("Explicit Modal execution record required")
    return {run + "/" + x for x in RUN_FILES} | {ADAPTER, TRANSFER, execution}


def verify_pairs(root, paths, base, base_root):
    base_items = {x["path"]: x for x in base["files"]}
    pilot_side = read_json(regular(root_path(base_root), 'runs/pilot-001/checkpoint.sha256.json'))
    initial = pilot_side.get('initial_weights_sha256')
    if (pilot_side.get('format') != 'udgam-owned-dense-scratch-v1' or
            pilot_side.get('sha256') != base_items['runs/pilot-001/checkpoint.pt']['sha256'] or
            pilot_side.get('data_manifest_sha256') != base_items['data/dense-tokens-production-001/manifest.json']['sha256'] or
            not isinstance(initial, str) or not re.fullmatch(r'[a-f0-9]{64}', initial)):
        raise ValueError('Sealed pilot initialization identity mismatch')
    for name in sorted(x for x in paths if x.endswith(".pt")):
        side_name = name.removesuffix(".pt") + ".sha256.json"
        if side_name not in paths: raise ValueError("Checkpoint pair incomplete")
        side = read_json(regular(root, side_name))
        if (side.get("format") != "udgam-owned-main-scratch-v1" or side.get("sha256") != digest(regular(root, name)) or
                side.get('initial_weights_sha256') != initial or
                side.get("pilot_checkpoint_sha256") != base_items["runs/pilot-001/checkpoint.pt"]["sha256"] or
                side.get("pilot_data_manifest_sha256") != base_items["data/dense-tokens-production-001/manifest.json"]["sha256"]):
            raise ValueError("Owned main checkpoint/parent mismatch")
    for name in (x for x in paths if x.endswith(".sha256.json")):
        if name.removesuffix(".sha256.json") + ".pt" not in paths: raise ValueError("Orphan checkpoint sidecar")
    return initial


def verify_standard_records(root, run, execution_name, base, initial, seal):
    """Cross-bind success evidence without importing torch or deserializing weights."""
    items = {x['path']: x for x in base['files']}
    execution = read_json(regular(root, execution_name))
    contract = read_json(regular(root, run + '/run-contract.json'))
    result = read_json(regular(root, run + '/result.json'))
    side = read_json(regular(root, run + '/checkpoint.sha256.json'))
    adapter_hash = digest(regular(root, ADAPTER))
    stage_hash = items['data/english-main-001/manifest.json']['sha256']
    parent_hash = items['runs/pilot-001/checkpoint.pt']['sha256']
    core_hashes = {name: items['scripts/' + name]['sha256'] for name in
                   ('train_main.py', 'train_dense.py', 'prepare_dense_tokens.py')}
    if (execution.get('format') != 'udgam-modal-execution-v1' or execution.get('profile') != 'modal-v1' or
            execution.get('package_manifest_sha256') != seal.sha256 or
            execution.get('stage_manifest_sha256') != stage_hash or
            execution.get('parent_checkpoint_sha256') != parent_hash or
            execution.get('adapter_sha256') != adapter_hash or execution.get('core_code_hashes') != core_hashes or
            execution.get('finite_input') is not True or execution.get('producer_enabled') is not False):
        raise ValueError('Modal execution input/source binding mismatch')
    execution_hash = hashlib.sha256(json.dumps(execution, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (contract.get('format') != 'udgam-owned-main-scratch-v1' or contract.get('provider_guard') != execution or
            contract.get('pilot_checkpoint_sha256') != parent_hash or contract.get('initial_weights_sha256') != initial or
            contract.get('adapter_code_hashes') != core_hashes):
        raise ValueError('Run contract does not bind exact execution/core/initialization')
    for record in (contract, result):
        if (record.get('execution_profile') != 'modal-v1' or record.get('modal_adapter_sha256') != adapter_hash or
                record.get('modal_execution_contract_sha256') != execution_hash or
                record.get('finite_source_manifest_sha256') != stage_hash or record.get('producer_enabled') is not False):
            raise ValueError('Run evidence Modal profile/source/contract binding mismatch')
    if (result.get('status') != 'main_session_completed' or result.get('candidate_valid') is not True or
            result.get('checkpoint_reload_verified') is not True or
            result.get('checkpoint_sha256') != side['sha256'] or
            type(result.get('checkpoint_bytes')) is not int or
            result['checkpoint_bytes'] != regular(root, run + '/checkpoint.pt').stat().st_size or
            result.get('initial_weights_sha256') != initial or result.get('provider_stop_enforced_externally') is not True):
        raise ValueError('Standard export lacks matching successful checkpoint reload evidence')
    failure = root / run / 'failure.json'
    if failure.exists(): raise ValueError('Standard success export cannot include a failure record; use explicit recovery')


def describe_omitted(directory, name):
    if name not in CHECKPOINT_FILES | TRANSIENT_FILES: raise ValueError("Unreviewed omission")
    path = directory / name
    if path.is_symlink(): raise ValueError("Symlink omitted output forbidden")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1: raise ValueError("Unsafe omitted output")
    checksum = digest(path); after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Omitted output changed during hashing")
    return {"name": name, "size_bytes": after.st_size, "sha256": checksum}


def build_delta(root, output, run, execution, trainer_pid, *, seal=Seal(), recovery_checkpoint=None):
    root = root_path(root); stopped(trainer_pid)
    if not re.fullmatch(r"records/modal-delta-[A-Za-z0-9_-]+\.json", output): raise ValueError("Explicit new delta manifest required")
    target = root / output
    if target.exists() or target.is_symlink(): raise FileExistsError("Delta manifest exists")
    if target.parent.is_symlink(): raise ValueError("Unsafe output parent")
    base = input_inventory(root, seal=seal)
    allowed = delta_paths(run, execution); directory = root / run
    if directory.is_symlink() or not directory.is_dir(): raise ValueError("Run directory missing or unsafe")
    paths = {p.relative_to(root).as_posix() for p in directory.iterdir()}; omitted = []
    if recovery_checkpoint is not None:
        if recovery_checkpoint not in {"checkpoint.pt", "checkpoint.recovery.pt"}: raise ValueError("Explicit recovery checkpoint required")
        pair = {recovery_checkpoint, recovery_checkpoint.removesuffix('.pt') + '.sha256.json'}
        required = {run + "/" + x for x in pair | {"failure.json", "run-contract.json"}}
        if not required <= paths or not paths <= allowed | {run+'/'+x for x in TRANSIENT_FILES}:
            raise ValueError("Missing failure recovery inputs or unknown run output")
        for name in sorted((CHECKPOINT_FILES - pair) | TRANSIENT_FILES):
            if run + '/' + name in paths:
                omitted.append(describe_omitted(directory, name)); paths.remove(run + '/' + name)
    elif not {run + "/" + x for x in REQUIRED_RUN} <= paths or not paths <= allowed:
        raise ValueError("Missing required or unknown run output")
    paths |= {ADAPTER, TRANSFER, execution}
    if digest(regular(root, TRANSFER)) != digest(Path(__file__)):
        raise ValueError("Transfer source copy differs from executing helper")
    for name in paths:
        p = regular(root, name)
        if not name.endswith(".pt"): scan_record(p)
    initial = verify_pairs(root, paths, base, root)
    if recovery_checkpoint is None: verify_standard_records(root, run, execution, base, initial, seal)
    items = [describe(root, name) for name in sorted(paths)]
    stopped(trainer_pid)
    value = {"format": DELTA_FORMAT, "manifest_path": output,
        "base_manifest_path": seal.path, "base_manifest_sha256": seal.sha256,
        "base_file_count": seal.files, "base_total_bytes": seal.total_bytes,
        "run": run, "execution_record": execution, "trainer_pid_observed_stopped": trainer_pid,
        "export_kind": "failure_recovery" if recovery_checkpoint else "standard",
        "selected_checkpoint": run + '/' + (recovery_checkpoint or 'checkpoint.pt'),
        "omitted_unselected_or_incomplete_files": omitted,
        "producer_started": False, "input_stage_unchanged": True,
        "files": items, "total_bytes": sum(x["size_bytes"] for x in items),
        "transfer_tool_sha256": digest(Path(__file__)),
        "contains_provider_credentials": False,
        "semantic_resume_after_restore_tested": False}
    with target.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    return value


def verify_delta(root, manifest_name, base_root, *, strict=False, seal=Seal(), expected_manifest_sha256=None):
    root = root_path(root)
    if not re.fullmatch(r"records/modal-delta-[A-Za-z0-9_-]+\.json", manifest_name): raise ValueError("Unsafe delta manifest path")
    manifest_path = regular(root, manifest_name)
    if expected_manifest_sha256 is not None and digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("Delta manifest differs from trusted transfer receipt")
    value = read_json(manifest_path); paths = check_items(value.get("files"))
    if (value.get("format") != DELTA_FORMAT or value.get("manifest_path") != manifest_name or
            value.get("base_manifest_path") != seal.path or value.get("base_manifest_sha256") != seal.sha256 or
            value.get("base_file_count") != seal.files or value.get("base_total_bytes") != seal.total_bytes or
            value.get("producer_started") is not False or value.get("input_stage_unchanged") is not True or
            value.get("contains_provider_credentials") is not False or
            value.get("semantic_resume_after_restore_tested") is not False or
            value.get("total_bytes") != sum(x["size_bytes"] for x in value["files"])):
        raise ValueError("Delta recovery contract mismatch")
    allowed = delta_paths(value["run"], value["execution_record"])
    selected = value.get('selected_checkpoint')
    permitted_checkpoints = {value['run'] + '/' + x for x in ('checkpoint.pt', 'checkpoint.recovery.pt')}
    if selected not in permitted_checkpoints: raise ValueError("Invalid selected recovery checkpoint")
    omitted = value.get('omitted_unselected_or_incomplete_files')
    if not isinstance(omitted, list): raise ValueError("Omission record required")
    if value.get('export_kind') == 'standard':
        if omitted or selected != value['run'] + '/checkpoint.pt': raise ValueError("Invalid standard export")
        required = {value["run"] + "/" + x for x in REQUIRED_RUN}
    elif value.get('export_kind') == 'failure_recovery':
        pair = {selected, selected.removesuffix('.pt') + '.sha256.json'}
        required = pair | {value['run']+'/failure.json', value['run']+'/run-contract.json'}
        if any(p.endswith('.pt') and p != selected for p in paths): raise ValueError("Unselected recovery checkpoint included")
        omitted_names = set()
        for item in omitted:
            if (set(item) != {'name','size_bytes','sha256'} or item['name'] not in CHECKPOINT_FILES | TRANSIENT_FILES or
                    item['name'] in omitted_names or value['run']+'/'+item['name'] in pair or
                    type(item['size_bytes']) is not int or item['size_bytes'] < 0 or not re.fullmatch(r'[a-f0-9]{64}',item['sha256'])):
                raise ValueError("Unsafe omission record")
            omitted_names.add(item['name'])
    else: raise ValueError("Unknown export kind")
    required |= {ADAPTER, TRANSFER, value["execution_record"]}
    if not required <= paths or not paths <= allowed: raise ValueError("Delta allowlist mismatch")
    transfer = next(x for x in value['files'] if x['path'] == TRANSFER)
    if transfer['sha256'] != value.get('transfer_tool_sha256'): raise ValueError("Transfer source identity mismatch")
    base = input_inventory(base_root, seal=seal)
    for item in value["files"]:
        if describe(root, item["path"]) != item: raise ValueError("Delta artifact mismatch")
        if not item["path"].endswith(".pt"): scan_record(regular(root, item["path"]))
    initial = verify_pairs(root, paths, base, base_root)
    if value['export_kind'] == 'standard':
        verify_standard_records(root, value['run'], value['execution_record'], base, initial, seal)
    if strict: exact_tree(root, paths | {manifest_name})
    return value


def export_delta(root, manifest_name, destination, *, seal=Seal()):
    value = verify_delta(root, manifest_name, root, seal=seal)
    items = value["files"] + [describe(root_path(root), manifest_name)]
    # Volume statfs is not capacity evidence. Hash/size bounds remain exact;
    # provider storage quotas/commit are the caller's explicit responsibility.
    out = copy_inventory(root, destination, items, reserve_bytes=0, check_free=False)
    verify_delta(out, manifest_name, root, strict=True, seal=seal)
    return {"exported": True, "destination": str(out), "files": len(items),
        "total_bytes": sum(x["size_bytes"] for x in items),
        "delta_manifest_sha256": digest(out / manifest_name), "volume_commit_required": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inventory", "materialize", "delta", "export", "verify-delta"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--destination", type=Path)
    p.add_argument("--manifest")
    p.add_argument("--base-root", type=Path)
    p.add_argument("--run", default="runs/main-modal-001")
    p.add_argument("--execution", default="records/modal-execution-001.json")
    p.add_argument("--trainer-pid", type=int)
    p.add_argument("--reserve-bytes", type=int, default=8 * 1024**3)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--expected-manifest-sha256")
    p.add_argument("--recovery-checkpoint", choices=('checkpoint.pt','checkpoint.recovery.pt'),
                   help="Explicit interrupted-run export; requires truthful failure.json, no invented result.")
    a = p.parse_args(argv)
    if a.action in {"materialize", "export"} and a.destination is None: p.error("--destination required")
    if a.action in {"delta", "export", "verify-delta"} and a.manifest is None: p.error("--manifest required")
    if a.action == "inventory": result = input_inventory(a.root, strict=a.strict)
    elif a.action == "materialize": result = materialize(a.root, a.destination, reserve_bytes=a.reserve_bytes)
    elif a.action == "delta": result = build_delta(a.root, a.manifest, a.run, a.execution, a.trainer_pid,
                                                   recovery_checkpoint=a.recovery_checkpoint)
    elif a.action == "export": result = export_delta(a.root, a.manifest, a.destination)
    else:
        if a.base_root is None: p.error("--base-root required")
        if not a.expected_manifest_sha256: p.error("--expected-manifest-sha256 from trusted export receipt required")
        value = verify_delta(a.root, a.manifest, a.base_root, strict=a.strict,
                             expected_manifest_sha256=a.expected_manifest_sha256)
        result = {"verified": True, "files": len(value["files"]), "total_bytes": value["total_bytes"],
            "delta_manifest_sha256": digest(a.root / a.manifest), "base_manifest_sha256": BASE_SHA,
            "semantic_resume_after_restore_tested": False}
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
