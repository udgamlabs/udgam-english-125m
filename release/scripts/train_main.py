#!/usr/bin/env python3
"""Finite main training from our owned pilot, across separately guarded sessions.

The main optimizer starts fresh ONCE because the pilot already annealed. The
main horizon/schedule remains fixed across sessions; exact main resumes restore
optimizer/RNG/cursor. No training-source wrapping or pretrained loading exists.
GPU sessions require the separate reserved main-v1 <=$7/120-minute lifecycle.
The original <=$4.50/40-minute pilot guard is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import train_dense as pilot
from prepare_dense_tokens import ROOT, REVISION, VENDOR, sha256, verify_prepared, write_json

FORMAT = "udgam-owned-main-scratch-v1"


def validate_main_guard(path, *, now=None, export_reserve_seconds=300):
    """Verify public local-controller evidence; it does not itself stop a Pod."""
    now = time.time() if now is None else now
    value = json.loads(Path(path).read_text())
    intent = value["intent"]; reservation = value["budget_reservation"]; budget = value["budget_snapshot"]
    required_intent = {"version", "pod_name", "armed_at", "deadline_at", "max_seconds",
        "max_gpu_rate", "max_extra_rate", "max_dollars", "poll_seconds", "stop_reserve_seconds",
        "budget_reserve", "expected_gpu_count", "profile", "reservation_id", "budget_path"}
    if set(intent) != required_intent or value.get("profile") != "main-v1" or intent["profile"] != "main-v1":
        raise ValueError("Explicit main-v1 guard profile required")
    checksum = hashlib.sha256(canonical(intent).encode()).hexdigest()
    if value.get("guard_intent_sha256") != checksum or reservation.get("intent_sha256") != checksum:
        raise ValueError("Main guard intent/reservation hash mismatch")
    rid = value.get("reservation_id", "")
    if not isinstance(rid, str) or not re.fullmatch(r"[a-f0-9]{32}", rid) or intent["reservation_id"] != rid:
        raise ValueError("Main reservation identity mismatch")
    pod = value.get("pod_id")
    if not isinstance(pod, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", pod):
        raise ValueError("Missing bound provider pod")
    if (type(intent["version"]) is not int or intent["version"] != 1 or
        type(intent["expected_gpu_count"]) is not int or intent["expected_gpu_count"] != 1 or
        value.get("pod_name") != intent["pod_name"] or reservation.get("pod_name") != intent["pod_name"] or
        not isinstance(intent["pod_name"], str) or not re.fullmatch(r"udgam-main-[a-f0-9]{32}", intent["pod_name"])):
        raise ValueError("Main lifecycle pod identity mismatch")
    if not isinstance(intent["budget_path"], str) or not Path(intent["budget_path"]).is_absolute() or Path(intent["budget_path"]).name != "budget.json":
        raise ValueError("Main canonical budget identity missing")
    if (reservation.get("version") != 1 or reservation.get("profile") != "main-v1" or
        reservation.get("stage") != "pretraining" or reservation.get("status") != "active" or
        reservation.get("pod_id") != pod or reservation.get("amount_usd") != 7 or
        not isinstance(reservation.get("state_path"), str) or not Path(reservation["state_path"]).is_absolute()):
        raise ValueError("Main budget reservation not active for this pod")
    observed = pilot.utc_timestamp(value["observed_at_utc"])
    if not 0 <= now - observed <= 120 or not 0 <= now - pilot.utc_timestamp(budget["observed_at_utc"]) <= 120:
        raise ValueError("Main billing/budget snapshot stale or in future")
    rate = pilot.finite_number(value["hourly_total_usd"])
    ceiling = pilot.finite_number(value["cash_ceiling_usd"])
    gpu_rate = pilot.finite_number(intent["max_gpu_rate"])
    extra_rate = pilot.finite_number(intent["max_extra_rate"], positive=False)
    seconds = pilot.finite_number(intent["max_seconds"])
    armed = pilot.finite_number(intent["armed_at"])
    deadline = pilot.utc_timestamp(value["provider_stop_deadline_utc"])
    if (ceiling != 7 or intent["max_dollars"] != 7 or not 60 < seconds <= 7200 or
        armed > now or abs(pilot.finite_number(intent["deadline_at"]) - armed - seconds) > .001 or
        abs(deadline - intent["deadline_at"]) > .001 or rate < gpu_rate + extra_rate):
        raise ValueError("Main lifecycle exceeds its declared time/rate/cash bounds")
    if intent["poll_seconds"] != 5 or intent["stop_reserve_seconds"] != 60 or intent["budget_reserve"] != .25:
        raise ValueError("Main controller safety reserves changed")
    guard = value["provider_stop_guard"]
    heartbeat = pilot.utc_timestamp(guard["heartbeat_at_utc"])
    if (guard.get("verified") is not True or guard.get("pod_id") != pod or
        pilot.utc_timestamp(guard["deadline_utc"]) != deadline or not 0 <= now - heartbeat <= 60):
        raise ValueError("Main provider guard identity/heartbeat mismatch")
    if (budget.get("experiment_ceiling_usd") != 99 or budget.get("stage") != "pretraining" or
        budget.get("stage_ceiling_usd") != 62 or budget.get("auto_top_up_authorized") is not False):
        raise ValueError("Main project/stage budget authorization mismatch")
    for key in ("consumed_credit_usd", "unsettled_charges_usd", "stage_consumed_credit_usd", "new_cash_deposits_usd",
                "observed_prepaid_credit_usd", "reserved_total_usd", "reserved_stage_usd"):
        pilot.finite_number(budget[key], positive=False)
    if (budget["new_cash_deposits_usd"] > 99 or budget["reserved_total_usd"] != 7 or budget["reserved_stage_usd"] != 7 or
        budget["consumed_credit_usd"] + budget["unsettled_charges_usd"] + budget["reserved_total_usd"] > 99 + 1e-9 or
        budget["stage_consumed_credit_usd"] + budget["unsettled_charges_usd"] + budget["reserved_stage_usd"] > 62 + 1e-9):
        raise ValueError("Main reservation exceeds project/stage capacity")
    cost = pilot.finite_number(value["conservative_elapsed_cost_usd"], positive=False)
    # The guard stops before the raw deadline or cash ceiling, whichever comes
    # first. Preserve export time BEFORE that earlier stop, including heartbeat
    # age and conservative monotonic elapsed accounting from the controller.
    elapsed = max(now - armed, cost * 3600 / rate + now - heartbeat)
    cash_remaining_seconds = (ceiling - intent["budget_reserve"]) * 3600 / rate - elapsed - intent["stop_reserve_seconds"]
    remaining = min(deadline - now - intent["stop_reserve_seconds"], seconds - elapsed - intent["stop_reserve_seconds"], cash_remaining_seconds)
    if remaining <= export_reserve_seconds + 30:
        raise ValueError("Insufficient main lifecycle time/cash for safe export")
    return value, now + remaining - export_reserve_seconds


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def code_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in
            ("train_main.py", "train_dense.py", "prepare_dense_tokens.py")}


def safe_file(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unsafe staged artifact path")
    path = directory / relative
    within = path
    while within != directory:
        if within.is_symlink(): raise ValueError("Unsafe staged artifact path")
        within = within.parent
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Unsafe staged artifact path")
    return path


def read_manifest(directory, *, attempts=5, sleep=time.sleep):
    """Builder publishes JSON then its digest; retry only that short pair race."""
    directory = Path(directory)
    previous = None
    for attempt in range(attempts):
        try:
            body = (directory / "manifest.json").read_bytes()
            expected = (directory / "manifest.sha256").read_text().strip()
            if hashlib.sha256(body).hexdigest() == expected:
                return json.loads(body), expected
            snapshot = (hashlib.sha256(body).hexdigest(), expected)
            if snapshot == previous:
                raise ValueError("Stable stage manifest checksum mismatch")
            previous = snapshot
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if attempt + 1 < attempts: sleep(.1)
    raise ValueError("Committed stage manifest missing or hash pair remains inconsistent")


class FiniteShards:
    """Append-only committed shard inventory with transactional row cursors."""
    def __init__(self, directory, pilot_data, pilot_manifest, pilot_digest, *, tiny=False,
                 contract=None, accepted_shards=None, cursor=None):
        self.directory, self.pilot_data = Path(directory), Path(pilot_data)
        self.pilot_manifest, self.pilot_digest = pilot_manifest, pilot_digest
        self.tiny = tiny; self.contract = contract
        self.shards = list(accepted_shards or [])
        self.cursor = dict(cursor or {"shard_id": 0, "row_in_shard": 0, "total_rows": 0})
        self.arrays = {}; self.verified = set(); self.manifest_hash = None
        self.metadata_verified = False
        # These parent artifacts were verified before construction. Cache their
        # identities, rather than rereading megabytes of provenance every step.
        self.parent_checks = {"pilot_manifest_sha256": pilot_digest,
            "source_report_sha256": sha256(self.pilot_data / "source-report.json"),
            "tokenizer_sha256": pilot_manifest["hashes"]["tokenizer.json"],
            "frozen_validation_sha256": pilot_manifest["hashes"]["val.npy"]}
        source_report = json.loads((self.pilot_data / "source-report.json").read_text())
        if not tiny:
            self.parent_checks.update(source_lock_sha256=source_report["source_lock_sha256"],
                benchmark_manifest_sha256=source_report["benchmark_manifest_sha256"],
                builder_sha256=sha256(Path(__file__).with_name("prepare_main.py")),
                token_bytes_sha256=pilot_manifest["hashes"]["token_bytes.npy"])
        self.helper_hashes = {name: sha256(Path(__file__).with_name(name)) for name in ("prepare_english.py", "prepare_production.py")} if not tiny else None
        self.refresh()
        self.validate_cursor()
        # Resume rechecks all checkpoint-accepted artifacts, including consumed
        # shards. A deleted or changed past shard is never silently skipped.
        for sid in range(len(self.shards)):
            self.array(sid)
        self.release_consumed_arrays()

    def refresh(self):
        value, current_hash = read_manifest(self.directory)
        if value.get("schema_version") != 1 or value.get("status") != "initialized":
            raise ValueError("Stage is blocked, failed or not fully initialized")
        if value.get("validation_policy") != "frozen_pilot_validation_excluded_from_new_train":
            raise ValueError("Main validation exclusion policy missing")
        lineage = value["lineage"]
        checks = self.parent_checks
        if not self.tiny and lineage.get("helper_sha256") != self.helper_hashes:
            raise ValueError("Main preparation helper identity mismatch")
        if any(lineage.get(k) != v for k, v in checks.items()):
            raise ValueError("Main stage and owned pilot provenance mismatch")
        if value["config"]["sequence_len"] != self.pilot_manifest["sequence_len"]:
            raise ValueError("Main token row width differs from pilot")
        # The path is descriptive provenance. --pilot-data explicitly maps the
        # immutable validation bytes after relocation to another /workspace.
        portable_lineage = {k: v for k, v in lineage.items() if k not in ("frozen_validation_path", "token_bytes_path")}
        # Coordinator may explicitly raise source download/storage bounds after
        # moving the stage. These do not change training data policy or horizon.
        data_config = {k: v for k, v in value["config"].items() if k not in ("max_download_bytes", "max_output_bytes", "disk_reserve_bytes")}
        contract = {"config": data_config, "lineage": portable_lineage,
                    "validation_policy": value["validation_policy"]}
        if self.contract is None:
            self.contract = contract
        elif contract != self.contract:
            raise ValueError("Main stage contract changed after launch")
        if not self.metadata_verified:
            if sha256(self.directory / "tokenizer.json") != checks["tokenizer_sha256"]:
                raise ValueError("Main tokenizer changed")
            metadata = lineage.get("metadata_artifact_hashes", {})
            if not self.tiny and not {"source-lock.json", "source-files.jsonl.gz", "benchmark-selection.json"} <= set(metadata):
                raise ValueError("Missing main source metadata artifacts")
            for name, checksum in metadata.items():
                if sha256(safe_file(self.directory, name)) != checksum:
                    raise ValueError("Main source metadata integrity failure")
            self.metadata_verified = True
        new = value["shards"]
        if len(new) < len(self.shards) or new[:len(self.shards)] != self.shards:
            raise ValueError("Committed shard inventory was rewritten or removed")
        width = self.pilot_manifest["sequence_len"] + 1
        for index, shard in enumerate(new):
            required = {f"shards/train_{index:05d}.npy", f"shards/train_{index:05d}.provenance.jsonl.gz"}
            if shard["id"] != index or type(shard["rows"]) is not int or shard["rows"] < 1:
                raise ValueError("Shard IDs must be consecutive with positive finite rows")
            if shard["row_width"] != width or shard["dtype"] != "<u2" or shard["wrapping"] is not False:
                raise ValueError("Invalid finite shard format")
            if shard["encoded_positions"] != shard["rows"] * width or shard["prediction_targets"] != shard["rows"] * (width - 1):
                raise ValueError("Shard position count mismatch")
            if set(shard["hashes"]) != required:
                raise ValueError("Shard requires exact array/provenance hash inventory")
        self.shards = new; self.manifest_hash = current_hash
        return value

    def validate_cursor(self):
        sid, row, total = (self.cursor[k] for k in ("shard_id", "row_in_shard", "total_rows"))
        if any(type(v) is not int or v < 0 for v in (sid, row, total)) or sid > len(self.shards):
            raise ValueError("Invalid finite cursor")
        if sid == len(self.shards):
            if row: raise ValueError("Cursor lies beyond committed supply")
        elif row >= self.shards[sid]["rows"]:
            raise ValueError("Cursor must advance at shard boundary")
        if total != sum(s["rows"] for s in self.shards[:sid]) + row:
            raise ValueError("Cursor total does not match shard position")

    def available_rows(self):
        return sum(s["rows"] for s in self.shards[self.cursor["shard_id"]:]) - self.cursor["row_in_shard"]

    def array(self, sid):
        import numpy as np
        if sid not in self.verified:
            shard = self.shards[sid]
            for name, expected in shard["hashes"].items():
                path = safe_file(self.directory, name)
                if not path.is_file() or sha256(path) != expected:
                    raise ValueError("Committed shard missing or hash mismatch; never skip it")
            arr = np.load(self.directory / f"shards/train_{sid:05d}.npy", mmap_mode="r", allow_pickle=False)
            if arr.dtype != np.dtype("<u2") or list(arr.shape) != [shard["rows"], shard["row_width"]] or int(arr.max()) >= self.pilot_manifest["vocab_size"]:
                raise ValueError("Committed token array shape, dtype or vocabulary mismatch")
            self.arrays[sid] = arr; self.verified.add(sid)
        return self.arrays[sid]

    def peek(self, rows):
        """No cursor mutation until the entire optimizer step succeeds."""
        import numpy as np
        if rows < 1 or rows > self.available_rows():
            raise ValueError("Insufficient finite committed supply")
        sid, offset = self.cursor["shard_id"], self.cursor["row_in_shard"]
        remaining, pieces = rows, []
        while remaining:
            take = min(remaining, self.shards[sid]["rows"] - offset)
            pieces.append(np.array(self.array(sid)[offset:offset + take], copy=True))
            remaining -= take; offset += take
            if offset == self.shards[sid]["rows"]: sid += 1; offset = 0
        cursor = {"shard_id": sid, "row_in_shard": offset, "total_rows": self.cursor["total_rows"] + rows}
        return np.concatenate(pieces, axis=0), cursor

    def commit(self, cursor):
        self.cursor = dict(cursor); self.validate_cursor()
        self.release_consumed_arrays()

    def release_consumed_arrays(self):
        # Retain only mappings for the current/future shard, not all training data.
        for sid in list(self.arrays):
            if sid < self.cursor["shard_id"]: del self.arrays[sid]; self.verified.remove(sid)

    def wait_for_rows(self, rows, *, max_wait, deadline, poll_seconds=2, clock=time.monotonic, sleep=time.sleep):
        start = clock()
        while True:
            self.refresh()
            if self.available_rows() >= rows: return True, clock() - start
            remaining = min(max_wait - (clock() - start), deadline - time.time())
            if remaining <= 0: return False, clock() - start
            sleep(min(poll_seconds, remaining))


def horizon_config(args, sequence_len):
    batch_tokens = args.batch_rows * args.accumulation_steps * sequence_len
    if args.horizon_targets < batch_tokens or min(args.batch_rows, args.accumulation_steps) < 1:
        raise ValueError("Invalid main horizon/batch")
    if not args.tiny_cpu_test and args.horizon_targets < 2_000_000_000:
        raise ValueError("Production main horizon must be declared before launch at>=2B targets")
    steps = args.horizon_targets // batch_tokens
    return {"declared_horizon_targets": args.horizon_targets, "effective_horizon_targets": steps * batch_tokens,
            "discarded_horizon_rounding_targets": args.horizon_targets % batch_tokens,
            "max_steps": steps, "batch_rows": args.batch_rows, "accumulation_steps": args.accumulation_steps,
            "warmup_steps": min(40, max(1, steps // 10)), "warmdown_ratio": .65, "final_lr_frac": .05,
            "model_compile": args.compile_model, "optimizer_compile": not args.tiny_cpu_test,
            "seed": args.seed, "optimizer_transition": "fresh_MuonAdamW_once_after_annealed_owned_pilot"}


def inspect_owned(path, *, main, torch):
    path = Path(path)
    side = json.loads(path.with_suffix(".sha256.json").read_text())
    required = FORMAT if main else pilot.CHECKPOINT_FORMAT
    if side.get("format") != required or side.get("sha256") != sha256(path):
        raise ValueError("Owned checkpoint hash/format mismatch")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("format") != required or payload.get("source_revision") != REVISION or payload.get("source_hashes") != pilot.source_hashes():
        raise ValueError("Owned checkpoint native source mismatch")
    if main and payload.get("adapter_code_hashes") != code_hashes():
        raise ValueError("Main resume adapter changed; explicit reviewed migration required")
    if side.get("initial_weights_sha256") != payload["state"]["initial_weights_sha256"]:
        raise ValueError("Original scratch initialization provenance mismatch")
    return payload, side


def save_main(path, model, optimizer, state, stream, torch):
    state["cursor"] = stream.cursor
    state["stage_contract"] = stream.contract
    state["accepted_shards"] = stream.shards
    state["last_stage_manifest_sha256"] = stream.manifest_hash
    payload = {"format": FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "model_config": asdict(model.config), "state": state, "rng": pilot.rng_state(torch),
               "source_revision": REVISION, "source_hashes": pilot.source_hashes(), "adapter_code_hashes": code_hashes()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)
    write_json(path.with_suffix(".sha256.json"), {"format": FORMAT, "sha256": sha256(path),
        "initial_weights_sha256": state["initial_weights_sha256"], "pilot_checkpoint_sha256": state["pilot_checkpoint_sha256"],
        "pilot_data_manifest_sha256": state["pilot_data_manifest_sha256"]})


def restore_main(payload, side, model, optimizer, stream, *, config, pilot_digest, torch):
    state = payload["state"]
    if payload["model_config"] != asdict(model.config) or state["horizon_config"] != config:
        raise ValueError("Main horizon/model configuration cannot change on resume")
    if state["pilot_data_manifest_sha256"] != pilot_digest or side.get("pilot_data_manifest_sha256") != pilot_digest or side.get("pilot_checkpoint_sha256") != state["pilot_checkpoint_sha256"]:
        raise ValueError("Main parent lineage mismatch")
    if state["stage_contract"] != stream.contract or state["accepted_shards"] != stream.shards[:len(state["accepted_shards"])]:
        raise ValueError("Main resume data inventory mismatch")
    expected_rows = state["step"] * config["batch_rows"] * config["accumulation_steps"]
    if state["cursor"]["total_rows"] != expected_rows or state["main_target_positions"] != expected_rows * model.config.sequence_len:
        raise ValueError("Main checkpoint optimizer-step/token cursor mismatch")
    model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
    pilot.restore_rng(payload["rng"], torch)
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument("--pilot-checkpoint", type=Path)
    origin.add_argument("--resume-main", type=Path)
    p.add_argument("--pilot-data", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--guard-evidence", type=Path)
    p.add_argument("--horizon-targets", type=int, default=2_000_000_000)
    p.add_argument("--batch-rows", type=int, default=16)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--session-compute-seconds", type=float, default=1500)
    p.add_argument("--export-reserve-seconds", type=float, default=1800,
        help="Time before the provider cutoff reserved for checkpoint and full-stage backup (300..1800).")
    p.add_argument("--starvation-seconds", type=float, default=60)
    p.add_argument("--project-budget-usd", type=float, default=99)
    p.add_argument("--project-spent-usd", type=float, default=0)
    p.add_argument("--eval-rows", type=int, default=128)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--tiny-cpu-test", action="store_true")
    p.add_argument("--stop-after-session-steps", type=int, default=0)
    return p


def run(args):
    process_started = time.time()
    if args.output.exists(): raise ValueError("Use a fresh immutable session output")
    export_reserve = getattr(args, "export_reserve_seconds", 1800)
    pilot.finite_number(export_reserve)
    if not 300 <= export_reserve <= 1800:
        raise ValueError("Main backup reserve must be between300 and1800 seconds")
    for value in (args.session_compute_seconds, args.starvation_seconds, args.project_budget_usd): pilot.finite_number(value)
    pilot.finite_number(args.project_spent_usd, positive=False)
    if args.session_compute_seconds > 1500 or args.starvation_seconds > 120 or args.project_budget_usd > 99 or args.project_spent_usd >= args.project_budget_usd:
        raise ValueError("Main session/project bound exceeded")
    if args.tiny_cpu_test and (args.session_compute_seconds > 60 or args.horizon_targets > 4096 or args.batch_rows > 4 or args.accumulation_steps > 2):
        raise ValueError("CPU flag permits only a tiny main continuation test")
    if not args.tiny_cpu_test and args.stop_after_session_steps:
        raise ValueError("Artificial interruption only allowed in CPU regression tests")
    data, data_digest = verify_prepared(args.pilot_data, production=not args.tiny_cpu_test)
    if args.eval_rows < 1 or args.eval_rows > data["splits"]["val"]["rows"]:
        raise ValueError("Insufficient finite frozen validation rows")
    config = horizon_config(args, data["sequence_len"])
    guard = None; stop_at = float("inf")
    if not args.tiny_cpu_test:
        if args.guard_evidence is None: raise ValueError("Fresh independent provider guard evidence required")
        guard, stop_at = validate_main_guard(args.guard_evidence, export_reserve_seconds=export_reserve)
        if guard["cash_ceiling_usd"] > args.project_budget_usd - args.project_spent_usd:
            raise ValueError("Declared remaining project budget cannot cover this guarded session")
    torch, GPT, GPTConfig, KVCache, dtype, backend = pilot.runtime(tiny=args.tiny_cpu_test,
                   attention="sdpa" if args.tiny_cpu_test else "fa3", optimizer_compile=not args.tiny_cpu_test)
    import numpy as np
    if not args.tiny_cpu_test and str(torch.__version__).split("+")[0] != "2.9.1":
        raise ValueError("Production main training requires the reviewed torch2.9.1 runtime")
    runtime_identity = {"torch": str(torch.__version__), "numpy": np.__version__, "dtype": str(dtype),
                        "attention_backend": backend, "fa3_revision": None if args.tiny_cpu_test else pilot.FA3_REVISION}
    parent_path = args.resume_main or args.pilot_checkpoint
    payload, side = inspect_owned(parent_path, main=bool(args.resume_main), torch=torch)
    model_config = payload["model_config"]
    pilot.validate_config({**model_config, "source_revision": REVISION, "initialization": "random_weights_only"}, tiny=args.tiny_cpu_test)
    if model_config["sequence_len"] != data["sequence_len"] or model_config["vocab_size"] != data["vocab_size"]:
        raise ValueError("Owned model and prepared tokenizer/sequence dimensions differ")
    previous = payload["state"] if args.resume_main else None
    if previous and previous.get("runtime_identity") != runtime_identity:
        raise ValueError("Main runtime identity changed; explicit reviewed migration required")
    stream = FiniteShards(args.stage, args.pilot_data, data, data_digest, tiny=args.tiny_cpu_test,
            contract=previous["stage_contract"] if previous else None,
            accepted_shards=previous["accepted_shards"] if previous else None,
            cursor=previous["cursor"] if previous else None)
    if args.resume_main and previous["horizon_config"] != config:
        raise ValueError("Main horizon/model configuration cannot change on resume")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cpu" if args.tiny_cpu_test else "cuda"
    with torch.device("meta"): model = GPT(GPTConfig(**model_config))
    model.to_empty(device=device); model.init_weights()
    optimizer, optimizer_info = pilot.setup_optimizer(model, config, GPT, GPTConfig, torch)
    if args.resume_main:
        state = restore_main(payload, side, model, optimizer, stream, config=config, pilot_digest=data_digest, torch=torch)
    else:
        if payload["state"]["data_manifest_sha256"] != data_digest or side.get("data_manifest_sha256") != data_digest:
            raise ValueError("Initial pilot checkpoint data lineage mismatch")
        if payload["state"]["step"] < 1:
            raise ValueError("Owned pilot must have completed optimizer updates")
        model.load_state_dict(payload["model"], strict=True)
        state = {"step": 0, "main_target_positions": 0, "cumulative_training_seconds": 0.,
                 "initial_weights_sha256": payload["state"]["initial_weights_sha256"],
                 "pilot_checkpoint_sha256": side["sha256"], "pilot_data_manifest_sha256": data_digest,
                 "pilot_target_positions": payload["state"]["token_offset"],
                 "horizon_config": config, "optimizer_transition": config["optimizer_transition"],
                 "runtime_identity": runtime_identity,
                 "initial_main_weights_sha256": pilot.model_hash(model), "sessions": [], "loss_history": []}
    del payload
    if state["step"] >= config["max_steps"]:
        raise ValueError("Main horizon already completed; extending it requires a separately declared experiment")
    args.output.mkdir(parents=True)
    write_json(args.output / "run-contract.json", {"format": FORMAT, "horizon": config, "model": model_config,
        "stage_contract": stream.contract, "pilot_checkpoint_sha256": state["pilot_checkpoint_sha256"],
        "initial_weights_sha256": state["initial_weights_sha256"], "optimizer": optimizer_info,
        "adapter_code_hashes": code_hashes(), "native_source_hashes": pilot.source_hashes(),
        "session_compute_seconds": args.session_compute_seconds, "starvation_seconds": args.starvation_seconds,
        "export_reserve_seconds": export_reserve,
        "provider_guard": guard, "project_budget_usd": args.project_budget_usd, "declared_project_spent_usd": args.project_spent_usd,
        "budget_enforced_by_this_script": False, "attention_backend": backend, "dtype": str(dtype), "torch": str(torch.__version__),
        "runtime_identity": runtime_identity,
        "main_optimizer_reused": bool(args.resume_main), "validation_path_mapping": str(args.pilot_data / "val.npy")})
    val = np.load(args.pilot_data / "val.npy", mmap_mode="r", allow_pickle=False)
    token_bytes = torch.from_numpy(np.load(args.pilot_data / "token_bytes.npy", allow_pickle=False)).to(device)
    execution = torch.compile(model, dynamic=False) if args.compile_model else model
    session = {"session_index": len(state["sessions"]), "start_step": state["step"], "compute_seconds": 0., "starvation_wait_seconds": 0.}
    state["sessions"].append(session)
    checkpoint = args.output / "checkpoint.pt"
    stop_reason = "main_horizon_complete"
    if device == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    rows_per_step = config["batch_rows"] * config["accumulation_steps"]
    try:
        with (args.output / "loss-history.jsonl").open("w") as log:
            while state["step"] < config["max_steps"]:
                if session["compute_seconds"] >= args.session_compute_seconds or time.time() >= stop_at:
                    stop_reason = "session_compute_limit" if session["compute_seconds"] >= args.session_compute_seconds else "provider_export_reserve"; break
                available, waited = stream.wait_for_rows(rows_per_step, max_wait=args.starvation_seconds, deadline=stop_at)
                session["starvation_wait_seconds"] += waited
                if not available: stop_reason = "finite_data_starvation"; break
                if time.time() >= stop_at: stop_reason = "provider_export_reserve"; break
                batch, next_cursor = stream.peek(rows_per_step)
                start = time.perf_counter()
                _, loss, lr = pilot.train_step(model, execution, optimizer, batch, 0, state["step"], config, optimizer_info, device, torch)
                if device == "cuda": torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                stream.commit(next_cursor)
                state["step"] += 1; state["main_target_positions"] += rows_per_step * model_config["sequence_len"]
                state["cumulative_training_seconds"] += elapsed; session["compute_seconds"] += elapsed
                item = {"step": state["step"], "loss": loss, "lr_multiplier": lr, "step_seconds": elapsed,
                        "main_target_positions": state["main_target_positions"], "session_compute_seconds": session["compute_seconds"]}
                state["loss_history"].append(item); log.write(canonical(item) + "\n"); log.flush(); print(canonical(item), flush=True)
                if args.save_every > 0 and state["step"] % args.save_every == 0: save_main(checkpoint, model, optimizer, state, stream, torch)
                if args.stop_after_session_steps and state["step"] - session["start_step"] >= args.stop_after_session_steps:
                    stop_reason = "explicit_cpu_test_interruption"; break
        # A newly discovered exclusion conflict invalidates the candidate; never
        # label its checkpoint valid merely because its prior steps completed.
        stream.refresh()
    except (ValueError, FileNotFoundError) as error:
        write_json(args.output / "failure.json", {"status": "data_or_integrity_failure", "error": str(error),
                    "main_step": state["step"], "candidate_valid": False, "automatic_retry": False})
        raise
    session.update(end_step=state["step"], stop_reason=stop_reason, elapsed_seconds_to_checkpoint=time.time() - process_started)
    save_main(checkpoint, model, optimizer, state, stream, torch)
    validation = pilot.evaluate(model, val, token_bytes, rows=args.eval_rows, batch_rows=args.batch_rows, device=device, torch=torch)
    before = pilot.model_hash(model)
    loaded, loaded_side = inspect_owned(checkpoint, main=True, torch=torch)
    with torch.no_grad(): next(model.parameters()).view(-1)[0].add_(1)
    restore_main(loaded, loaded_side, model, optimizer, stream, config=config, pilot_digest=data_digest, torch=torch)
    if pilot.model_hash(model) != before: raise AssertionError("Main checkpoint reload hash mismatch")
    del loaded
    result = {"status": "tiny_main_cpu_test" if args.tiny_cpu_test else "main_session_completed",
              "stop_reason": stop_reason, "main_horizon_complete": state["step"] == config["max_steps"],
              "step": state["step"], "horizon_steps": config["max_steps"], "main_target_positions": state["main_target_positions"],
              "total_lineage_target_positions": state["pilot_target_positions"] + state["main_target_positions"],
              "cumulative_training_seconds": state["cumulative_training_seconds"], "session": session,
              "session_process_wall_seconds": time.time() - process_started,
              "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
              "checkpoint_reload_verified": True, "final_weights_sha256": before,
              "initial_weights_sha256": state["initial_weights_sha256"], "cursor": stream.cursor,
              "frozen_validation": validation, "candidate_valid": True,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "provider_stop_enforced_externally": True, "confirmatory_benchmarks_evaluated": False}
    write_json(args.output / "result.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
