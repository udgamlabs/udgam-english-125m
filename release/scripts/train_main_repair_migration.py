"""SDK-free, explicit metadata migration for the owned step6124 checkpoint.

This does not train, write a checkpoint, alter a budget guard, or authorize a
launch. The caller must obtain the receipt/stage hashes from its independently
verified final input seal. Only a copied state dictionary is changed; tensor,
optimizer and RNG objects are retained. The frozen restore_main remains the
authority for the numerical continuation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import train_main as core
import modal_transfer as files

FORMAT = "udgam-main-repair-migration-v1"
RECEIPT = "repair/receipt.json"
PLAN = "repair/plan.json"
SOURCE_NAMES = ("repair_main_suffix.py", "pack_main_repaired.py")
EVIDENCE = {
    "repair/evidence/receipt.json": "f5fbe95937e09c30ec36a129d6fcea4b6ade415a20418a39dfd78cee0171d4c0",
    "repair/evidence/manifest.json": "65a29a4b88191e8404ca0be12d6ab6a6562b2d9a2e7544ba31870bc9760afce1",
    "repair/evidence/evidence/hook/conflict.json": "7c8e529c47346a62dc916005caa5e171b1087c84a5864d165a70645adeb827b4",
    "repair/evidence/evidence/shard-exposure.json": "bae6fa641fc1c56f0049d955055786b3e8bfbd0070ca656adb5ce40f98967cbe",
    "repair/evidence/evidence/pilot-families.json": "82ca9c14bca08d9b5ab289d9da79768bff8ebf6ff410099aea5c887a394b74ac",
    "repair/evidence/evidence/benchmark-matches.json": "6353667ea5bc17379aaa9dd31acfcb3c2070c4f30aa358779cef7ee484e12885",
}
PRODUCTION = {
    "checkpoint_sha256": "9fc9fdfa07b202acee33c351542448ee55968bbd5311c079fdfc55a823637fdb",
    "old_stage_contract_sha256": "17a1761bfe727e213a110cba89d40907bc0c2bffc70fe8f761fd7b9f553d2acd",
    "source_parent": {
        "receipt_sha256": "7749963bb51a589d689be477ac8f721ef422fc1ccd975acf332ead9cd2474297",
        "manifest_sha256": "2988d4e9fc2291f342d191894700265bddc481e2d8d1ad92825f32079d204c72",
        "stage_manifest_sha256": "7a038a60b2612d062254ae6fef38530bc2c425958676b9b9d8a4c7fd08142235",
    },
    "evidence": EVIDENCE,
    "identity_hashes": [
        "11a29b1175530be2c879c58680c4a8de993b3bd1e39d69dab88f962b125b6880",
        "2e33de07f7f9b3a6afb3e679b2783e44f906cbeea25c5f7e56f47cfdbf797525",
        "38b20fe75bcc57e6c87e983c764b74edaef4bf4d473a4c311c9ca2f8cd2b99d4",
        "4435db99500194c531102f516f1d949b544a581f7d28511c68baa20f4985cdc5",
        "67cb7a1c181a0cc3e90969e896f7030d9a4bc0438face315097526bfda4d856b",
    ],
    "retired_document_ids": [1796455], "retired_encoded_positions": 332,
    "retired_ordinary_positions": 331, "retired_component": "finephrase_table",
    "protected_shards": 36, "protected_rows": 97986, "step": 6124,
    "cursor": {"shard_id": 35, "row_in_shard": 168, "total_rows": 97984},
    "tranche_before": 100000000, "tranche_after": 2000000000,
    "model": {"n_embd": 768, "n_head": 6, "n_kv_head": 2, "n_layer": 12,
        "sequence_len": 2048, "vocab_size": 16384, "window_pattern": "SSSL"},
    "horizon": {"accumulation_steps": 1, "batch_rows": 16,
        "declared_horizon_targets": 2000000000, "discarded_horizon_rounding_targets": 5120,
        "effective_horizon_targets": 1999994880, "final_lr_frac": .05, "max_steps": 61035,
        "model_compile": True, "optimizer_compile": True,
        "optimizer_transition": "fresh_MuonAdamW_once_after_annealed_owned_pilot",
        "seed": 20260922, "warmdown_ratio": .65, "warmup_steps": 40},
}
RECEIPT_FIELDS = {
    "format", "complete", "ready_for_training", "plan_sha256", "checkpoint_sha256",
    "source_parent", "forensic", "old_stage_contract", "new_stage_contract",
    "old_stage_contract_sha256", "new_stage_contract_sha256", "protected_shards",
    "protected_inventory_sha256", "protected_prefix_no_overlap",
    "source_cursor_before_sha256", "source_cursor_after_sha256",
    "retained_suffix_stream_sha256_before", "retained_suffix_stream_sha256_after",
    "retired_document_ids", "retired_encoded_positions", "retired_ordinary_positions",
    "identity_hashes", "counters_before", "counters_after", "allowed_diff", "schedule_migration",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected explicit SHA256 identity")
    return value


def portable_contract(stage):
    return {"config": {k: v for k, v in stage["config"].items()
            if k not in ("max_download_bytes", "max_output_bytes", "disk_reserve_bytes")},
        "lineage": {k: v for k, v in stage["lineage"].items()
            if k not in ("frozen_validation_path", "token_bytes_path")},
        "validation_policy": stage["validation_policy"]}


def _checked_json(root, relative, checksum):
    path = files.regular(root, relative)
    if files.digest(path) != _digest(checksum):
        raise ValueError("Sealed migration artifact hash mismatch: " + relative)
    return files.read_json(path)


def _validate_counters(receipt, protected, policy):
    before, after = receipt["counters_before"], receipt["counters_after"]
    if not isinstance(before, dict) or set(before) != set(after) or policy["retired_component"] not in before:
        raise ValueError("Repair component inventory changed")
    packed = {name: 0 for name in before}
    for shard in protected:
        counts = shard.get("ordinary_positions_by_component", {})
        if set(counts) - set(packed) or sum(counts.values()) != shard["ordinary_positions"]:
            raise ValueError("Protected component counts inconsistent")
        for name, count in counts.items():
            if type(count) is not int or count < 0: raise ValueError("Invalid protected token count")
            packed[name] += count
    for name, old in before.items():
        new = after[name]
        if set(old) != {"accepted", "packed", "weight"} or set(new) != set(old):
            raise ValueError("Unexpected component counter fields")
        for item in (old, new):
            if any(type(item[k]) is not int or item[k] < 0 for k in ("accepted", "packed")):
                raise ValueError("Invalid component token counters")
            if type(item["weight"]) not in (int, float) or not math.isfinite(item["weight"]) or item["weight"] <= 0:
                raise ValueError("Invalid component weight")
            if item["packed"] > item["accepted"]: raise ValueError("Packed count exceeds accepted supply")
        removed = policy["retired_ordinary_positions"] if name == policy["retired_component"] else 0
        if new != {"accepted": old["accepted"] - removed, "packed": packed[name], "weight": old["weight"]}:
            raise ValueError("Repair counter delta exceeds declared retirement/prefix reset")


def _validate_bundle(payload, side, stage_root, receipt_sha, stage_sha, policy, sources):
    """Internal test seam; the public entrypoint never accepts a policy override."""
    root = files.root_path(stage_root)
    receipt = _checked_json(root, RECEIPT, receipt_sha)
    stage = _checked_json(root, "manifest.json", stage_sha)
    if files.regular(root, "manifest.sha256").read_text().strip() != stage_sha:
        raise ValueError("Published stage manifest pair differs")
    if (stage.get("schema_version") != 1 or stage.get("status") != "initialized"
            or stage.get("network_request_unreconciled", False) is not False):
        raise ValueError("Repaired stage is blocked or incomplete")
    if (set(receipt) != RECEIPT_FIELDS or receipt["format"] != "udgam-main-suffix-repair-receipt-v1"
            or receipt["complete"] is not True or receipt["ready_for_training"] is not False
            or receipt["schedule_migration"] is not True or receipt["protected_prefix_no_overlap"] is not True):
        raise ValueError("Exact completed diagnostic-only repair receipt required")
    state = payload["state"]
    if "data_repair_migration" in state: raise ValueError("Parent already migrated; do not reapply")
    old = state["stage_contract"]
    if (side.get("sha256") != policy["checkpoint_sha256"] or receipt["checkpoint_sha256"] != side["sha256"]
            or receipt["old_stage_contract"] != old
            or canonical_hash(old) != policy["old_stage_contract_sha256"]
            or receipt["old_stage_contract_sha256"] != canonical_hash(old)):
        raise ValueError("Original checkpoint/contract identity mismatch")
    if (state["step"] != policy["step"] or state["cursor"] != policy["cursor"]
            or payload["model_config"] != policy["model"] or state["horizon_config"] != policy["horizon"]
            or state["main_target_positions"] != state["cursor"]["total_rows"] * payload["model_config"]["sequence_len"]):
        raise ValueError("Original model, horizon or consumed cursor changed")
    for field in ("initial_weights_sha256", "pilot_checkpoint_sha256", "pilot_data_manifest_sha256"):
        if side.get(field) != state.get(field) or not state.get(field):
            raise ValueError("Original scratch/pilot identity changed")
    if not payload.get("model") or not payload.get("optimizer") or not payload.get("rng"):
        raise ValueError("Exact continuation requires model, optimizer and RNG")
    if receipt["source_parent"] != policy["source_parent"]:
        raise ValueError("Wrong repaired data parent")
    evidence = policy["evidence"]
    forensic = {"receipt_sha256": evidence["repair/evidence/receipt.json"],
        "manifest_sha256": evidence["repair/evidence/manifest.json"],
        "conflict_sha256": evidence["repair/evidence/evidence/hook/conflict.json"]}
    if receipt["forensic"] != forensic: raise ValueError("Wrong diagnostic ancestry")
    for key in ("identity_hashes", "retired_document_ids", "retired_encoded_positions", "retired_ordinary_positions"):
        if canonical_hash(receipt[key]) != canonical_hash(policy[key]):
            raise ValueError("Undeclared excluded family or retirement")
    for before_key, after_key in (("source_cursor_before_sha256", "source_cursor_after_sha256"),
            ("retained_suffix_stream_sha256_before", "retained_suffix_stream_sha256_after")):
        a, b = receipt[before_key], receipt[after_key]
        if _digest(a) != _digest(b): raise ValueError("Repair changed retained source cursor or token stream")
    protected = state["accepted_shards"]
    if (len(protected) != policy["protected_shards"] or receipt["protected_shards"] != protected
            or canonical_hash(protected) != receipt["protected_inventory_sha256"]
            or stage["shards"][:len(protected)] != protected
            or sum(s["rows"] for s in protected) != policy["protected_rows"]):
        raise ValueError("Complete checkpoint-accepted prefix changed")
    for index, shard in enumerate(protected):
        names = {f"shards/train_{index:05d}.npy", f"shards/train_{index:05d}.provenance.jsonl.gz"}
        if shard["id"] != index or set(shard["hashes"]) != names:
            raise ValueError("Unexpected protected shard inventory")
        for name, expected in shard["hashes"].items():
            if files.digest(files.regular(root, name)) != _digest(expected):
                raise ValueError("Protected shard bytes changed")
    _validate_counters(receipt, protected, policy)
    new = receipt["new_stage_contract"]
    if portable_contract(stage) != new or canonical_hash(new) != receipt["new_stage_contract_sha256"]:
        raise ValueError("Repaired stage and declared new contract differ")
    source_hashes = {name: files.digest(files.regular(sources, name)) for name in SOURCE_NAMES}
    additions = {PLAN: receipt["plan_sha256"],
        **{"repair/source/" + name: h for name, h in source_hashes.items()}, **evidence}
    expected_diff = {"config.tranche_tokens": {"before": policy["tranche_before"], "after": policy["tranche_after"]},
        "lineage_additions": ["data_repair"], "metadata_additions": additions}
    if receipt["allowed_diff"] != expected_diff: raise ValueError("Repair diff exceeds reviewed path allowlist")
    expected = copy.deepcopy(old)
    if (old["config"]["tranche_tokens"] != policy["tranche_before"]
            or old["config"]["stage_tokens"] != policy["tranche_after"]
            or "data_repair" in old["lineage"]
            or set(additions) & set(old["lineage"]["metadata_artifact_hashes"])):
        raise ValueError("Unexpected old data schedule/provenance")
    expected["config"]["tranche_tokens"] = policy["tranche_after"]
    expected["lineage"]["data_repair"] = {"plan_path": PLAN, "plan_sha256": receipt["plan_sha256"],
        "source_sha256": source_hashes}
    expected["lineage"]["metadata_artifact_hashes"].update(additions)
    if canonical_hash(new) != canonical_hash(expected):
        raise ValueError("Unreviewed config, guard, holdout or lineage change")
    for name, expected_hash in new["lineage"]["metadata_artifact_hashes"].items():
        if files.digest(files.regular(root, name)) != _digest(expected_hash):
            raise ValueError("Immutable stage provenance changed")
    if files.digest(files.regular(root, "tokenizer.json")) != new["lineage"]["tokenizer_sha256"]:
        raise ValueError("Owned tokenizer changed")
    plan = _checked_json(root, PLAN, receipt["plan_sha256"])
    if plan.get("format") != "udgam-main-suffix-repair-plan-v1": raise ValueError("Unexpected repair plan")
    # Required plan fields are deliberately redundant with the completed receipt.
    for key in ("checkpoint_sha256", "source_parent", "forensic", "old_stage_contract",
                "old_stage_contract_sha256", "protected_shards", "protected_inventory_sha256",
                "identity_hashes", "retired_document_ids", "source_cursor_before_sha256", "counters_before"):
        if plan.get(key) != receipt[key]: raise ValueError("Plan and completed repair disagree: " + key)
    if (plan.get("source_sha256") != source_hashes or plan.get("eligibility_policy_changed") is not False
            or plan.get("schedule_migration") is not True
            or plan.get("order_policy") != "protected-prefix; recovered old stream including old pending tail; new documents by original order_key,id"):
        raise ValueError("Plan source, eligibility or future ordering differs")
    return receipt, new


def _migrate_verified(payload, side, stage_root, *, expected_receipt_sha256,
                      expected_stage_manifest_sha256, policy, sources):
    receipt, new = _validate_bundle(payload, side, stage_root, expected_receipt_sha256,
        expected_stage_manifest_sha256, policy, sources)
    proof = {"format": FORMAT, "original_checkpoint_sha256": side["sha256"],
        "old_stage_contract_sha256": receipt["old_stage_contract_sha256"],
        "new_stage_contract_sha256": receipt["new_stage_contract_sha256"],
        "repair_receipt_path": RECEIPT, "repair_receipt_sha256": expected_receipt_sha256,
        "repair_plan_sha256": receipt["plan_sha256"],
        "initial_repaired_stage_manifest_sha256": expected_stage_manifest_sha256,
        "protected_inventory_sha256": receipt["protected_inventory_sha256"],
        "original_cursor": copy.deepcopy(payload["state"]["cursor"]),
        "original_step": payload["state"]["step"],
        "horizon_sha256": canonical_hash(payload["state"]["horizon_config"]),
        "helper_sha256": files.digest(Path(__file__)),
        "only_declared_future_data_contract_changed": True,
        "model_optimizer_rng_cursor_preserved": True}
    result = dict(payload)
    result["state"] = dict(payload["state"])
    result["state"]["stage_contract"] = copy.deepcopy(new)
    result["state"]["data_repair_migration"] = proof
    return result, side, copy.deepcopy(proof)


def inspect_and_migrate(parent_path, stage_directory, *, expected_receipt_sha256,
                        expected_stage_manifest_sha256, torch):
    """First transition only, under the caller's separately checked resource guard.

    Returns (payload, original_sidecar, proof). Pass payload to unchanged
    restore_main. Keep state.data_repair_migration in every saved checkpoint and
    copy proof to the run contract and result. Later exact resumes must verify
    this proof/receipt, not apply the migration to another arbitrary parent.
    """
    path = Path(parent_path)
    files.regular(path.parent, path.name)
    side = files.read_json(files.regular(path.parent, path.with_suffix(".sha256.json").name))
    if side.get("sha256") != PRODUCTION["checkpoint_sha256"]:
        raise ValueError("Only the original verified step6124 checkpoint may migrate")
    payload, side = core.inspect_owned(path, main=True, torch=torch)
    return _migrate_verified(payload, side, stage_directory,
        expected_receipt_sha256=_digest(expected_receipt_sha256),
        expected_stage_manifest_sha256=_digest(expected_stage_manifest_sha256),
        policy=PRODUCTION, sources=Path(__file__).resolve().parent)
