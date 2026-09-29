#!/usr/bin/env python3
"""Offline pinned HF base-model comparison using evaluate_dense's frozen policy.

No network client, download command, training, or provider operation exists here.
Use a separate pinned evaluation environment and hash-verified local snapshots.
CUDA BF16 is mandatory for the CLI; tiny CPU tests call the pure adapters only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import evaluate_dense as shared

ROOT = shared.ROOT
PINS = ROOT / "research/baseline-pins"
MODELS = {
    "sarvam-1": ("sarvamai/sarvam-1", "e9607337286ddf496d4a2562b194e489dcf3feea"),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
}
VERSIONS = {"torch": "2.9.1", "transformers": "5.17.0", "huggingface-hub": "1.32.0",
            "tokenizers": "0.23.2", "safetensors": "0.8.0", "numpy": "2.3.5"}
SNAPSHOT_FORMAT = "udgam-pinned-hf-inference-snapshot-v1"
MANIFEST_NAME = "baseline-snapshot.json"
INFERENCE_NAMES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt",
                   "model.safetensors", "model.safetensors.index.json",
                   "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
DOCUMENT_NAMES = {"README.md", "LICENSE.md"}


def metadata(model, pins=PINS):
    repo, revision = MODELS[model]
    path = Path(pins) / (repo.replace("/", "--") + "--blobs.json")
    value = json.loads(path.read_text())
    if value.get("id") != repo or value.get("sha") != revision or value.get("config", {}).get("model_type") != "llama":
        raise ValueError("Pinned baseline metadata identity changed")
    entries = {r["rfilename"]: r for r in value["siblings"] if r["rfilename"] in INFERENCE_NAMES | DOCUMENT_NAMES}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(entries):
        raise ValueError("Incomplete pinned tokenizer/config inventory")
    weights = {n for n in entries if n.endswith(".safetensors")}
    expected = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"} if model == "sarvam-1" else {"model.safetensors"}
    if weights != expected or any("lfs" not in entries[n] for n in weights):
        raise ValueError("Pinned safetensors inventory changed")
    return value, entries, shared.sha256(path)


def file_hashes(path):
    size = path.stat().st_size
    git = hashlib.sha1(f"blob {size}\0".encode()); sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk); git.update(chunk)
    return size, sha.hexdigest(), git.hexdigest()


def inspect_snapshot(model, directory, *, pins=PINS):
    """Check published LFS SHA256 or Git blob SHA1, not a claimed revision label.

    Hub cache symlinks are allowed; each target is hashed. Extraneous files such
    as Python modules, pickle weights and adapters are rejected. .cache metadata
    from snapshot_download(local_dir=...) is not model input and is excluded.
    """
    directory = Path(directory)
    meta, entries, metadata_hash = metadata(model, pins)
    if not directory.is_dir():
        raise ValueError("A complete local snapshot is required; no downloads are performed")
    actual = {p.name for p in directory.iterdir() if p.name not in {MANIFEST_NAME, ".cache", ".gitattributes"}}
    if actual != set(entries):
        raise ValueError(f"Snapshot allowlist mismatch: missing={sorted(set(entries) - actual)}, extra={sorted(actual - set(entries))}")
    files = {}
    for name, entry in sorted(entries.items()):
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError("Snapshot file missing or published size differs: " + name)
        size, sha, git = file_hashes(path)
        if (sha != entry["lfs"]["sha256"] if "lfs" in entry else git != entry["blobId"]):
            raise ValueError("Snapshot published content hash mismatch: " + name)
        files[name] = {"bytes": size, "sha256": sha}
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "llama" or config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("Only native unquantized Llama baselines are supported")
    if config.get("max_position_embeddings", 0) < 2048:
        raise ValueError("Baseline cannot support the common context policy")
    if "model.safetensors.index.json" in entries:
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        if set(index["weight_map"].values()) != {n for n in entries if n.endswith(".safetensors")}:
            raise ValueError("Safetensors index points outside the pinned weight set")
    return {"format": SNAPSHOT_FORMAT, "model": model, "repo": MODELS[model][0], "revision": MODELS[model][1],
            "metadata_sha256": metadata_hash, "files": files,
            "inference_runtime_file_bytes": sum(f["bytes"] for n, f in files.items() if n in INFERENCE_NAMES),
            "weight_file_bytes": sum(f["bytes"] for n, f in files.items() if n.endswith(".safetensors")),
            "snapshot_bytes_including_documentation": sum(f["bytes"] for f in files.values()),
            "published_parameter_count": meta["safetensors"]["total"],
            "license_scope": "Sarvam AI Research License; research/non-commercial only" if model == "sarvam-1" else "Apache-2.0",
            "trust_remote_code": False, "safetensors_only": True}


def verify_snapshot(model, directory, *, pins=PINS):
    observed = inspect_snapshot(model, directory, pins=pins)
    if json.loads((Path(directory) / MANIFEST_NAME).read_text()) != observed:
        raise ValueError("Snapshot differs from the sealed, published-hash verified manifest")
    return observed


def artifact_bytes(snapshot, directory):
    """Full sealed inference bundle including docs/license and its manifest.

    Common installed Python/CUDA dependencies and duplicate Hub cache bytes are
    excluded, just as installed dependencies are excluded from native exports.
    Native vendored model code remains part of that native inference bundle.
    """
    return snapshot["snapshot_bytes_including_documentation"] + (Path(directory) / MANIFEST_NAME).stat().st_size


class TokenizerAdapter:
    """Never add implicit specials or a chat template; shared code adds one BOS."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.bos_policy = "native_bos"
        if self.bos_id is None:
            self.bos_id = tokenizer.eos_token_id
            self.bos_policy = "fallback_eos_because_native_bos_absent"
        if type(self.bos_id) is not int or self.bos_id < 0:
            raise ValueError("Tokenizer requires an explicit native BOS or declared EOS fallback")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.bos_id

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def get_bos_token_id(self):
        return self.bos_id


def continuation_scores(model, encoded, *, device, torch, pad_id, batch_size=4):
    """Same FP32 log-softmax/FP64 sum as shared evaluator; right pads masked."""
    scores = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            group = encoded[start:start + batch_size]
            if any(not item["supported"] for item in group):
                raise ValueError("Unsupported inputs must never enter scoring")
            width = max(len(item["tokens"]) - 1 for item in group)
            if width < 1:
                raise ValueError("At least a BOS input is required")
            x = torch.full((len(group), width), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(x)
            for i, item in enumerate(group):
                length = len(item["tokens"]) - 1
                x[i, :length] = torch.tensor(item["tokens"][:-1], device=device)
                mask[i, :length] = 1
            logits = model(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True).logits
            for i, item in enumerate(group):
                begin = item["answer_start"] - 1; end = len(item["tokens"]) - 1
                targets = torch.tensor(item["tokens"][item["answer_start"]:], device=device)
                logprobs = torch.log_softmax(logits[i, begin:end].float(), dim=-1)
                value = float(logprobs.gather(-1, targets[:, None]).double().sum())
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite baseline answer likelihood")
                scores.append(value)
    return scores


def forced_latency(model, tokenizer, torch, device, deadline, *, clock=time):
    """Identical public prompt strings, five warmups, 128 forced greedy tokens."""
    prompts = shared.fixed_latency_prompts(); results = []; warmups = 0
    def sync():
        if device == "cuda": torch.cuda.synchronize()
    for index, text in enumerate([prompts[0]] * 5 + prompts):
        if clock.time() >= deadline:
            break
        prompt = [tokenizer.get_bos_token_id()] + tokenizer.encode(text)
        if len(prompt) + 128 > 2048:
            if index >= 5: results.append({"prompt_index": index - 5, "supported": False})
            continue
        generated = []
        with torch.inference_mode():
            sync(); start = clock.perf_counter()
            x = torch.tensor([prompt], device=device)
            out = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=True, return_dict=True)
            token = out.logits[:, -1:, :].argmax(-1); generated.append(int(token.item()))
            cache = out.past_key_values
            del out
            sync(); first = clock.perf_counter()
            for step in range(127):
                # No padding in single-prompt decode. Explicit full mask grows
                # with the cache; Llama derives absolute positions from cache.
                mask = torch.ones((1, len(prompt) + step + 1), dtype=torch.long, device=device)
                out = model(input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
                token = out.logits[:, -1:, :].argmax(-1); generated.append(int(token.item()))
                cache = out.past_key_values
                del out
            sync(); finish = clock.perf_counter()
        del cache
        if index < 5:
            warmups += 1
        else:
            output_bytes = len(tokenizer.decode(generated).encode())
            results.append({"prompt_index": index - 5, "supported": True, "prompt_utf8_bytes": len(text.encode()),
                "prompt_tokens": len(prompt), "forced_output_tokens": 128, "output_utf8_bytes": output_bytes,
                "ttft_seconds": first - start, "decode_seconds": finish - first,
                "decode_tokens_per_second": 127 / (finish - first),
                "total_output_bytes_per_second": output_bytes / (finish - start), "elapsed_seconds": finish - start})
    valid = [r for r in results if r["supported"]]
    return {"kind": "forced_128_token_microbenchmark_not_response_latency", "warmups_requested": 5,
            "warmups_completed": warmups, "complete": len(results) == 30 and warmups == 5,
            "prompt_strings_sha256": shared.digest(shared.canonical(prompts)), "results": results,
            "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in valid) if valid else None,
            "p95_ttft_seconds": sorted(r["ttft_seconds"] for r in valid)[math.ceil(.95 * len(valid)) - 1] if valid else None,
            "median_decode_tokens_per_second": statistics.median(r["decode_tokens_per_second"] for r in valid) if valid else None,
            "stops_on_eos": False, "quality_evidence": False}


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if any(versions[name].split("+")[0] != version for name, version in VERSIONS.items()):
        raise ValueError("Use the isolated pinned evaluation environment; runtime versions differ")
    return versions


def load_model(directory, *, torch, auto_model, auto_tokenizer):
    tokenizer = TokenizerAdapter(auto_tokenizer.from_pretrained(str(directory), local_files_only=True,
        trust_remote_code=False, use_fast=True, token=False))
    model = auto_model.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, attn_implementation="sdpa", token=False)
    model.to("cuda"); model.eval()
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("Baseline parameters are not uniformly BF16")
    return model, tokenizer


def validate_development_gate(directory, model_identity, task_hash, development_directory):
    """Dev evidence checks implementation/timing, never chooses by accuracy."""
    directory = Path(directory)
    _, suite = shared.load_development_smoke(development_directory)
    requests = shared.make_requests(suite)
    contract = json.loads((directory / "evaluation-contract.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    request_hash = shared.digest("".join(shared.canonical(r) + "\n" for r in requests))
    if (contract.get("policy") != shared.POLICY or contract.get("policy_sha256") != shared.digest(shared.canonical(shared.POLICY)) or
        contract.get("model_identity") != model_identity or contract.get("suite_kind") != "development_smoke_not_confirmatory" or
        contract.get("task_source_manifest_sha256") != task_hash or
        contract.get("benchmark_manifest_sha256") != shared.sha256(Path(development_directory) / "manifest.json") or
        contract.get("requests_sha256") != request_hash or shared.sha256(directory / "requests.jsonl") != request_hash):
        raise ValueError("Dev-100 evidence is for a different baseline/policy/source")
    items, _ = shared.cached_items(directory / "items.jsonl", requests, contract)
    if (len(items) != 100 or not summary.get("complete") or summary.get("contract") != contract or
        summary.get("items_sha256") != shared.sha256(directory / "items.jsonl") or
        not summary.get("cost_history_complete") or
        not math.isfinite(summary.get("total_recorded_process_seconds", float("nan"))) or
        summary["total_recorded_process_seconds"] <= 0):
        raise ValueError("Complete, intact dev-100 timing evidence is required before final evaluation")
    for item, request in zip(items, requests, strict=True):
        if (any(item.get(key) != request[key] for key in ("task", "gold", "primary_metric", "document_sha256")) or
            any(type(item.get(key)) is not int or item[key] not in (0, 1) for key in ("acc", "acc_norm", "primary_correct")) or
            type(item.get("supported")) is not bool or (not item["supported"] and item["primary_correct"])):
            raise ValueError("Dev cache must contain complete shared-policy scoring records")
    if summary.get("development_macro_accuracy") != shared.summarize(items, requests)["Q"]:
        raise ValueError("Dev summary does not match its scoring cache")
    return {"summary_sha256": shared.sha256(directory / "summary.json"), "items_sha256": summary["items_sha256"],
            "development_seconds": summary["total_recorded_process_seconds"],
            "quality_used_for_selection": False}


def main(argv=None):
    process_started = time.time()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("seal", "evaluate"))
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--development-smoke", type=Path)
    p.add_argument("--development-evidence", type=Path)
    p.add_argument("--development-data", type=Path, default=ROOT / "data/benchmark-smoke-001")
    p.add_argument("--benchmarks", type=Path, default=ROOT / "data/benchmark-exclusions-002")
    p.add_argument("--task-definitions", type=Path, default=ROOT / "research/harness-task-definitions")
    p.add_argument("--guard-evidence", type=Path)
    p.add_argument("--max-eval-seconds", type=float, default=180)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--resume-evaluation", action="store_true")
    p.add_argument("--latency", action="store_true")
    a = p.parse_args(argv)
    if a.command == "seal":
        result = inspect_snapshot(a.model, a.snapshot)
        target = a.snapshot / MANIFEST_NAME
        if target.exists() and json.loads(target.read_text()) != result:
            raise ValueError("Refusing to replace a different snapshot identity")
        shared.write_json(target, result)
        print(json.dumps({"model": a.model, "revision": result["revision"], "inference_artifact_bytes": artifact_bytes(result, a.snapshot)}))
        return result
    if (not a.output or not a.guard_evidence or not math.isfinite(a.max_eval_seconds) or
        not 0 < a.max_eval_seconds <= 600 or not 1 <= a.batch_size <= 8 or
        (a.output.exists() != a.resume_evaluation)):
        raise ValueError("Output/resume state, fresh guard and bounded evaluation time/batch are required")
    if not a.development_smoke and not a.development_evidence:
        raise ValueError("Complete this model's dev-100 timing run before final-2500")
    raw_guard = json.loads(a.guard_evidence.read_text())
    validator = shared.main_training.validate_main_guard if raw_guard.get("profile") == "main-v1" else shared.training.validate_guard
    guard, guard_stop = validator(a.guard_evidence, export_reserve_seconds=120)
    deadline = min(process_started + a.max_eval_seconds, guard_stop)
    snapshot = verify_snapshot(a.model, a.snapshot)
    versions = runtime_versions()
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 device required; no real-model CPU evaluation is supported")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(a.snapshot, torch=torch, auto_model=AutoModelForCausalLM, auto_tokenizer=AutoTokenizer)
    torch.cuda.synchronize()
    cold_load = time.time() - process_started
    loaded_memory = torch.cuda.memory_allocated()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != snapshot["published_parameter_count"]:
        raise ValueError("Loaded parameter count differs from pinned upstream metadata")
    identity = {"model": a.model, "repo": snapshot["repo"], "revision": snapshot["revision"],
        "snapshot_sha256": shared.digest(shared.canonical(snapshot)), "runtime_versions": versions,
        "adapter_sha256": shared.sha256(Path(__file__)), "shared_evaluator_sha256": shared.sha256(Path(shared.__file__)),
        "attention_backend": "sdpa", "parameter_dtype": "bfloat16", "bos_token_id": tokenizer.bos_id,
        "bos_policy": tokenizer.bos_policy, "padding": "right_masked", "batch_size": a.batch_size,
        "python_version": sys.version, "device": {"name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}}
    benchmark, suite = shared.load_development_smoke(a.development_smoke) if a.development_smoke else shared.load_frozen_suite(a.benchmarks)
    source = shared.verify_task_definitions(a.task_definitions)
    task_hash = shared.sha256(a.task_definitions / "manifest.json")
    dev_gate = None
    if not a.development_smoke:
        if not a.development_evidence:
            raise ValueError("Complete this model's dev-100 timing run before final-2500")
        dev_gate = validate_development_gate(a.development_evidence, identity, task_hash, a.development_data)
    requests = shared.make_requests(suite)
    request_text = "".join(shared.canonical(r) + "\n" for r in requests)
    contract = {"policy": shared.POLICY, "policy_sha256": shared.digest(shared.canonical(shared.POLICY)),
        "model_identity": identity, "checkpoint_sha256": identity["snapshot_sha256"],
        "benchmark_manifest_sha256": shared.sha256((a.development_smoke or a.benchmarks) / "manifest.json"),
        "task_source_manifest_sha256": task_hash, "harness_revision": source["revision"],
        "requests_sha256": shared.digest(request_text), "declared_items": len(requests),
        "suite_kind": "development_smoke_not_confirmatory" if a.development_smoke else "frozen_final_suite",
        "development_gate": dev_gate, "scoring_dtype": "bfloat16", "no_test_feedback_for_checkpoint_selection": True,
        "scope": "Pinned base-model completion comparison; no instruction-following or general superiority claim"}
    if a.resume_evaluation:
        if json.loads((a.output / "evaluation-contract.json").read_text()) != contract or shared.sha256(a.output / "requests.jsonl") != contract["requests_sha256"]:
            raise ValueError("Response cache belongs to another model, policy, runtime or request set")
    else:
        a.output.mkdir(parents=True)
        (a.output / "requests.jsonl").write_text(request_text)
        shared.write_json(a.output / "evaluation-contract.json", contract)
    items, chain = shared.cached_items(a.output / "items.jsonl", requests, contract)
    prior_count = len(items)
    previous = json.loads((a.output / "summary.json").read_text()) if (a.output / "summary.json").exists() else None
    cost_complete = not a.resume_evaluation or (previous is not None and previous.get("cost_history_complete", False) and
        previous.get("items_sha256") == shared.sha256(a.output / "items.jsonl"))
    scorer = lambda pairs: continuation_scores(model, pairs, device="cuda", torch=torch, pad_id=tokenizer.pad_id, batch_size=a.batch_size)
    with (a.output / "items.jsonl").open("a", encoding="utf-8") as log:
        for request in requests[prior_count:]:
            if time.time() >= deadline: break
            item = shared.score_request(request, tokenizer, scorer)
            chain = shared.digest(chain + shared.canonical(item))
            log.write(shared.canonical({**item, "cache_chain_sha256": chain}) + "\n"); log.flush(); os.fsync(log.fileno())
            items.append(item)
    summary = shared.summarize(items, requests)
    if a.development_smoke:
        summary["development_macro_accuracy"] = summary.pop("Q")
        summary["quality_floors_pass"] = summary["macro_floor_pass"] = None
        summary["quality_gate_scope"] = "No confirmatory quality gate on 100 development examples"
    summary.update(contract=contract, total_parameters=parameters, checkpoint_sha256=identity["snapshot_sha256"],
        snapshot=snapshot, inference_artifact_bytes=artifact_bytes(snapshot, a.snapshot),
        artifact_bytes_scope="Sealed local inference bundle including tokenizer/config/weights/docs/license/manifest; installed runtime dependencies and Hub cache copies excluded",
        parameter_tensor_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        cold_load_seconds=cold_load, cold_load_scope="Standalone HF inference files: published hash verification, imports, tokenizer/model load, CPU-to-CUDA transfer; no optimizer",
        loaded_cuda_memory_bytes=loaded_memory, attention_backend="sdpa", dtype="torch.bfloat16",
        gpu_name=torch.cuda.get_device_name(), gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        evaluation_seconds=time.time() - process_started, reused_cached_items=prior_count,
        items_sha256=shared.sha256(a.output / "items.jsonl"), cash_cap_enforced_by_this_script=False)
    # Persist scored results before optional latency. Interrupted latency leaves
    # a useful intact cache but incomplete cost history, never an invented cost.
    summary["cost_history_complete"] = False
    summary["sessions"] = previous.get("sessions", []) if previous else []
    summary["total_recorded_process_seconds"] = sum(s["elapsed_seconds_including_verification_load_scoring_latency"] for s in summary["sessions"])
    summary["estimated_recorded_compute_cost_usd"] = sum(s["estimated_compute_cost_usd"] for s in summary["sessions"])
    shared.write_json(a.output / "summary.json", summary)
    if previous and previous.get("latency", {}).get("complete"):
        summary["latency"] = previous["latency"]
    elif a.latency and time.time() < deadline:
        summary["latency"] = forced_latency(model, tokenizer, torch, "cuda", deadline)
    seconds = time.time() - process_started
    rate = guard["hourly_total_usd"]
    session = {"elapsed_seconds_including_verification_load_scoring_latency": seconds, "new_items": len(items) - prior_count,
               "hourly_total_usd": rate, "estimated_compute_cost_usd": seconds * rate / 3600}
    summary["sessions"] = (previous.get("sessions", []) if previous else []) + [session]
    summary["cost_history_complete"] = bool(cost_complete)
    summary["total_recorded_process_seconds"] = sum(s["elapsed_seconds_including_verification_load_scoring_latency"] for s in summary["sessions"])
    summary["estimated_recorded_compute_cost_usd"] = sum(s["estimated_compute_cost_usd"] for s in summary["sessions"])
    summary["cost_scope"] = "Recorded evaluator sessions only; external downloads/setup/storage and interrupted sessions require the provider ledger"
    summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    summary["process_peak_ram_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    shared.write_json(a.output / "summary.json", summary)
    print(json.dumps({"model": a.model, "complete": summary["complete"], "scored": len(items), "Q": summary.get("Q"),
                      "development_macro_accuracy": summary.get("development_macro_accuracy")}))
    return summary


if __name__ == "__main__":
    main()
