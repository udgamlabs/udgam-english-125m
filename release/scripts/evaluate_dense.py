#!/usr/bin/env python3
"""Offline, frozen zero-shot completion scoring for an owned scratch checkpoint.

Task strings/character normalization follow the pinned EleutherAI definitions.
The explicit shared model policy is one native BOS, no EOS, no chat template,
no truncation, and FP32 log-softmax. Over-context or ambiguous token boundaries
make the entire item unsupported/incorrect. This is a custom 2,500-item suite,
not a stock leaderboard run. No downloads, retrieval or tools are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import resource
import shutil
import statistics
import sys
import time

import train_dense as training
import train_main as main_training
from prepare_dense_tokens import ROOT, REVISION, VENDOR, load_portable_tokenizer, sha256, verify_prepared, write_json

HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
TASKS = ("arc_challenge", "hellaswag", "piqa", "boolq", "winogrande")
NORMALIZED = {"arc_challenge", "hellaswag", "piqa"}
FLOORS = {"arc_challenge": .30, "hellaswag": .35, "piqa": .60, "boolq": .65, "winogrande": .55}
POLICY = {"version": "english-small-v1-explicit-bos", "fewshot": 0, "chat_template": False,
          "context_limit_including_bos_and_answer": 2048, "prefix": "one_native_BOS; HF baseline must declare fallback EOS if BOS absent",
          "append_eos": False, "normalization": "Unicode code points in original answer choice; excludes added delimiter",
          "boundary": "move trailing context whitespace; joint encoding must preserve the context token prefix",
          "overflow": "whole item unsupported and incorrect; never truncate", "ties": "first choice",
          "log_softmax_dtype": "float32", "sum_dtype": "float64", "external_tools": False}
SFT_CHECKPOINT_FORMAT = "udgam-owned-human-sft-v1"


def sft_adapter_module():
    # Main/pilot packages intentionally do not ship instruction-training code.
    # Load it only when an explicit owned SFT checkpoint is selected.
    import sft_checkpoint_adapter
    return sft_checkpoint_adapter


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load_frozen_suite(directory):
    """Verify full split bytes and exact registered selection without resampling."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "frozen_before_training_acquisition" or manifest.get("selection_seed") != 20260922:
        raise ValueError("Benchmark selection is not the registered frozen suite")
    if set(manifest["tasks"]) != set(TASKS):
        raise ValueError("Frozen suite tasks changed")
    result = {}
    for task in TASKS:
        info = manifest["tasks"][task]
        path = directory / Path(info["local_file"]).name
        if sha256(path) != info["sha256"]:
            raise ValueError("Frozen benchmark file hash mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        if len(docs) != info["total_rows"] or len(info["selected"]) != 500:
            raise ValueError("Frozen benchmark row count mismatch")
        seen, expected = set(), []
        for index, doc in enumerate(docs):
            h = digest(canonical(doc))
            if h in seen:
                continue
            seen.add(h)
            key = digest(f"english-small-v1|20260922|{task}|{index}|{h}")
            expected.append({"original_split_row_index": index, "document_sha256": h, "selection_hash": key,
                             "original_id": doc.get("id", doc.get("ind", doc.get("qID")))})
        expected.sort(key=lambda row: row["selection_hash"])
        if expected[:500] != info["selected"]:
            raise ValueError("Frozen selection identity/order mismatch")
        result[task] = [{**identity, "document": docs[identity["original_split_row_index"]]}
                        for identity in info["selected"]]
    return manifest, result


def load_development_smoke(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Development manifest integrity failure")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "udgam-benchmark-development-smoke-v1" or manifest.get("status") != "ready_development_only" or set(manifest["tasks"]) != set(TASKS):
        raise ValueError("Complete development-only smoke set required")
    suite = {}
    for task in TASKS:
        info = manifest["tasks"][task]
        if info["split"] != "train" or Path(info["file"]).name != info["file"] or sha256(directory / info["file"]) != info["sha256"]:
            raise ValueError("Development source split/hash mismatch")
        with gzip.open(directory / info["file"], "rt") as f:
            entries = [json.loads(line) for line in f]
        if len(entries) != 20 or len({r["original_split_row_index"] for r in entries}) != 20:
            raise ValueError("Development sample count/identity mismatch")
        for row in entries:
            if digest(canonical(row["document"])) != row["document_sha256"]:
                raise ValueError("Development document hash mismatch")
        suite[task] = entries
    return manifest, suite


def verify_task_definitions(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"lm_eval/tasks/arc/arc_challenge.yaml", "lm_eval/tasks/arc/arc_easy.yaml",
                "lm_eval/tasks/hellaswag/hellaswag.yaml", "lm_eval/tasks/hellaswag/utils.py",
                "lm_eval/tasks/piqa/piqa.yaml", "lm_eval/tasks/super_glue/boolq/default.yaml",
                "lm_eval/tasks/winogrande/default.yaml", "lm_eval/tasks/winogrande/preprocess_winogrande.py",
                "lm_eval/api/task.py", "lm_eval/api/model.py", "lm_eval/config/task.py"}
    if manifest.get("revision") != HARNESS_REVISION or not required <= set(manifest["files"]):
        raise ValueError("Pinned task definitions incomplete or wrong revision")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(directory / name) != expected:
            raise ValueError("Pinned task source hash mismatch")
    return manifest


def hellaswag_clean(text):
    # Exact small helper from the pinned upstream utils.py, not prompt tuning.
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def task_request(task, doc):
    """Return tokenizer-independent strings shared unchanged with HF baselines."""
    if task == "arc_challenge":
        contexts = [f"Question: {doc['question']}\nAnswer:"] * len(doc["choices"]["text"])
        choices = doc["choices"]["text"]
        gold = doc["choices"]["label"].index(doc["answerKey"])
    elif task == "hellaswag":
        context = hellaswag_clean(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        choices = [hellaswag_clean(t) for t in doc["endings"]]
        contexts = [context] * len(choices); gold = int(doc["label"])
    elif task == "piqa":
        choices = [doc["sol1"], doc["sol2"]]
        contexts = [f"Question: {doc['goal']}\nAnswer:"] * 2; gold = int(doc["label"])
    elif task == "boolq":
        choices = ["no", "yes"]
        contexts = [f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"] * 2; gold = int(doc["label"])
    elif task == "winogrande":
        if doc["sentence"].count("_") != 1:
            raise ValueError("WinoGrande requires one blank")
        before, after = doc["sentence"].split("_")
        contexts = [before + doc["option1"], before + doc["option2"]]
        choices = [after.strip()] * 2; gold = {"1": 0, "2": 1}[doc["answer"]]
    else:
        raise ValueError("Unsupported task")
    if not 0 <= gold < len(choices) or any(not isinstance(x, str) or not x for x in choices):
        raise ValueError("Invalid task choices")
    return {"task": task, "gold": gold, "contexts": contexts,
            "continuations": [" " + text for text in choices],
            "normalization_lengths": [len(text) for text in (contexts if task == "winogrande" else choices)],
            "primary_metric": "acc_norm" if task in NORMALIZED else "acc"}


def make_requests(suite):
    requests = []
    for task in TASKS:
        for row in suite[task]:
            request = task_request(task, row["document"])
            request.update({k: v for k, v in row.items() if k != "document"})
            request["request_sha256"] = digest(canonical(request))
            requests.append(request)
    return requests


def encode_pair(encode, bos_id, context, continuation, max_length=2048):
    """BPE boundaries are joined, never independently concatenated or guessed."""
    if not isinstance(bos_id, int) or not continuation:
        raise ValueError("Explicit native BOS and nonempty continuation required")
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation; context = context[:-trailing]
    ctx = list(encode(context)); whole = list(encode(context + continuation))
    if whole[:len(ctx)] != ctx:
        return {"supported": False, "reason": "tokenizer_boundary_prefix_changed"}
    answer = whole[len(ctx):]
    if not answer:
        return {"supported": False, "reason": "empty_tokenized_continuation"}
    tokens = [bos_id] + whole
    if len(tokens) > max_length:
        return {"supported": False, "reason": "over_context_limit", "total_tokens": len(tokens)}
    return {"supported": True, "tokens": tokens, "answer_start": 1 + len(ctx),
            "answer_tokens": len(answer), "prompt_tokens": 1 + len(ctx), "total_tokens": len(tokens)}


def continuation_scores(model, encoded, *, device, torch, batch_size=4):
    """Right padding cannot affect previous causal positions; only answers count."""
    scores = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            group = encoded[start:start + batch_size]
            if any(not item["supported"] for item in group):
                raise ValueError("Unsupported inputs must never enter model scoring")
            width = max(len(item["tokens"]) - 1 for item in group)
            if width < 2:
                raise ValueError("Dense smear path requires two input positions")
            x = torch.zeros((len(group), width), dtype=torch.long, device=device)
            for i, item in enumerate(group):
                x[i, :len(item["tokens"]) - 1] = torch.tensor(item["tokens"][:-1], device=device)
            with training.autocast(torch, device):
                logits = model(x)
            for i, item in enumerate(group):
                begin = item["answer_start"] - 1
                end = len(item["tokens"]) - 1
                answer_logits = logits[i, begin:end].float()
                targets = torch.tensor(item["tokens"][item["answer_start"]:], device=device)
                logprobs = torch.log_softmax(answer_logits, dim=-1)
                value = float(logprobs.gather(-1, targets[:, None]).double().sum())
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite answer likelihood")
                scores.append(value)
    return scores


def score_request(request, tokenizer, scorer, *, max_length=2048):
    pairs = [encode_pair(tokenizer.encode, tokenizer.get_bos_token_id(), ctx, cont, max_length)
             for ctx, cont in zip(request["contexts"], request["continuations"], strict=True)]
    base = {k: v for k, v in request.items() if k not in ("contexts", "continuations")}
    base["token_lengths"] = [{k: v for k, v in pair.items() if k != "tokens"} for pair in pairs]
    if any(not pair["supported"] for pair in pairs):
        return {**base, "supported": False, "scores": None, "pred_acc": None, "pred_acc_norm": None,
                "acc": 0, "acc_norm": 0, "primary_correct": 0}
    scores = scorer(pairs)
    if len(scores) != len(pairs) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Incomplete/nonfinite model scores")
    normalized = [score / length for score, length in zip(scores, request["normalization_lengths"], strict=True)]
    raw_pred = max(range(len(scores)), key=lambda i: scores[i])
    norm_pred = max(range(len(scores)), key=lambda i: normalized[i])
    raw_correct = int(raw_pred == request["gold"]); norm_correct = int(norm_pred == request["gold"])
    return {**base, "supported": True, "scores": scores, "normalized_scores": normalized,
            "pred_acc": raw_pred, "pred_acc_norm": norm_pred, "acc": raw_correct, "acc_norm": norm_correct,
            "primary_correct": norm_correct if request["task"] in NORMALIZED else raw_correct}


def summarize(items, requests):
    tasks = {}
    for task in TASKS:
        expected = [r for r in requests if r["task"] == task]
        done = [r for r in items if r["task"] == task]
        labels = Counter(r["gold"] for r in expected)
        majority = max(labels.values()) / len(expected)
        complete = len(done) == len(expected)
        acc = sum(r["acc"] for r in done) / len(expected) if complete else None
        norm = sum(r["acc_norm"] for r in done) / len(expected) if complete else None
        primary = norm if task in NORMALIZED else acc
        tasks[task] = {"complete": complete, "expected": len(expected), "scored": len(done),
                       "unsupported": sum(not r["supported"] for r in done), "acc": acc,
                       "acc_norm": norm if task in NORMALIZED else None, "primary": primary,
                       "majority_baseline": majority,
                       "uniform_choice_baseline": statistics.mean(1 / len(r["contexts"]) for r in expected),
                       "floor": FLOORS[task], "floor_pass": primary >= FLOORS[task] and (task != "boolq" or primary > majority) if complete else None}
    complete = all(t["complete"] for t in tasks.values())
    macro = statistics.mean(t["primary"] for t in tasks.values()) if complete else None
    return {"complete": complete, "tasks": tasks, "Q": macro, "macro_floor": .52,
            "macro_floor_pass": macro >= .52 if complete else None,
            "quality_floors_pass": all(t["floor_pass"] for t in tasks.values()) and macro >= .52 if complete else None,
            "incomplete_scores_must_not_be_reported_as_full_suite": not complete}


def cached_items(path, requests, contract):
    """Only an exact, intact prefix for this model/policy may avoid inference."""
    items = []
    chain = digest(canonical(contract))
    if not path.exists():
        return items, chain
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            claimed = row.pop("cache_chain_sha256", None)
            chain = digest(chain + canonical(row))
            if index >= len(requests) or claimed != chain or row.get("request_sha256") != requests[index]["request_sha256"]:
                raise ValueError("Evaluation cache identity, order or integrity mismatch")
            items.append(row)
    return items, chain


def read_owned_checkpoint(path, data_digest, torch, *, sft_data=None, main_parent=None, main_stage=None, pilot_data=None, tiny=False):
    side = json.loads(path.with_suffix(".sha256.json").read_text())
    if side.get("format") == SFT_CHECKPOINT_FORMAT:
        sft_adapter = sft_adapter_module()
        payload, verified = sft_adapter.read(path, data_digest, torch, sft_data=sft_data,
            main_parent=main_parent, main_stage=main_stage, pilot_data=pilot_data, tiny=tiny)
        payload["_sft_verification"] = verified
        return payload
    if side.get("format") == main_training.FORMAT:
        payload, side = main_training.inspect_owned(path, main=True, torch=torch)
        if payload["state"]["pilot_data_manifest_sha256"] != data_digest or side.get("pilot_data_manifest_sha256") != data_digest:
            raise ValueError("Owned main checkpoint data identity mismatch")
        if payload["state"]["pilot_checkpoint_sha256"] != side.get("pilot_checkpoint_sha256"):
            raise ValueError("Owned main checkpoint parent identity mismatch")
        return payload
    if side.get("format") != training.CHECKPOINT_FORMAT or sha256(path) != side.get("sha256"):
        raise ValueError("Owned checkpoint integrity failure")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("format") != training.CHECKPOINT_FORMAT or payload.get("source_revision") != REVISION or payload.get("source_hashes") != training.source_hashes():
        raise ValueError("Owned checkpoint source mismatch")
    if payload["state"]["data_manifest_sha256"] != data_digest or side.get("data_manifest_sha256") != data_digest:
        raise ValueError("Owned checkpoint data identity mismatch")
    if payload["state"]["initial_weights_sha256"] != side.get("initial_weights_sha256"):
        raise ValueError("Owned checkpoint random initialization identity mismatch")
    return payload


def checkpoint_lineage(payload):
    state = payload["state"]
    if payload["format"] == SFT_CHECKPOINT_FORMAT:
        return sft_adapter_module().lineage(payload)
    if payload["format"] == main_training.FORMAT:
        return {"phase": "main", "step": state["step"],
            "token_offset": state["pilot_target_positions"] + state["main_target_positions"],
            "main_target_positions": state["main_target_positions"],
            "pilot_target_positions": state["pilot_target_positions"],
            "pilot_checkpoint_sha256": state["pilot_checkpoint_sha256"],
            "initial_weights_sha256": state["initial_weights_sha256"]}
    return {"phase": "pilot", **{k: state[k] for k in ("step", "token_offset", "initial_weights_sha256")}}


def verify_main_stage(payload, directory, data_directory, data, data_digest, *, tiny=False):
    if payload["format"] == SFT_CHECKPOINT_FORMAT:
        if directory is None or "_sft_verification" not in payload:
            raise ValueError("SFT checkpoint has no verified current main stage")
        current, digest = main_training.read_manifest(directory)
        if current.get("status") != "initialized" or digest != payload["_sft_verification"]["main_stage_manifest_sha256"]:
            raise ValueError("SFT parent main stage changed or was blocked during verification")
        return digest
    if payload["format"] != main_training.FORMAT:
        return None
    if directory is None:
        raise ValueError("Main evaluation requires its current stage to detect later exclusion conflicts")
    state = payload["state"]
    stream = main_training.FiniteShards(directory, data_directory, data, data_digest, tiny=tiny,
        contract=state["stage_contract"], accepted_shards=state["accepted_shards"], cursor=state["cursor"])
    return stream.manifest_hash


def model_from_payload(payload, GPT, GPTConfig, torch, device):
    """Preserve exact checkpoint storage dtypes, including SFT FP32 embeddings."""
    config = payload["model_config"]
    with torch.device("meta"): model = GPT(GPTConfig(**config))
    model.load_state_dict(payload["model"], strict=True, assign=True)
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len,
        config["n_embd"] // config["n_head"], device="cpu")
    model.to(device); model.eval()
    return model


def export_weights(model, data_dir, destination, checkpoint_hash, torch, *, lineage=None, model_stage="base_pretraining"):
    """Exact inference weights, never dtype-converted or advertised as HF format."""
    notices = None
    report_path = data_dir / "source-report.json"
    if report_path.exists():
        source_report = json.loads(report_path.read_text())
        if source_report.get("source_lock_sha256"):
            notices = ROOT / "research/data_sources.json"
            if sha256(notices) != source_report["source_lock_sha256"]:
                raise ValueError("Export source notices no longer match the training provenance")
    destination.mkdir(parents=True, exist_ok=False)
    torch.save({"format": "udgam-dense-native-inference-v1", "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": asdict(model.config), "source_revision": REVISION,
                "source_checkpoint_sha256": checkpoint_hash}, destination / "model.pt")
    shutil.copyfile(data_dir / "tokenizer.json", destination / "tokenizer.json")
    write_json(destination / "model-config.json", asdict(model.config))
    shutil.copytree(VENDOR / "nanochat", destination / "vendor" / ("nanochat-" + REVISION) / "nanochat",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copyfile(VENDOR / name, destination / "vendor" / ("nanochat-" + REVISION) / name)
    shutil.copyfile(Path(__file__).with_name("infer_dense.py"), destination / "infer.py")
    assets = ROOT / "configs/inference"
    for name in ("README.md", "requirements.txt", "LICENSE.inference"):
        shutil.copyfile(assets / name, destination / name)
    for name in ("LICENSE", "LICENSE_SCOPE.md"):
        shutil.copyfile(ROOT / name, destination / name)
    if notices is not None:
        (destination / "attribution").mkdir()
        shutil.copyfile(notices, destination / "attribution/data-sources.json")
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    files = {str(p.relative_to(destination)): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in destination.rglob("*") if p.is_file()}
    result = {"format": "udgam-dense-native-inference-v1", "source_checkpoint_sha256": checkpoint_hash,
              "total_parameters": sum(p.numel() for p in model.parameters()), "parameter_tensor_bytes": weights,
              "dtype_policy": "original mixed dtypes preserved exactly", "files": files,
              "inference_artifact_bytes_before_this_manifest": sum(f["bytes"] for f in files.values()),
              "optimizer_included": False, "hf_compatible": False,
              "model_stage": model_stage, "lineage": lineage or {},
              "license": "MIT", "license_scope": "LICENSE_SCOPE.md; upstream and data terms retained",
              "training_source_notices": "attribution/data-sources.json" if notices is not None else None,
              "loader": "python infer.py --prompt TEXT; strict owned weights-only mmap load, assign tensors, reconstruct rotary buffers; JSON tokenizer; offline SDPA cached generation",
              "chat_template": "nanochat native text-only markers; system merged into first user; requires corresponding SFT for learned chat behavior",
              "runtime_dependencies": "Pinned requirements.txt; Python3.11/3.12; no optimizer state, rustbpe trainer, model download or tool engine"}
    write_json(destination / "export-manifest.json", result)
    result["inference_artifact_bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    return result


def fixed_latency_prompts():
    # Public development strings, independent of every benchmark document/label.
    seed = "Explain how careful observation, clear definitions, and checking an example can improve everyday reasoning. "
    return [(f"Development example {i:02d}. " + seed * 30)[:size] for size in (128, 512, 2048) for i in range(10)]


def forced_latency(model, tokenizer, torch, KVCache, dtype, device, deadline):
    """Separate forced128-token microbenchmark; never user-response latency."""
    prompts = fixed_latency_prompts(); results = []
    def sync():
        if device == "cuda": torch.cuda.synchronize()
    for index, text in enumerate([prompts[0]] * 5 + prompts):
        if time.time() >= deadline:
            break
        prompt = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
        if len(prompt) + 128 > model.config.sequence_len:
            if index >= 5: results.append({"prompt_index": index - 5, "supported": False})
            continue
        cache = KVCache(batch_size=1, num_heads=model.config.n_kv_head, seq_len=len(prompt) + 128,
                        head_dim=model.config.n_embd // model.config.n_head, num_layers=model.config.n_layer,
                        device=device, dtype=dtype)
        generated = []
        with torch.inference_mode(), training.autocast(torch, device):
            sync(); start = time.perf_counter()
            logits = model(torch.tensor([prompt], device=device), kv_cache=cache)
            token = logits[:, -1:, :].argmax(-1); generated.append(int(token.item()))
            sync(); first = time.perf_counter()
            for _ in range(127):
                logits = model(token, kv_cache=cache); token = logits[:, -1:, :].argmax(-1)
                generated.append(int(token.item()))
            sync(); finish = time.perf_counter()
        if index >= 5:
            output_bytes = len(tokenizer.decode(generated).encode())
            results.append({"prompt_index": index - 5, "supported": True, "prompt_utf8_bytes": len(text.encode()),
                            "prompt_tokens": len(prompt), "forced_output_tokens": 128,
                            "output_utf8_bytes": output_bytes,
                            "ttft_seconds": first - start, "decode_seconds": finish - first,
                            "decode_tokens_per_second": 127 / (finish - first),
                            "total_output_bytes_per_second": output_bytes / (finish - start), "elapsed_seconds": finish - start})
    valid = [r for r in results if r["supported"]]
    p95 = lambda xs: sorted(xs)[max(0, math.ceil(.95 * len(xs)) - 1)] if xs else None
    return {"kind": "forced_128_token_microbenchmark_not_response_latency", "warmups_requested": 5,
            "complete": len(results) == 30, "prompt_strings_sha256": digest(canonical(prompts)), "results": results,
            "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in valid) if valid else None,
            "p95_ttft_seconds": p95([r["ttft_seconds"] for r in valid]),
            "median_decode_tokens_per_second": statistics.median(r["decode_tokens_per_second"] for r in valid) if valid else None,
            "stops_on_eos": False, "quality_evidence": False}


def main(argv=None):
    process_started = time.time()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--main-stage", type=Path, help="Required for main checkpoints; verify current exclusion status and all data lineage")
    p.add_argument("--sft-data", type=Path, help="Required for completed owned SFT checkpoints; verify finite pass and tokenizer lineage")
    p.add_argument("--main-parent", type=Path, help="Required owned main parent checkpoint for SFT lineage verification")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--benchmarks", type=Path, default=ROOT / "data/benchmark-exclusions-002")
    p.add_argument("--development-smoke", type=Path, help="Explicit100-item development set; never confirmatory scoring")
    p.add_argument("--task-definitions", type=Path, default=ROOT / "research/harness-task-definitions")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-requests-only", action="store_true")
    p.add_argument("--resume-evaluation", action="store_true", help="Explicitly reuse an intact same-model response-cache prefix")
    p.add_argument("--guard-evidence", type=Path)
    p.add_argument("--max-eval-seconds", type=float, default=180)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--export-weights", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--tiny-cpu-test", action="store_true")
    a = p.parse_args(argv)
    training.finite_number(a.max_eval_seconds)
    if a.max_eval_seconds > 600 or not 1 <= a.batch_size <= 8 or (a.output.exists() and not a.resume_evaluation):
        raise ValueError("Invalid evaluation bound or existing output")
    if a.resume_evaluation and (not a.output.exists() or a.prepare_requests_only):
        raise ValueError("Evaluation resume needs an existing scored run")
    if not a.prepare_requests_only and not a.checkpoint:
        raise ValueError("Owned checkpoint required")
    sft_selected = a.checkpoint is not None and json.loads(a.checkpoint.with_suffix(".sha256.json").read_text()).get("format") == SFT_CHECKPOINT_FORMAT
    sft_dependencies = {}
    if sft_selected:
        adapter = sft_adapter_module()
        sft_dependencies = {"sft_checkpoint_adapter.py": sha256(Path(adapter.__file__)), **adapter.sft.code_hashes()}
    elif a.sft_data is not None or a.main_parent is not None:
        raise ValueError("SFT data/main-parent options require an owned SFT checkpoint")
    data, data_digest = verify_prepared(a.data, production=not a.tiny_cpu_test)
    benchmark, suite = load_development_smoke(a.development_smoke) if a.development_smoke else load_frozen_suite(a.benchmarks)
    source = verify_task_definitions(a.task_definitions)
    requests = make_requests(suite)
    benchmark_digest = sha256((a.development_smoke or a.benchmarks) / "manifest.json")
    training_source = json.loads((a.data / "source-report.json").read_text())
    exclusion_digest = benchmark["excluded_frozen_manifest_sha256"] if a.development_smoke else benchmark_digest
    if not a.tiny_cpu_test and training_source.get("benchmark_manifest_sha256") != exclusion_digest:
        raise ValueError("Evaluation holdout does not match the training exclusion manifest")
    request_text = "".join(canonical(request) + "\n" for request in requests)
    if not a.resume_evaluation:
        a.output.mkdir(parents=True)
        (a.output / "requests.jsonl").write_text(request_text, encoding="utf-8")
    elif sha256(a.output / "requests.jsonl") != hashlib.sha256(request_text.encode()).hexdigest():
        raise ValueError("Cached request strings changed")
    contract = {"policy": POLICY, "benchmark_manifest_sha256": benchmark_digest,
                "task_source_manifest_sha256": sha256(a.task_definitions / "manifest.json"),
                "harness_revision": source["revision"], "requests_sha256": sha256(a.output / "requests.jsonl"),
                "training_data_manifest_sha256": data_digest, "script_sha256": sha256(Path(__file__)),
                "checkpoint_sha256": sha256(a.checkpoint) if a.checkpoint else None,
                "main_stage_manifest_sha256": main_training.read_manifest(a.main_stage)[1] if a.main_stage else None,
                "sft_manifest_sha256": sha256(a.sft_data / "manifest.json") if a.sft_data else None,
                "sft_main_parent_sha256": sha256(a.main_parent) if a.main_parent else None,
                "scoring_dtype": "float32" if a.tiny_cpu_test else "bfloat16",
                "adapter_dependency_sha256": {"train_dense.py": sha256(Path(training.__file__)),
                     "train_main.py": sha256(Path(main_training.__file__)),
                     **sft_dependencies,
                     "prepare_dense_tokens.py": sha256(Path(sys.modules["prepare_dense_tokens"].__file__))},
                "native_source_sha256": training.source_hashes(),
                "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "tiktoken", "rustbpe", "numpy")},
                "python_version": sys.version,
                "fa3_revision": None if a.tiny_cpu_test else training.FA3_REVISION,
                "declared_items": len(requests), "no_test_feedback_for_checkpoint_selection": True,
                "suite_kind": "development_smoke_not_confirmatory" if a.development_smoke else "frozen_final_suite",
                "scope": "implementation_prepared; no general superiority claim"}
    if a.resume_evaluation:
        if json.loads((a.output / "evaluation-contract.json").read_text()) != contract:
            raise ValueError("Response cache belongs to a different model, source or scoring policy")
    else:
        write_json(a.output / "evaluation-contract.json", contract)
    if a.prepare_requests_only:
        return contract
    guard_stop = float("inf")
    if not a.tiny_cpu_test:
        if not a.guard_evidence:
            raise ValueError("Fresh provider guard evidence required")
        evidence_profile = json.loads(a.guard_evidence.read_text()).get("profile")
        validator = main_training.validate_main_guard if evidence_profile == "main-v1" else training.validate_guard
        _, guard_stop = validator(a.guard_evidence, export_reserve_seconds=120)
    started = time.time(); deadline = min(started + a.max_eval_seconds, guard_stop)
    torch, GPT, GPTConfig, KVCache, dtype, backend = training.runtime(tiny=a.tiny_cpu_test,
                                     attention="sdpa" if a.tiny_cpu_test else "fa3", optimizer_compile=False)
    payload = read_owned_checkpoint(a.checkpoint, data_digest, torch, sft_data=a.sft_data,
        main_parent=a.main_parent, main_stage=a.main_stage, pilot_data=a.data, tiny=a.tiny_cpu_test)
    stage_hash = verify_main_stage(payload, a.main_stage, a.data, data, data_digest, tiny=a.tiny_cpu_test)
    if stage_hash != contract["main_stage_manifest_sha256"]:
        raise ValueError("Main evaluation stage changed during verification or was supplied for a pilot")
    config = payload["model_config"]
    training.validate_config({**config, "source_revision": REVISION, "initialization": "random_weights_only"}, tiny=a.tiny_cpu_test)
    device = "cpu" if a.tiny_cpu_test else "cuda"
    model = model_from_payload(payload, GPT, GPTConfig, torch, device)
    lineage = checkpoint_lineage(payload)
    sft_verification = payload.get("_sft_verification")
    if sft_verification is not None and (sft_verification["sft_manifest_sha256"] != contract["sft_manifest_sha256"] or
            sft_verification["main_parent_sha256"] != contract["sft_main_parent_sha256"]):
        raise ValueError("SFT data/parent changed during evaluation contract verification")
    del payload
    tokenizer = load_portable_tokenizer(a.data)
    if device == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    cold_load = time.time() - started
    loaded_memory = torch.cuda.memory_allocated() if device == "cuda" else None
    scorer = lambda pairs: continuation_scores(model, pairs, device=device, torch=torch, batch_size=a.batch_size)
    items, chain = cached_items(a.output / "items.jsonl", requests, contract)
    prior_item_count = len(items)
    previous_summary = json.loads((a.output / "summary.json").read_text()) if a.resume_evaluation and (a.output / "summary.json").exists() else None
    cost_history_complete = not a.resume_evaluation or (previous_summary is not None and previous_summary.get("items_sha256") == sha256(a.output / "items.jsonl") and previous_summary.get("cost_history_complete", False))
    with (a.output / "items.jsonl").open("a", encoding="utf-8") as log:
        for request in requests[prior_item_count:]:
            if time.time() >= deadline: break
            item = score_request(request, tokenizer, scorer, max_length=min(2048, config["sequence_len"]))
            chain = digest(chain + canonical(item))
            log.write(canonical({**item, "cache_chain_sha256": chain}) + "\n"); log.flush(); items.append(item)
    summary = summarize(items, requests)
    if a.development_smoke:
        summary["development_macro_accuracy"] = summary.pop("Q")
        summary["quality_floors_pass"] = summary["macro_floor_pass"] = None
        summary["quality_gate_scope"] = "No confirmatory quality gate on100 development examples"
    summary.update(contract=contract, checkpoint_sha256=sha256(a.checkpoint), checkpoint_bytes=a.checkpoint.stat().st_size,
                   lineage=lineage, total_parameters=sum(p.numel() for p in model.parameters()),
                   attention_backend=backend, dtype=str(dtype), torch_version=str(torch.__version__),
                   cold_load_seconds=cold_load, evaluation_seconds=time.time() - started,
                   cold_load_scope="Training checkpoint load includes optimizer tensors on CPU; not a standalone inference-artifact cold-load comparison",
                   loaded_cuda_memory_bytes=loaded_memory,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
                   process_peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                   cash_cap_enforced_by_this_script=False)
    summary["reused_cached_items"] = prior_item_count
    summary["items_sha256"] = sha256(a.output / "items.jsonl")
    write_json(a.output / "summary.json", summary)
    if a.export_weights and (a.output / "inference").exists():
        manifest = json.loads((a.output / "inference/export-manifest.json").read_text())
        if manifest["source_checkpoint_sha256"] != summary["checkpoint_sha256"] or manifest.get("lineage") != lineage or any(
            sha256(a.output / "inference" / name) != info["sha256"] for name, info in manifest["files"].items()
        ):
            raise ValueError("Cached native export identity/integrity mismatch")
        summary["export"] = manifest
        summary["export"]["inference_artifact_bytes"] = sum(p.stat().st_size for p in (a.output / "inference").rglob("*") if p.is_file())
    elif a.export_weights and time.time() < guard_stop:
        if sft_verification is not None:
            summary["export"] = sft_adapter_module().export(model, a.data, a.sft_data, a.output / "inference",
                summary["checkpoint_sha256"], torch, verified=sft_verification, lineage_record=lineage)
        else:
            summary["export"] = export_weights(model, a.data, a.output / "inference", summary["checkpoint_sha256"], torch, lineage=lineage)
    if a.latency and time.time() < deadline:
        summary["latency"] = forced_latency(model, tokenizer, torch, KVCache, dtype, device, deadline)
    rate = json.loads(a.guard_evidence.read_text())["hourly_total_usd"] if a.guard_evidence and not a.tiny_cpu_test else None
    session = {"elapsed_seconds_including_export_and_latency": time.time() - process_started,
               "hourly_total_usd": rate, "new_items": len(items) - prior_item_count}
    session["estimated_compute_cost_usd"] = session["elapsed_seconds_including_export_and_latency"] * rate / 3600 if rate is not None else None
    summary["sessions"] = (previous_summary.get("sessions", []) if previous_summary else []) + [session]
    summary["cost_history_complete"] = cost_history_complete
    summary["total_recorded_process_seconds"] = sum(s["elapsed_seconds_including_export_and_latency"] for s in summary["sessions"])
    summary["estimated_recorded_compute_cost_usd"] = sum(s["estimated_compute_cost_usd"] for s in summary["sessions"] if s["estimated_compute_cost_usd"] is not None) if rate is not None else None
    summary["cost_scope"] = "Recorded process sessions including verification, checkpoint loading, scoring, export and optional latency; missing interrupted sessions are flagged; provider setup/storage/rounding require external billing ledger"
    summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if device == "cuda" else None
    summary["process_peak_ram_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    write_json(a.output / "summary.json", summary)
    print(json.dumps({"complete": summary["complete"], "scored": len(items), "Q": summary.get("Q"), "development_macro_accuracy": summary.get("development_macro_accuracy")}))
    return summary


if __name__ == "__main__":
    main()
