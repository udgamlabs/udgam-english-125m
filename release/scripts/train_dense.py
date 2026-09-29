#!/usr/bin/env python3
"""Finite, single-GPU scratch pilot using pinned nanochat GPT and MuonAdamW.

This script cannot create/stop a cloud resource. Production requires fresh
external guard evidence and still depends on that independent guard remaining
alive. Its timer is a training limit, never a provider-enforced cash cap.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from prepare_dense_tokens import ROOT, REVISION, VENDOR, load_portable_tokenizer, sha256, verify_prepared, write_json

FA3_REPO = "varunneal/flash-attention-3"
FA3_REVISION = "de87b9b5af06dd9984df595bef90b2eba44b181a"
MODEL_KEYS = ("sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd", "window_pattern")
PRIMARY = dict(sequence_len=2048, vocab_size=16384, n_layer=12, n_head=6, n_kv_head=2, n_embd=768, window_pattern="SSSL")
CHECKPOINT_FORMAT = "udgam-owned-dense-scratch-v1"


def finite_number(value, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("Invalid finite numeric bound")
    return float(value)


def utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("UTC date required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("Timezone-aware UTC date required")
    return result.timestamp()


def validate_guard(path, *, now=None, export_reserve_seconds=300):
    """Validate launch evidence, not the existence/future operation of a guard."""
    now = time.time() if now is None else now
    evidence = json.loads(Path(path).read_text())
    pod = evidence.get("pod_id")
    if not isinstance(pod, str) or not pod or len(pod) > 100:
        raise ValueError("Missing bound provider pod")
    observed = utc_timestamp(evidence["observed_at_utc"])
    if not 0 <= now - observed <= 120:
        raise ValueError("Billing observation stale or in future")
    rate = finite_number(evidence["hourly_total_usd"])
    ceiling = finite_number(evidence["cash_ceiling_usd"])
    if ceiling > 4.50:
        raise ValueError("First-pilot cash ceiling exceeds authorization")
    deadline = utc_timestamp(evidence["provider_stop_deadline_utc"])
    guard = evidence["provider_stop_guard"]
    if guard.get("verified") is not True or guard.get("pod_id") != pod:
        raise ValueError("Provider guard not verified for this pod")
    if utc_timestamp(guard["deadline_utc"]) != deadline:
        raise ValueError("Provider stop deadlines disagree")
    if not 0 <= now - utc_timestamp(guard["heartbeat_at_utc"]) <= 60:
        raise ValueError("Provider guard heartbeat stale or in future")
    if not export_reserve_seconds + 30 < deadline - now <= 2400:
        raise ValueError("Insufficient safe runtime or deadline outside pilot bound")
    if (deadline - observed) * rate / 3600 > ceiling:
        raise ValueError("Remaining estimated runtime exceeds declared cash ceiling")
    # Already-incurred setup cost and storage are the controller's responsibility.
    return evidence, deadline - export_reserve_seconds


def validate_config(config, *, tiny=False):
    c = {k: config[k] for k in MODEL_KEYS}
    for key in MODEL_KEYS[:-1]:
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError("Invalid model dimension")
    if c["n_embd"] % c["n_head"] or c["n_head"] % c["n_kv_head"] or (c["n_embd"] // c["n_head"]) % 2:
        raise ValueError("Invalid grouped query attention dimensions")
    if c["sequence_len"] < 2 or not c["window_pattern"] or set(c["window_pattern"]) - set("SL"):
        raise ValueError("Invalid context/window pattern")
    if tiny:
        if c["n_layer"] > 4 or c["n_embd"] > 64 or c["vocab_size"] > 1024 or c["sequence_len"] > 64:
            raise ValueError("Local test flag permits only a tiny CPU model")
    elif c != PRIMARY or config.get("source_revision") != REVISION or config.get("initialization") != "random_weights_only":
        raise ValueError("Production configuration is not the reviewed primary candidate")
    return c


def runtime(*, tiny=False, attention="fa3", optimizer_compile=True):
    os.environ["NANOCHAT_DTYPE"] = "float32" if tiny else "bfloat16"
    os.environ.setdefault("NANOCHAT_BASE_DIR", str(ROOT / ".cache/nanochat"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import torch
    if tiny:
        torch.set_num_threads(1)
    elif not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Production requires BF16 CUDA")
    if not tiny and attention == "fa3":
        import importlib.metadata
        import kernels
        if importlib.metadata.version("kernels") != "0.12.2":
            raise ValueError("Reviewed FA3 loader requires kernels==0.12.2")
        original = kernels.get_kernel

        def pinned_kernel(repo_id, *args, **kwargs):
            if repo_id != FA3_REPO or ("revision" in kwargs and kwargs["revision"] != FA3_REVISION):
                raise ValueError("Unreviewed attention kernel request")
            kwargs["revision"] = FA3_REVISION
            return original(repo_id, *args, **kwargs)

        kernels.get_kernel = pinned_kernel
    elif not tiny:
        # Avoid an implicit kernel download while selecting the explicit fallback.
        os.environ["HF_HUB_OFFLINE"] = "1"
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.engine import KVCache
    from nanochat import flash_attention, optim
    from nanochat.common import COMPUTE_DTYPE
    expected_dtype = torch.float32 if tiny else torch.bfloat16
    if COMPUTE_DTYPE != expected_dtype:
        raise ValueError("nanochat was already imported with an incompatible compute dtype")
    if tiny or attention == "sdpa":
        flash_attention.USE_FA3 = False
    elif not flash_attention.USE_FA3:
        raise ValueError("Pinned FA3 did not load; refuse the slow SSSL production fallback")
    if not optimizer_compile:
        for name in ("adamw_step_fused", "muon_step_fused"):
            fn = getattr(optim, name)
            setattr(optim, name, getattr(fn, "_torchdynamo_orig_callable", fn))
    return torch, GPT, GPTConfig, KVCache, COMPUTE_DTYPE, "fa3" if flash_attention.USE_FA3 else "sdpa"


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    return {name: sha256(VENDOR / name) for name in (
        "nanochat/gpt.py", "nanochat/optim.py", "nanochat/engine.py", "nanochat/flash_attention.py", "nanochat/tokenizer.py")}


def setup_optimizer(model, train_config, GPT, GPTConfig, torch):
    tokens = train_config["batch_rows"] * train_config["accumulation_steps"] * model.config.sequence_len
    scale = math.sqrt(tokens / 2**19)
    # Upstream scales decay with compute-optimal horizons even when iterations
    # are manually overridden. Keep that rule rather than amplifying decay for
    # this very short pilot. D_REF is depth12, width768, MHA (the upstream default).
    reference = asdict(model.config)
    reference.update(n_layer=12, n_embd=768, n_head=6, n_kv_head=6)
    with torch.device("meta"):
        ref_model = GPT(GPTConfig(**reference))
    count = lambda m: m.num_scaling_params()["transformer_matrices"] + m.num_scaling_params()["lm_head"]
    decay = 0.28 * scale * count(ref_model) / count(model)
    del ref_model
    optimizer = model.setup_optimizer(unembedding_lr=.008 * scale, embedding_lr=.3 * scale,
                                      matrix_lr=.02 * scale, scalar_lr=.5 * scale, weight_decay=decay)
    return optimizer, {"batch_tokens": tokens, "batch_lr_scale": scale, "initial_muon_weight_decay": decay,
                       "rule": "upstream_base_train_with_MHA_d12_reference_and_GQA_candidate"}


def schedule(optimizer, step, train_config, scaled_decay):
    total = train_config["max_steps"]
    warmup = train_config["warmup_steps"]
    down = max(1, round(train_config["warmdown_ratio"] * total))
    if warmup and step < warmup:
        multiplier = (step + 1) / warmup
    elif step <= total - down:
        multiplier = 1.0
    else:
        progress = (total - step) / down
        multiplier = progress + (1 - progress) * train_config["final_lr_frac"]
    if step < 400:
        momentum = .85 + (.97 - .85) * step / 400
    elif step >= total - down:
        progress = (step - (total - down)) / down
        momentum = .97 * (1 - progress) + .90 * progress
    else:
        momentum = .97
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
        if group["kind"] == "muon":
            group["momentum"] = momentum
            group["weight_decay"] = scaled_decay * .5 * (1 + math.cos(math.pi * step / total))
    return multiplier


def batch(array, start, count, device, torch):
    import numpy as np
    if start < 0 or count <= 0 or start + count > len(array):
        raise ValueError("Finite token supply exhausted; wrapping is forbidden")
    data = torch.from_numpy(np.array(array[start:start + count], dtype=np.int64)).to(device)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def autocast(torch, device):
    return nullcontext() if device == "cpu" else torch.autocast("cuda", dtype=torch.bfloat16)


def train_step(model, execution_model, optimizer, array, row_offset, step, train_config, optimizer_info, device, torch):
    model.train()
    lr_multiplier = schedule(optimizer, step, train_config, optimizer_info["initial_muon_weight_decay"])
    losses = []
    for _ in range(train_config["accumulation_steps"]):
        x, y = batch(array, row_offset, train_config["batch_rows"], device, torch)
        with autocast(torch, device):
            loss = execution_model(x, targets=y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        (loss / train_config["accumulation_steps"]).backward()
        losses.append(float(loss.detach()))
        row_offset += train_config["batch_rows"]
    optimizer.step(); model.zero_grad(set_to_none=True)
    return row_offset, sum(losses) / len(losses), lr_multiplier


def evaluate(model, array, token_bytes, *, rows, batch_rows, device, torch):
    if rows <= 0 or rows > len(array):
        raise ValueError("Not enough independent validation rows")
    model.eval()
    nats = 0.; byte_nats = 0.; targets = 0; nbytes = 0
    with torch.inference_mode():
        for start in range(0, rows, batch_rows):
            x, y = batch(array, start, min(batch_rows, rows - start), device, torch)
            with autocast(torch, device):
                losses = model(x, targets=y, loss_reduction="none").view_as(y)
            sizes = token_bytes[y]
            nats += float(losses.double().sum()); targets += y.numel()
            byte_nats += float((losses.double() * (sizes > 0)).sum()); nbytes += int(sizes.sum())
    if not math.isfinite(nats) or nbytes <= 0:
        raise FloatingPointError("Invalid held-out evaluation")
    return {"loss_nats_per_target": nats / targets, "bpb": byte_nats / math.log(2) / nbytes,
            "rows": rows, "target_positions": targets, "raw_target_bytes": nbytes,
            "available_rows": len(array), "wrapping": False,
            "bpb_policy": "special_token_targets_excluded; sum_nats / ln2 / raw_token_bytes"}


def greedy(model, prompt, new_tokens, device, torch, KVCache, dtype):
    if not prompt or new_tokens < 1 or len(prompt) + new_tokens > model.config.sequence_len:
        raise ValueError("Generation exceeds finite cache context")
    cache = KVCache(batch_size=1, num_heads=model.config.n_kv_head, seq_len=len(prompt) + new_tokens,
                    head_dim=model.config.n_embd // model.config.n_head, num_layers=model.config.n_layer,
                    device=device, dtype=dtype)
    generated = []
    model.eval()
    with torch.inference_mode(), autocast(torch, device):
        logits = model(torch.tensor([prompt], device=device), kv_cache=cache)
        for index in range(new_tokens):
            token = logits[:, -1:, :].argmax(-1)
            generated.append(int(token.item()))
            if index + 1 < new_tokens:
                logits = model(token, kv_cache=cache)
    return generated


def rng_state(torch):
    import numpy as np
    ns = np.random.get_state()
    return {"python": random.getstate(), "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, torch):
    import numpy as np
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, optimizer, state, torch):
    path = Path(path)
    payload = {"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "model_config": asdict(model.config), "state": state, "rng": rng_state(torch),
               "source_revision": REVISION, "source_hashes": source_hashes()}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    write_json(path.with_suffix(".sha256.json"), {"format": CHECKPOINT_FORMAT, "sha256": sha256(path),
               "data_manifest_sha256": state["data_manifest_sha256"], "initial_weights_sha256": state["initial_weights_sha256"]})


def load_checkpoint(path, model, optimizer, *, data_digest, train_config, torch, restore=True):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".sha256.json").read_text())
    if sidecar.get("format") != CHECKPOINT_FORMAT or sidecar.get("sha256") != sha256(path):
        raise ValueError("Only an intact owned scratch checkpoint may be resumed")
    # Safe torch weights-only loading: RNGs are stored as basic values/tensors.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT or ckpt.get("source_revision") != REVISION or ckpt.get("source_hashes") != source_hashes():
        raise ValueError("Checkpoint source identity mismatch")
    state = ckpt["state"]
    if ckpt["model_config"] != asdict(model.config) or state["data_manifest_sha256"] != data_digest or state["train_config"] != train_config:
        raise ValueError("Resume model/data/training configuration mismatch")
    if sidecar.get("data_manifest_sha256") != data_digest or sidecar.get("initial_weights_sha256") != state["initial_weights_sha256"]:
        raise ValueError("Checkpoint provenance sidecar mismatch")
    expected_offset = state["step"] * train_config["batch_rows"] * train_config["accumulation_steps"]
    if state["row_offset"] != expected_offset or state["token_offset"] != expected_offset * model.config.sequence_len:
        raise ValueError("Checkpoint finite token cursor mismatch")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if restore:
        restore_rng(ckpt["rng"], torch)
    return state


def run(args):
    config = json.loads(args.config.read_text())
    c = validate_config(config, tiny=args.tiny_cpu_test)
    if args.max_steps < 1 or args.batch_rows < 1 or args.accumulation_steps < 1 or args.eval_rows < 1:
        raise ValueError("Step/batch/evaluation bounds must be positive")
    finite_number(args.compute_seconds)
    if args.compute_seconds > 525 or (args.tiny_cpu_test and (args.max_steps > 8 or args.batch_rows > 4 or args.accumulation_steps > 2 or args.compute_seconds > 60)):
        raise ValueError("Pilot/test bound exceeded")
    if not args.tiny_cpu_test and args.attention != "fa3":
        raise ValueError("The SSSL primary production pilot requires pinned FA3")
    if args.output.exists():
        raise ValueError("Output exists; select a fresh run directory, including for explicit resume")
    data, digest = verify_prepared(args.data, production=not args.tiny_cpu_test)
    if args.tiny_cpu_test and data["profile"] != "tiny_cpu_test":
        raise ValueError("CPU smoke requires explicitly marked tiny test data")
    if (data["sequence_len"], data["vocab_size"]) != (c["sequence_len"], c["vocab_size"]):
        raise ValueError("Model/tokenizer/packing dimensions mismatch")
    rows_per_step = args.batch_rows * args.accumulation_steps
    if data["splits"]["train"]["rows"] < args.max_steps * rows_per_step:
        raise ValueError("Insufficient finite training rows for requested horizon; lower max-steps explicitly")
    if data["splits"]["val"]["rows"] < args.eval_rows:
        raise ValueError("Insufficient distinct held-out token positions")
    train_config = dict(max_steps=args.max_steps, batch_rows=args.batch_rows, accumulation_steps=args.accumulation_steps,
                        warmup_steps=min(40, max(1, args.max_steps // 10)), warmdown_ratio=.65, final_lr_frac=.05,
                        compute_seconds=args.compute_seconds, model_compile=args.compile_model,
                        optimizer_compile=not args.tiny_cpu_test, seed=args.seed)
    guard_evidence = None
    stop_at = float("inf")
    if not args.tiny_cpu_test:
        if args.guard_evidence is None:
            raise ValueError("Fresh billing and provider-stop evidence is required before GPU allocation")
        guard_evidence, stop_at = validate_guard(args.guard_evidence)
        if shutil.disk_usage(args.output.parent).free < 4 * 1024**3:
            raise ValueError("Need 4GiB free output storage for checkpoint/export reserve")
    torch, GPT, GPTConfig, KVCache, dtype, backend = runtime(tiny=args.tiny_cpu_test,
                           attention=args.attention, optimizer_compile=not args.tiny_cpu_test)
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.tiny_cpu_test:
        torch.cuda.manual_seed_all(args.seed)
    device = "cpu" if args.tiny_cpu_test else "cuda"
    with torch.device("meta"):
        model = GPT(GPTConfig(**c))
    parameters = sum(p.numel() for p in model.parameters())
    if not args.tiny_cpu_test and parameters != 125829314:
        raise ValueError("Unexpected primary parameter count")
    model.to_empty(device=device); model.init_weights()
    optimizer, optimizer_info = setup_optimizer(model, train_config, GPT, GPTConfig, torch)
    initial_hash = model_hash(model)
    state = dict(step=0, row_offset=0, token_offset=0, training_seconds=0.,
                 data_manifest_sha256=digest, initial_weights_sha256=initial_hash, train_config=train_config,
                 initial_evaluation=None, loss_history=[])
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, data_digest=digest, train_config=train_config, torch=torch)
    if state["step"] >= args.max_steps or state["training_seconds"] >= args.compute_seconds:
        raise ValueError("Owned checkpoint has already reached this pilot horizon")
    train = np.load(args.data / "train.npy", mmap_mode="r", allow_pickle=False)
    val = np.load(args.data / "val.npy", mmap_mode="r", allow_pickle=False)
    token_bytes = torch.from_numpy(np.load(args.data / "token_bytes.npy", allow_pickle=False)).to(device)
    tokenizer = load_portable_tokenizer(args.data)
    args.output.mkdir(parents=True)
    write_json(args.output / "run-contract.json", {"model": c, "train": train_config, "parameters": parameters,
               "optimizer": optimizer_info, "source_revision": REVISION, "source_hashes": source_hashes(),
               "data_manifest_sha256": digest, "initial_weights_sha256": state["initial_weights_sha256"],
               "checkpoint_loaded": str(args.resume) if args.resume else None,
               "attention_backend": backend, "fa3_repository": FA3_REPO if backend == "fa3" else None,
               "fa3_revision": FA3_REVISION if backend == "fa3" else None, "torch_version": str(torch.__version__),
               "compute_dtype": str(dtype), "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
               "training_script_sha256": sha256(Path(__file__)), "guard_evidence": guard_evidence,
               "cash_cap_enforced_by_this_script": False,
               "timing_policy": "all synchronous optimizer steps including first compilation; stop between steps; external wall deadline leaves 300s export reserve",
               "schedule_policy": "upstream step schedules; warmup=min(40,max(1,max_steps//10)); time cutoff may precede final annealing",
               "quality_claim": "none; scratch pilot, not a pretrained assistant"})
    if state["initial_evaluation"] is None:
        state["initial_evaluation"] = evaluate(model, val, token_bytes, rows=args.eval_rows,
                                                  batch_rows=args.batch_rows, device=device, torch=torch)
    execution_model = torch.compile(model, dynamic=False) if args.compile_model else model
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    checkpoint = args.output / "checkpoint.pt"
    reason = "max_steps"
    with (args.output / "loss-history.jsonl").open("w") as log:
        for past in state["loss_history"]:
            log.write(json.dumps(past) + "\n")
        while state["step"] < args.max_steps:
            if state["training_seconds"] >= args.compute_seconds or time.time() >= stop_at:
                reason = "compute_limit" if state["training_seconds"] >= args.compute_seconds else "provider_export_reserve"
                break
            start = time.perf_counter()
            offset, loss, multiplier = train_step(model, execution_model, optimizer, train, state["row_offset"],
                                                  state["step"], train_config, optimizer_info, device, torch)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            state["step"] += 1; state["row_offset"] = offset
            state["token_offset"] = offset * c["sequence_len"]
            state["training_seconds"] += elapsed
            item = {"step": state["step"], "loss": loss, "lr_multiplier": multiplier, "step_seconds": elapsed,
                    "training_seconds": state["training_seconds"], "trained_target_positions": state["token_offset"]}
            state["loss_history"].append(item); log.write(json.dumps(item) + "\n"); log.flush()
            print(json.dumps(item), flush=True)
            if args.save_every > 0 and state["step"] % args.save_every == 0:
                save_checkpoint(checkpoint, model, optimizer, state, torch)
            if args.stop_after_steps and state["step"] >= args.stop_after_steps:
                reason = "explicit_test_interruption"; break
    if not state["step"]:
        raise ValueError("No training step completed within safe wall deadline")
    save_checkpoint(checkpoint, model, optimizer, state, torch)
    final_eval = evaluate(model, val, token_bytes, rows=args.eval_rows, batch_rows=args.batch_rows, device=device, torch=torch)
    prompt_text = "The purpose of science is"
    prompt = tokenizer.encode(prompt_text, prepend=tokenizer.get_bos_token_id())
    prompt = prompt[:max(1, min(len(prompt), c["sequence_len"] // 2))]
    n_new = min(16, c["sequence_len"] - len(prompt))
    before = greedy(model, prompt, n_new, device, torch, KVCache, dtype)
    weights_hash = model_hash(model)
    # Reuse the allocation; changing one weight proves reload actually restores it.
    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1)
    load_checkpoint(checkpoint, model, optimizer, data_digest=digest, train_config=train_config, torch=torch, restore=True)
    after = greedy(model, prompt, n_new, device, torch, KVCache, dtype)
    if before != after or weights_hash != model_hash(model):
        raise AssertionError("Owned checkpoint reload/generation roundtrip failed")
    result = {"status": "tiny_cpu_smoke_passed" if args.tiny_cpu_test else "pilot_completed",
              "stop_reason": reason, "step": state["step"], "trained_target_positions": state["token_offset"],
              "training_seconds": state["training_seconds"],
              "target_positions_per_training_second": state["token_offset"] / state["training_seconds"],
              "initial_evaluation": state["initial_evaluation"], "final_evaluation": final_eval,
              "initial_weights_sha256": state["initial_weights_sha256"], "final_weights_sha256": weights_hash,
              "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
              "checkpoint_reload_verified": True, "cached_greedy_roundtrip_verified": True,
              "greedy_prompt_token_ids": prompt, "greedy_new_token_ids": after,
              "greedy_text": tokenizer.decode(prompt + after),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "attention_backend": backend, "source_revision": REVISION, "data_manifest_sha256": digest,
              "not_an_assistant_quality_evaluation": True}
    write_json(args.output / "result.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/nanochat_dense_reference.json")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--guard-evidence", type=Path)
    p.add_argument("--resume", type=Path, help="Only an explicit verified owned scratch checkpoint")
    p.add_argument("--max-steps", type=int, default=1024)
    p.add_argument("--batch-rows", type=int, default=16)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--compute-seconds", type=float, default=525)
    p.add_argument("--eval-rows", type=int, default=128)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--attention", choices=["fa3", "sdpa"], default="fa3")
    p.add_argument("--tiny-cpu-test", action="store_true")
    p.add_argument("--stop-after-steps", type=int, default=0, help="Tiny resume regression interruption only")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.stop_after_steps and not args.tiny_cpu_test:
        raise ValueError("Artificial interruption is reserved for tiny regression tests")
    print(json.dumps(run(args), indent=2))
