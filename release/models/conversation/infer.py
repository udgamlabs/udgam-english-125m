#!/usr/bin/env python3
"""Standalone offline text generation for an owned native export. MIT licensed.

The exported directory includes this file, model weights, tokenizer ranks,
configuration and pinned nanochat source. No training optimizer/checkpoint,
dataset, pickle, hosted model, calculator or network service is used.
"""
from __future__ import annotations

import argparse
import base64
import codecs
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

FORMAT = "udgam-dense-native-inference-v1"
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def artifact_path(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Unsafe export artifact path")
    path = directory / name
    for parent in (path, *path.parents):
        if parent == directory: break
        if parent.is_symlink(): raise ValueError("Export artifact symlinks are unsupported")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Export artifact escapes its directory")
    return path


def verify_export(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "export-manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("optimizer_included") is not False:
        raise ValueError("An owned inference export without optimizer state is required")
    required = {"model.pt", "model-config.json", "tokenizer.json",
        f"vendor/nanochat-{REVISION}/nanochat/gpt.py", f"vendor/nanochat-{REVISION}/nanochat/common.py",
        f"vendor/nanochat-{REVISION}/nanochat/optim.py", f"vendor/nanochat-{REVISION}/nanochat/flash_attention.py"}
    if not required <= set(manifest["files"]):
        raise ValueError("Inference export is incomplete")
    for name, expected in manifest["files"].items():
        path = artifact_path(directory, name)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Export artifact integrity failure: " + name)
    return manifest


class Tokenizer:
    """Read only our plain JSON ranks, with no tokenizer pickle or Rust trainer."""
    def __init__(self, path, vocab_size):
        import tiktoken
        value = json.loads(Path(path).read_text())
        if value.get("format") != "udgam-owned-rustbpe-v1":
            raise ValueError("Unsupported owned tokenizer format")
        ranks = {base64.b64decode(key, validate=True): rank for key, rank in value["mergeable_ranks"]}
        specials = value["special_tokens"]
        ids = list(ranks.values()) + list(specials.values())
        if len(ranks) != len(value["mergeable_ranks"]) or len(ids) != vocab_size or set(ids) != set(range(vocab_size)):
            raise ValueError("Tokenizer ranks do not match the exported model vocabulary")
        self.enc = tiktoken.Encoding(name="owned-native-inference", pat_str=value["pattern"],
                                    mergeable_ranks=ranks, special_tokens=specials)
        self.specials = specials; self.bos = specials[value["bos_token"]]

    def encode(self, text): return self.enc.encode_ordinary(text)
    def decode(self, ids): return self.enc.decode(ids)
    def token_bytes(self, token): return self.enc.decode_single_token_bytes(token)

    def prompt(self, text):
        return [self.bos] + self.encode(text)

    def conversation(self, messages, system=""):
        # Matches nanochat's text-only render_conversation: optional system text
        # is merged into the first user turn. Content never injects token markers.
        if not messages or len(messages) % 2 != 1:
            raise ValueError("Conversation must end with a user turn")
        required = ("<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>")
        if any(token not in self.specials for token in required):
            raise ValueError("This tokenizer has no native conversation markers")
        ids = [self.bos]
        for index, message in enumerate(messages):
            role = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != role or not isinstance(message.get("content"), str):
                raise ValueError("Text conversations must alternate user and assistant")
            content = (system + "\n\n" if index == 0 and system else "") + message["content"]
            ids += [self.specials[f"<|{role}_start|>"]] + self.encode(content) + [self.specials[f"<|{role}_end|>"]]
        return ids + [self.specials["<|assistant_start|>"]]


class KVCache:
    """Text-only cache protocol adapted from the bundled MIT nanochat engine.

    The engine/tool module is intentionally not imported. These are plain key,
    value, position and previous-embedding tensors consumed by GPT.forward.
    """
    def __init__(self, config, device, dtype, torch):
        self.n_layers = config.n_layer
        shape = (config.n_layer, 1, config.sequence_len, config.n_kv_head, config.n_embd // config.n_head)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(1, device=device, dtype=torch.int32)
        self.prev_embedding = None

    def get_pos(self): return self.cache_seqlens[0].item()
    def get_layer_cache(self, index): return self.k_cache[index], self.v_cache[index]
    def advance(self, count): self.cache_seqlens += count


def load(directory, device="auto"):
    started = time.perf_counter()
    directory = Path(directory).resolve(); manifest = verify_export(directory)
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Unsupported device")
    if device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available(): raise ValueError("MPS is unavailable")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    os.environ["NANOCHAT_DTYPE"] = "bfloat16" if dtype == torch.bfloat16 else "float32"
    os.environ["HF_HUB_OFFLINE"] = "1"
    vendor = directory / "vendor" / ("nanochat-" + REVISION)
    for name, module in list(sys.modules.items()):
        if name == "nanochat" or name.startswith("nanochat."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(vendor):
                raise ValueError("Use a fresh process: another nanochat source is already imported")
    sys.path.insert(0, str(vendor))
    # Upstream attempts a dynamic FA3 download on compatible CUDA. Explicitly
    # select portable SDPA without importing a kernel downloader at all.
    blocker = types.ModuleType("kernels")
    def no_download(*args, **kwargs): raise RuntimeError("Standalone inference is offline SDPA")
    blocker.get_kernel = no_download; blocker.has_kernel = lambda *a, **k: False
    previous_kernel_module = sys.modules.get("kernels")
    sys.modules["kernels"] = blocker
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from nanochat.gpt import GPT, GPTConfig
            from nanochat import common, flash_attention
    finally:
        if previous_kernel_module is None: sys.modules.pop("kernels", None)
        else: sys.modules["kernels"] = previous_kernel_module
    if common.COMPUTE_DTYPE != dtype:
        raise ValueError("Use a fresh process when changing compute device/dtype")
    flash_attention.USE_FA3 = False
    payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    allowed = {"format", "model", "model_config", "source_revision", "source_checkpoint_sha256"}
    if set(payload) != allowed or payload["format"] != FORMAT or payload["source_revision"] != REVISION:
        raise ValueError("Unexpected model payload; training state is not an inference artifact")
    if payload["source_checkpoint_sha256"] != manifest["source_checkpoint_sha256"]:
        raise ValueError("Export checkpoint lineage mismatch")
    config = json.loads((directory / "model-config.json").read_text())
    if config != payload["model_config"]: raise ValueError("Export configuration mismatch")
    with torch.device("meta"), contextlib.redirect_stdout(sys.stderr): model = GPT(GPTConfig(**config))
    # Assign owned tensors directly, preserving their saved dtypes. Only the
    # nonpersistent rotary buffers need reconstruction; no random/full duplicate
    # model initialization or training optimizer allocation is performed.
    model.load_state_dict(payload["model"], strict=True, assign=True)
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, config["n_embd"] // config["n_head"], device="cpu")
    model.to(device); model.eval(); model.requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != manifest["total_parameters"]:
        raise ValueError("Export parameter count mismatch")
    tokenizer = Tokenizer(directory / "tokenizer.json", config["vocab_size"])
    if device == "cuda": torch.cuda.synchronize()
    metadata = {"device": device, "compute_dtype": str(dtype), "attention": "sdpa", "torch": str(torch.__version__),
        "cold_load_seconds": time.perf_counter() - started, "training_optimizer_loaded": False,
        "external_tools": False, "network_downloads": False, "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "model_stage": manifest.get("model_stage", "base_pretraining"), "context_tokens": config["sequence_len"]}
    return model, tokenizer, torch, dtype, metadata


def generate(model, tokenizer, tokens, torch, dtype, *, max_new_tokens=128, temperature=0., top_k=0, seed=42, on_text=None):
    if (type(max_new_tokens) is not int or max_new_tokens < 1 or type(top_k) is not int or top_k < 0 or
        isinstance(temperature, bool) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("Invalid generation bound or sampling setting")
    if not tokens or len(tokens) + max_new_tokens > model.config.sequence_len:
        raise ValueError("Prompt plus response exceeds context; shorten it or reset the conversation")
    device = next(model.parameters()).device
    cache = KVCache(model.config, device, dtype, torch)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    generated, fragments = [], []
    started = time.perf_counter(); first_token_seconds = None; reason = "length"
    with torch.inference_mode():
        logits = model(torch.tensor([tokens], device=device), kv_cache=cache)[0, -1]
        for index in range(max_new_tokens):
            if temperature == 0:
                token = int(logits.argmax())
            else:
                scores = logits.float() / temperature
                if top_k:
                    threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                    scores = scores.masked_fill(scores < threshold, -float("inf"))
                token = int(torch.multinomial(torch.softmax(scores, -1).cpu(), 1, generator=generator))
            if first_token_seconds is None: first_token_seconds = time.perf_counter() - started
            if token in tokenizer.specials.values():
                reason = "end_marker"; break
            generated.append(token)
            fragment = decoder.decode(tokenizer.token_bytes(token))
            fragments.append(fragment)
            if on_text and fragment: on_text(fragment)
            if index + 1 < max_new_tokens:
                logits = model(torch.tensor([[token]], device=device), kv_cache=cache)[0, -1]
    tail = decoder.decode(b"", final=True); fragments.append(tail)
    if on_text and tail: on_text(tail)
    return {"text": "".join(fragments), "token_ids": generated, "prompt_tokens": len(tokens),
            "generated_tokens": len(generated), "stop_reason": reason, "first_token_seconds": first_token_seconds,
            "generation_seconds": time.perf_counter() - started, "seed": seed, "temperature": temperature, "top_k": top_k}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--prompt")
    p.add_argument("--chat", action="store_true", help="Native conversation format; interactive when --prompt is absent")
    p.add_argument("--system", default="")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true", help="Machine-readable one-prompt text and timing")
    p.add_argument("--info", action="store_true", help="Verify and show export metadata without loading tensors")
    args = p.parse_args(argv)
    if args.info:
        print(json.dumps(verify_export(args.model), indent=2)); return 0
    if args.prompt is None and not args.chat: p.error("Use --prompt TEXT or --chat")
    if args.json and args.prompt is None: p.error("--json requires --prompt")
    if args.system and not args.chat: p.error("--system requires --chat")
    model, tokenizer, torch, dtype, metadata = load(args.model, args.device)
    options = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    output = lambda text: print(text, end="", flush=True)
    if args.prompt is not None:
        tokens = tokenizer.conversation([{"role": "user", "content": args.prompt}], args.system) if args.chat else tokenizer.prompt(args.prompt)
        result = generate(model, tokenizer, tokens, torch, dtype, on_text=None if args.json else output, **options)
        if args.json: print(json.dumps({**result, "runtime": metadata}, ensure_ascii=False))
        else: print()
        return 0
    print(f"Local {metadata['model_stage']} model. /reset clears context; /exit quits.")
    history = []
    while True:
        try: text = input("You: ")
        except (EOFError, KeyboardInterrupt): print(); break
        if text.strip() == "/exit": break
        if text.strip() == "/reset": history.clear(); print("Context cleared."); continue
        if not text.strip(): continue
        candidate = history + [{"role": "user", "content": text}]
        try:
            tokens = tokenizer.conversation(candidate, args.system)
            print("Model: ", end="", flush=True)
            result = generate(model, tokenizer, tokens, torch, dtype, on_text=output, **options); print()
        except ValueError as error:
            print(str(error), file=sys.stderr); continue
        history = candidate + [{"role": "assistant", "content": result["text"]}]
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print("Error: " + str(error), file=sys.stderr); raise SystemExit(2)
