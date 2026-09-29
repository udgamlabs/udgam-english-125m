# Owned native text model

This folder is a standalone local inference package: owned model tensors, the
owned tokenizer, pinned nanochat source and `infer.py`. It needs no training
checkpoint, dataset, original project checkout, API key or hosted service.
The export manifest records the training checkpoint and available random-weight
lineage. This interface does not establish model quality.

Use Python 3.11 or 3.12. Install dependencies once in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

For CUDA, install the matching official PyTorch wheel for your hardware. Runtime
inference is offline and uses PyTorch SDPA with a key/value cache. It makes no
kernel or model downloads. This portable path has not been tuned for peak GPU
decode speed; FA3 training throughput is not its inference speed.

```sh
python infer.py --info
python infer.py --device cpu --prompt "A useful way to check an explanation is" --max-new-tokens 64
python infer.py --chat
python infer.py --chat --prompt "Rewrite this sentence clearly: The report is hard to understand." --max-new-tokens 80 --json
```

The default is reproducible greedy decoding. Optional sampling uses
`--temperature 0.7 --top-k 40 --seed 42`. `--device auto` selects CUDA, MPS, then CPU
when available; CPU is the portable smoke-tested path. Device support and speed
must be measured on the target system. Models retain their stored parameter
dtypes; compute uses BF16 on capable CUDA and FP32 otherwise.

`--chat` uses the native user/assistant boundary tokens planned for instruction
tuning. A `base_pretraining` export has no recorded chat tuning, so use completion
mode to assess its learned text continuation. Chat formatting alone does not
turn it into a capable assistant. `--system` merges text into the first user turn,
matching the source tokenizer convention.

In interactive chat, `/reset` clears the conversation and `/exit` quits. History
stays in memory and is not saved. The tool refuses requests whose prompt plus
requested response exceeds the model context; shorten the request or reset.
The model never executes code, tools or generated commands. Reserved boundary
tokens end generation and are not printed as conversation text.

Every listed artifact is hash-checked before loading. Tensors are loaded with
PyTorch `weights_only=True` and memory mapping; no tokenizer pickle or optimizer
state is read. Cold-load timing in JSON output measures this inference artifact,
including integrity checks and dependency imports, rather than a training resume.

The root `LICENSE` and `LICENSE_SCOPE.md` establish MIT terms for this experiment's
original code, tokenizer and owned model weights when released. `LICENSE.inference`
also covers the entrypoint/documentation. Upstream source licensing is retained
under `vendor/`; actual training source cards/terms are copied under `attribution/`
when available and remain separately licensed. The accompanying model card must
state release status, measured quality and remaining provenance limitations.
This folder is a native Python/PyTorch format, not a Transformers or GGUF model.
