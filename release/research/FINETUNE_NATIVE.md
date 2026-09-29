# Fine-tune a native model locally

This additive example changes a verified owned native export using your local
prompt/response data. It does not continue the original experiment's optimizer,
reproduce its SFT run, contact a model service or download weights. In addition
to tiny fixtures, the actual 125,829,314-parameter base passed one CPU update,
export and fresh-process inference check on two synthetic smoke examples. The
step limit deliberately consumed only one example, and the output correctly
declared an incomplete pass with new weights and explicit parent lineage. The
original model stayed unchanged. This was a tool integration check on PyTorch
2.11, not a quality experiment, throughput benchmark or CUDA validation; full-pass
resource requirements and performance remain unmeasured.

Install the original export's `requirements.txt` in an isolated environment.
Keep `finetune_native.py` beside the reviewed `infer_dense.py` in the public tools
folder (or use their original `scripts/` paths). Input model folders remain exact
original exports; no file is added to or edited in them.

Write UTF-8 JSONL with exactly one example per line:

```json
{"prompt":"Rewrite clearly: Please be advised that we need it today.","response":"We need it today."}
{"system":"Answer briefly.","prompt":"What does a thermometer measure?","response":"Temperature."}
```

`prompt` and `response` must be nonempty strings; `system` is optional. Reserved
marker text in your strings is treated as ordinary text. No raw JSONL is copied
into the resulting package. You remain responsible for your data's rights,
privacy and resulting model terms.

Get the trusted parent `export-manifest.json` SHA256 from the verified release
manifest. Then run, replacing the placeholder with that exact hash:

```sh
python tools/finetune_native.py \
  --model models/base \
  --expected-parent-sha256 PARENT_EXPORT_MANIFEST_SHA256 \
  --data train.jsonl --output models/my-adaptation \
  --device cpu --batch-size 4 --learning-rate 0.00001 \
  --max-steps 100 --max-seconds 300
```

Use `--device cuda` only on your own compatible GPU environment. This command
has no provider launcher or billing integration. It uses offline SDPA, FP32
parameters and fresh AdamW; the learning rate warms up over 3% of the planned
finite pass, then follows a cosine decay to 10% of its peak. The original tokenizer
and architecture stay fixed.

Each source row is visited at most once, in file order. Loss covers response
content and the assistant-end token; prompt/system/role prefixes and batch
padding have no target loss. An overlong example fails rather than being
truncated. Supply is capped at 10,000 examples, 32MiB of JSONL and five million
processed positions. There is no repetition to fill a requested step count.

Both limits are mandatory. The time limit is checked before each update; loading,
a running update and final verification/export are not hard-interruptible. If a
limit ends a pass early, `finite_pass_complete` is false and the output is explicitly
`local_finetune_incomplete`. No update means no export. The tool has no exact
training-resume feature; applying it again starts a new finite adaptation with a
fresh optimizer. Use a fresh output directory every time.

The new package includes its own `model.pt` SHA and tensor identity, a hash-only
data record, finite cursor and loss history, the original parent manifest/lineage,
unchanged source notices/licenses and an offline inference entrypoint. Since no
training checkpoint was created, `source_checkpoint_sha256` is null; changed
tensors are never labeled as the parent checkpoint. The parent licenses remain,
but do not automatically license your additional data or establish new rights.

Run the new artifact in a fresh process:

```sh
python models/my-adaptation/infer.py --info
python models/my-adaptation/infer.py --device cpu --chat --max-new-tokens 64
```

`finetune.py` is also included in the new artifact beside `infer.py`, so a later
adaptation can use those two files without the project checkout. A successful
load or falling training loss is not evidence of useful instruction following.
The original model's evaluation scores do **not** apply to the adapted model;
evaluate it separately on held-out material you did not use for training.
