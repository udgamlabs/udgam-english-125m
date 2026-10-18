# Udgam English 125M v0.1.0

First public distribution of the sealed 24 September 2026 experiment. The publication wrapper adds current navigation and download instructions; model weights, code, tokenizer and scientific evidence are unchanged.

- **Two native models:** base text completion and a conversation-tuned variant, each with 125,829,314 parameters and a 2,048-token context.
- **Scratch lineage:** random initial weights, owned 16,384-token BPE, 2,033,549,312 pretraining targets and one finite conversation pass over 7,237 prepared OASST2 examples.
- **Runnable tools:** offline native inference and an additive JSONL fine-tuning tool. Both original exports passed relocated CPU loads and exact checkpoint-tensor comparison. The fine-tuner passed tiny CPU tests and one update on the actual base; full-pass and CUDA adaptation are unmeasured.
- **Negative result retained:** base **45.72%**, conversation **46.40%**, Sarvam-1 **68.08%**, SmolLM2-135M **51.76%**, SmolLM2-360M **57.40%** on the same frozen 2,500-item suite. Both owned models failed all quality floors and passed **0/24** strict instruction diagnostics. Twelve human-rubric items per variant remain unscored. No reliable-assistant or Pareto-frontier result is claimed.
- **Open evidence:** source/configurations, pinned attribution, public provenance summary, per-item numeric outcomes, all six comparisons and the dated cost record. Raw source text, private training traces, baseline weights, optimizer backups and account records are excluded.

## Downloads and integrity

The Hugging Face snapshot at `udgamlabs/udgam-english-125m`, tag `v0.1.0`, contains the complete unchanged distribution under `release/`. The GitHub repository is source-only and omits both `model.pt` files. The attached original sealed ZIP is the complete distribution; GitHub's automatic source-code archives are not substitutes for it.

| Artifact | SHA256 |
|---|---|
| `udgam-english-125m-20260924.zip` (861,034,373 bytes) | `14331f0eca671d77e70483bfea508f9dfa9e61e835ced00db614056ceecc5fce` |
| Complete distribution `RELEASE_MANIFEST.json` | `13f9eff874e58c8bbf10d8bf14959a9aece76a7be10a764b1fb10439f9bdfa40` |
| `models/base/model.pt` | `9f91792f90160991a8e7c58922e65e88ed6d0c6c5ef3720649912f8ddf35cf73` |
| `models/conversation/model.pt` | `3c08b783626915fad21231ff479e71fc8c24b8812e9d5168df3065ff160628e8` |

The complete sealed directory contains 193 files / 947,854,673 bytes, including manifest and checksums. Current publication-wrapper files are additional and are outside that historical manifest. Verify `SHA256SUMS` from the complete distribution before use; native loaders additionally check their own inventories.

## Known limits

Native PyTorch only: no Transformers conversion, GGUF, quantization, mobile validation or hosted inference service. The published GPU latency microbenchmark used a different backend from portable native inference and must not be presented as its speed. The source records explain the future-data repair and the narrow at-most-four-ULP portable learning-rate-history comparison; neither permits altered weights or broad approximate validation.

The retained 24 September 2026, 17:18 UTC cost observation is about **US$16.58 gross provider usage**, with **US$0 Modal paid charges**. Cleanup, access revocation and storage/transfer settlement were pending in that dated record; it is not a settled bill or new account check. The existing Mac, internet and engineering time are unpriced.

Owned code, documentation, tokenizer and weights are MIT licensed; upstream and dataset terms remain in force. Full historical cloud training requires pinned inputs, provenance and real lifecycle controls. Standalone inference and local fine-tuning are supported entrypoints; the historical recipe is not a one-command cloud retraining service.
