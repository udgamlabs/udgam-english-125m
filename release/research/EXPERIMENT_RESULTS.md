# Udgam English 125M: completed experiment, performance target missed

**24 September 2026.** Training and all 12 cloud evaluation jobs are complete. Every evaluation artifact and both conversation checkpoints have been returned and checked against the committed run. Both native inference exports match their selected checkpoints tensor for tensor and run from separate folders in fresh CPU processes without cloud access. The complete sealed training-input backup and returned artifacts have now passed full local portable provenance validation. No public publication has occurred; distribution integrity is checked separately through the release manifest and checksums.

The 125,829,314-parameter English candidate started from random weights and an owned 16,384-token BPE tokenizer. It completed 2,033,549,312 pretraining targets including its pilot, followed by one finite conversation pass over 7,237 prepared examples and 1,892,764 assistant targets. The India-based lab's target audience is global.

## Result

**This recipe did not beat Sarvam-1, did not pass the agreed general-quality floors, and did not establish a Pareto-frontier result.** The small same-scale SmolLM2-135M baseline also scored higher. This is a completed low-cost training experiment with a negative performance result, not a reliable general assistant. It does not prove that every possible under-$100 approach must fail.

The fixed suite contains 500 items from each task, with ARC-Challenge/HellaSwag/PIQA using normalized accuracy and BoolQ/WinoGrande using accuracy. Values below are percentages; the macro average weights all five tasks equally. Every model scored all 2,500 items, with zero unsupported items.

| Model | ARC-C | HellaSwag | PIQA | BoolQ | WinoGrande | Macro |
|---|---:|---:|---:|---:|---:|---:|
| Udgam base | 25.80 | 33.60 | 58.60 | 59.60 | 51.00 | **45.72** |
| Udgam conversation | 26.40 | 34.20 | 58.40 | 61.80 | 51.20 | **46.40** |
| Sarvam-1 | 51.20 | 67.60 | 75.20 | 83.40 | 63.00 | **68.08** |
| SmolLM2-135M | 30.00 | 44.20 | 69.60 | 60.40 | 54.60 | **51.76** |
| SmolLM2-360M | 35.20 | 57.60 | 72.20 | 61.40 | 60.60 | **57.40** |

Both owned models failed all five task floors and the 52% macro floor. BoolQ additionally requires beating its selected-suite majority baseline. The conversation model improved the macro point score by 0.68 percentage points over the base, which is not a claim of a statistically established improvement.

Against Sarvam-1, the base difference was −22.36 percentage points (paired source-cluster bootstrap 95% interval −24.55 to −20.15); the conversation difference was −21.68 points (−23.84 to −19.48). All six fixed owned-model/baseline comparisons are retained. Their pairwise intervals are unadjusted; cross-candidate comparisons are exploratory under the frozen protocol. No checkpoint or winner was selected from these results.

Sarvam-1 has 2,525,087,744 actual stored parameters in this evaluation, about twenty times the candidate's count. SmolLM2-135M is the closer parameter-scale reference. Their training investment is not inferred from this experiment.

## Efficiency measurements and their limits

All five models completed the same 30 supported prompt strings in the forced 128-token GPU microbenchmark. These are implementation measurements, not typical user-response latency or isolated architecture comparisons.

| Model | Parameters | Inference package MB, decimal | Median decode tokens/s | Median first-token ms |
|---|---:|---:|---:|---:|
| Udgam base | 125,829,314 | 428.80 | 184.4 | 5.98 |
| Udgam conversation | 125,829,314 | 504.32 | 180.7 | 6.10 |
| Sarvam-1 | 2,525,087,744 | 5061.47 | 94.8 | 11.22 |
| SmolLM2-135M | 134,515,008 | 272.45 | 85.4 | 12.91 |
| SmolLM2-360M | 361,821,120 | 727.06 | 80.2 | 13.62 |

The owned scoring/latency path uses FA3 and BF16 compute; baselines use SDPA with BF16 parameter storage. The owned base export preserves mixed parameter dtypes, while the conversation export has FP32 master parameters. Consequently, the owned native packages are **larger on disk than SmolLM2-135M**, despite having fewer parameters. The separate portable offline inference entrypoint uses SDPA; the FA3 measurements above must not be advertised as its speed.

Peak CUDA-memory scopes differ: the owned evaluator captured peaks before optional export/latency, while the baseline evaluator captured them afterward. Those figures are retained in the machine-readable record but do not establish matched standalone inference memory or resource dominance. No global Pareto frontier follows from this comparison.

## Native instruction diagnostics

Both owned models scored **0/24** on the original automatic instruction checks. These are small strict diagnostics, not a full instruction-following benchmark or a comparison against external chat models. All 36 responses per model were generated; the 12 human-rubric items remain unscored, with human review explicitly incomplete. The automatic result further limits any assistant-quality claim. An independent audit found no demonstrated prompt, mask or scoring bug. Some generated content was semantically relevant but failed exact formatting: for example, a requested single-word answer appeared inside a longer sentence. Thus 0/24 is the strict diagnostic pass rate, not a claim that every generated token lacked useful content.

## Cost record

No GPU remains active. The completed evaluation function used $1.09585349 in metered resources. At compute close, the dated Modal workspace gross summary was $15.57605529, credits applied were $15.01, and Modal paid charges were $0. Adding the previously recorded $0.87 Runpod consumption gives **$16.44605529 of observed gross provider usage**, before final storage/transfer settlement. This is below the $99 project ceiling; no top-up is needed for the completed compute work.

The detailed Modal per-run resource reports total $13.90959718. Their $1.66645811 difference from the workspace gross summary is retained as unresolved accounting/timing/storage scope, not invented into particular runs. The failed first conversation attempt's $0.74201848 is included. Holds, a legacy $7 Runpod allocation reservation with no confirmed resource, and a separate conservative unallocated allowance are not presented as actual spending.

Free credits are distinct from economic compute value. Existing Mac compute/storage, internet and engineering/supervision time were not priced, so this is not a claim that the full human effort costs $16.45. Another lab's prices, setup efficiency and credits may differ. The sanitized compute-close ledger preserves successful and failed attempts; final cleanup and settlement remain pending.

A later dated account check, after backup compression, reports about $16.58 of observed gross provider usage and $0 Modal paid charges. This updates the observation date; it does not replace the historical compute-close figures above or turn unsettled storage/transfer amounts into a final invoice. Refer to the dated public cost summary for the latest retained observation.

## Reproducibility and release

The recipe and exact initialization/tokenizer/data/model hashes are documented in the reproduction guide. It uses public educational and synthetic text with disclosed teachers, pinned nanochat code and one finite OASST2 conversation pass. No external pretrained weights, embeddings or tokenizer vocabulary initialize the candidate. A recorded future-data repair removed one captured benchmark family outside the already accepted prefix; that does not prove zero paraphrased contamination.

The local package prepared for public distribution contains both exact native exports, owned code/tokenizer under MIT, upstream notices, public acquisition and training source, these results and the cost record. Raw third-party text, baseline weights, private operational/account records and credentials are excluded. A separate offline JSONL fine-tuning tool passed tiny fixtures and a one-update integration check on the actual 125M base, followed by a fresh CPU load. That synthetic smoke adaptation was deliberately partial, remained private and was never evaluated or selected as another candidate. Adapting a model creates new hashes and does not inherit these benchmark scores; full-pass resource needs and CUDA adaptation remain unmeasured.

Full private recovery/provenance verification is complete: all returned artifact bytes and the entire sealed training input passed their inventories, checkpoint/export tensors match exactly, and both native models passed relocated standalone loads. The portable validator changes only one comparison of recomputed positive finite historical learning rates, allowing at most four binary64 ULP; the original stored records, optimizer endpoint and all other checks remain exact. It does not replay the original Linux math implementation. The separately labeled scientific public summary identifies the original private evidence by hash; operational records remain private. This package has not been published publicly.

Evidence: `records/final-comparison-001/analysis-manifest.json`, `metadata-verification.json`, `quality-and-resources.json` and all six pairwise records; `records/modal-compute-closed-cost-basis-001.json`; the exact returned instruction summaries. See `REPRODUCE_ENGLISH_125M.md`, `evaluation_protocol.md`, and `FINETUNE_NATIVE.md` for methods and limitations.
