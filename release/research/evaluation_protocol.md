# English quality and efficiency evaluation protocol

Date: 2026-09-22. Status: protocol prepared; no results, model downloads, paid resources, or completed benchmark runs are implied.

## Objective and limits

Evaluate a small, general English language model trained from random initial weights by an India-based lab, within a total experiment budget below US$100. Server geography is unrestricted. Measure useful quality together with actual storage, memory, latency and cost. English is the primary language; this is not an Indic-language or command-parser benchmark.

The defensible target is an attractive measured quality/resource tradeoff among named baselines. General superiority over Sarvam-1 is an experimental hypothesis, not an expected or guaranteed outcome. Efficient training recipes and a lower validation loss do not establish that superiority. A $100 ceiling also limits replication, hyperparameter search and the strength of any broad claim.

## Models and immutable records

Use the following comparison order; freeze checkpoint revisions before any final candidate scores are inspected:

| Priority | Model | Role |
| --- | --- | --- |
| Required | Our final random-initialized base checkpoint | Candidate; choose once using training-development loss and the declared compute limit |
| Required | `sarvamai/sarvam-1` | Named comparison; base completion model |
| Required | `HuggingFaceTB/SmolLM2-135M` | Existing very small English base baseline |
| Required | `HuggingFaceTB/SmolLM2-360M` | Second size point; prevents a misleading comparison only against a much larger model |
| Budget-conditional, declared before scores | `Qwen/Qwen3.5-0.8B-Base` | Contemporary small pretrained baseline; requires compatible text-only scoring support |

Sarvam describes Sarvam-1 as a 2B text-completion model, explicitly unsuitable for direct chat without adaptation. Its advertised Indic scores are not our English scores. Its repository currently lists about 5.06 GB of files and the Hub rounds its tensor count to 3B: record the actual count and bytes instead of relying on the marketing size. [Official model card](https://huggingface.co/sarvamai/sarvam-1), [file inventory](https://huggingface.co/sarvamai/sarvam-1/tree/main).

SmolLM2's 135M and 360M base checkpoints are appropriate size references; use base variants here. Qwen explicitly identifies its `-Base` repository as pretrained-only, although its generic overview includes post-training language. It includes a vision component: record both complete downloaded size and the text-only loaded subset, if supported. [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base).

Pretrained baselines supply evaluation results only: no baseline weights, hidden states, logits or generated answers enter candidate training. Sarvam-M is optional context in a separate instruction-model table, never a substitute for Sarvam-1. A baseline's restrictive license does not become our release license merely by comparing scores; do not redistribute its weights as part of our release.

The run manifest must contain full model/code/data commit hashes, checkpoint SHA-256, tokenizer files and hash, framework versions, task YAML hashes, precision, parameter count, context length, seeds, raw provider quote, final bill, and candidate training/SFT lineage. Record unavailable revisions as unresolved; do not silently use a moving `main` branch. This protocol is not fully registered until the selected item manifest and these revisions are saved before final scoring.

## Fixed base-model suite: 2,500 items

Run zero-shot continuation likelihood, with the pinned EleutherAI task definitions. Do not apply a chat template or system instruction. Score every answer continuation, not just the probability of letters A/B/C/D. Use each model's own tokenizer and the task's existing normalization. The aggregate below is our custom small suite, not an official leaderboard score.

| Task | Fixed public split and count | Main metric | Coverage | Minimum point-score floor |
| --- | --- | --- | --- | --- |
| ARC-Challenge | test, 500 | `acc_norm` | Elementary science knowledge and reasoning | 30% |
| HellaSwag | validation, 500 | `acc_norm` | Plausible event completion | 35% |
| PIQA | validation, 500 | `acc_norm` | Physical commonsense | 60% |
| BoolQ | validation, 500 | `acc` | Reading and yes/no comprehension | 65%, and above the selected-set majority baseline |
| WinoGrande XL | validation, 500 | `acc` | Reference resolution and commonsense | 55% |

Task definitions and current dataset routing: [ARC](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_challenge.yaml), [HellaSwag](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/hellaswag.yaml), [PIQA](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml), [BoolQ](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/super_glue/boolq/default.yaml), [WinoGrande](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/winogrande/default.yaml). Verify mirror content and license against the linked task's original dataset before redistribution; a loader mirror is not an independent provenance guarantee.

Selection is deterministic: for each task, hash `english-small-v1|20260922|task_name|original_split_row_index|canonical_document_sha256`, sort the hexadecimal hashes ascending, and take 500 distinct documents. Canonical JSON uses sorted keys, UTF-8, no extra whitespace. Persist original row indices, IDs, hashes, dataset revision and exact order. Do not balance labels, remove difficult examples, swap tasks, or redraw this sample after seeing scores. Duplicate source documents are grouped before sampling; preserve the earliest original row index for each identical document.

The primary score Q is the unweighted mean of the five percentages. Report every task, its sample size, raw `acc` alongside `acc_norm` where available, and the selected-set majority/choice baseline. The proposed research floor is Q >= 52% and every task floor above; these are deliberately explicit project thresholds, not an established standard of general intelligence or deployment readiness. Show all failed floors.

The 500-item results have limited precision: around 50% accuracy, an individual task's binomial standard error is roughly 2.2 percentage points. The suite does not establish coding, arithmetic, factual reliability, long-context use, multi-turn ability or safety. A product claim covering those capabilities needs additional tests.

## Fair scoring and leakage controls

1. Keep final benchmark rows and labels out of pretraining, tokenizer fitting, SFT, synthetic-data generation, optimizer selection and checkpoint selection. Use separate training-development text for loss and 20 disjoint training-split examples per task for adapter smoke checks. Public validation splits are final holdouts for this experiment, not proof of secrecy from historical model pretraining.
2. Before candidate training, scan the actual selected training documents against benchmark questions/passages and full examples using normalized exact matches and overlapping 13-word sequences. Remove matching training documents and record counts; do not remove the corresponding test items to improve scores. Add a documented near-duplicate pass where feasible. This reduces detectable overlap; it cannot prove absence of paraphrase or baseline contamination. Do not promise a contamination-free comparison when Sarvam's full training corpus is unavailable.
3. Freeze a common scoring context of 2,048 tokens, provided the candidate natively supports it. If it does not, register its smaller supported context before scoring. Preserve the same full input strings across models. Flag any example that does not fit with its answer under any tokenizer; score that item as unsupported/incorrect for that model in the primary result, rather than silently truncating, dropping, or replacing it. Report a common-fit diagnostic separately. Do not extend RoPE after observing test failures.
4. Run all base models in BF16 where supported, with FP32 log-softmax/summation where practical. Do not handicap only a baseline with low-bit quantization. A separately labeled quantized deployment comparison must remeasure quality and resources for every included artifact.
5. The candidate may need a custom `lm_eval.api.model.LM` adapter rather than an HF export. Check next-token alignment, prompt/answer boundary tokenization, BOS/EOS handling, padding, causal masking, and that only continuation tokens contribute to log likelihood. Require native-versus-adapter likelihood agreement on disjoint smoke strings; investigate differences above 1e-3 nats/token against the same FP32 reference path before running benchmarks. Do not force an incompatible architecture through an approximate converter. [Harness custom-model interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md).
6. No external retrieval, tools, answer repair, benchmark-specific prompt search, repeated sampling or LLM judge in the primary suite. Preserve numeric answer scores and item outcomes, not just averages. If a correctness bug forces a rerun, document it and rerun all affected models under the same fixed rules.

## Uncertainty and interpretation

For each candidate/baseline pair, compute paired differences using the same item IDs. Bootstrap 10,000 times with seed 20260922, resampling within each task and retaining pairs; resample source-document clusters together when several questions share a source. Compute a 95% percentile interval for the macro difference and each task difference. Bootstrap inference is conditional on the selected benchmark distribution and the fixed trained checkpoint; it does not measure training-seed variability or all real-world usage.

Choose one final candidate without final-test feedback. If the budget permits additional training seeds, report all of them rather than the best seed. If multiple candidate designs are compared on this final suite, relabel it development data and obtain a new final holdout before making confirmatory claims. Task-specific superiority claims require multiplicity correction or explicit exploratory labeling.

Predeclared conclusions:

- **Better on this base suite:** Q is at least 2 percentage points above Sarvam-1, the paired 95% interval for the macro difference is wholly above zero, all quality floors pass, and no individual task point score is more than 5 points below Sarvam-1. Still say "on this 2,500-item English suite", not "better than Sarvam AI".
- **Useful smaller tradeoff:** all floors pass; macro quality is non-inferior within 3 points (lower paired 95% bound >= -3 points); no task point deficit exceeds 5 points; complete inference artifacts are at least 2x smaller; measured peak memory is at least 25% lower. Report any latency/cost regressions. This permits a small quality concession and therefore is not strict Pareto dominance.
- **Observed Pareto point:** no evaluated model/artifact is at least as good in Q while using no more bytes, peak memory, latency and cost, with one strict advantage. Label this descriptive and limited to the tested set. A confidence-supported dominance claim requires non-negative quality-difference bounds and measured resource improvements; overlapping intervals are inconclusive.
- **Otherwise:** publish the actual tradeoff and failed gates. A large relative gain over random guessing, a single narrow task win, or lower pretraining BPB does not satisfy the broad claim.

## Instruction and generation: separate product track

If a chat/SFT version is produced from our own random-initialized base, preserve both checkpoints and count SFT compute/data in the $100 ledger. Test the product version against appropriate instruction counterparts, such as SmolLM2-135M-Instruct and SmolLM2-360M-Instruct; Sarvam-1's chat weakness is not the success criterion.

Use a fixed 100-item sample of Google IFEval, selected by the same hash rule with task name `ifeval`. Reserve it from all training. Use official verifiers and the pinned harness's 1,280-token generation cap; greedy decoding, no repair, no tools, each instruction model's documented native chat template, and no extra custom system prompt. Report strict prompt-level and strict instruction-level accuracy, plus loose metrics as secondary. Log truncations. If any model cannot support prompt plus output budget, disclose the limitation; a changed cap is a separately named derivative test. [Official IFEval implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval), [harness configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/ifeval.yaml).

A provisional product gate is >=40% strict prompt accuracy and no more than a 5-point deficit to SmolLM2-135M-Instruct, plus review of every failure category. This is a research floor, not a claim of reliable instruction compliance. IFEval measures verifiable constraints; it does not establish factual correctness, helpfulness or coherent conversation. If human review is available, use 30 newly written prompts across explanation, summarization, everyday reasoning, writing and ambiguity; blind model identities, retain all responses, and label a single-reviewer result exploratory. Without such review, explicitly leave broad assistant usefulness unvalidated. Keep this track out of Q and the base-model victory claim.

## Measure actual efficiency

On the same rented GPU/CPU allocation and software versions, sequentially measure every mandatory base checkpoint; use the same runtime family where supported. If architecture-specific kernels differ, identify the result as whole-system performance rather than an isolated architecture effect.

Record total stored parameters and total inference-artifact bytes, including embeddings, output heads, n-gram/lookup tables, sparse arrays, indexes, quantization scales, tokenizer and required auxiliary files. Count CPU-resident and memory-mapped tables too; "active parameters" is an additional figure, never a replacement. Exclude optimizer states from inference-artifact bytes but report them separately as training storage. This matters especially for memory-augmented speedrun architectures.

Record idle-loaded and peak GPU memory, allocator-reserved memory, process peak RAM, and cold-load time. Measure batch-one time to first token and decode speed on 30 fixed English development prompts, with the same input text (roughly 128, 512 and 2,048 bytes, ten prompts each). Use five unreported warmups followed by the fixed workload, synchronizing the GPU at measurement boundaries. Report median and p95 latency, output tokens/sec and UTF-8 bytes/sec, actual output lengths, EOS rate and prompt token counts. Include a separate forced 128-token microbenchmark if desired; do not represent forced-length decoding as user response latency. A repeated long response that produces more bytes is not inherently better quality.

Report cost for the fixed workload from measured elapsed time and the exact instance quote, plus storage. Also show sustained batch-one dollars per million output tokens with its tokenizer caveat and cost per completed fixed prompt. Do not infer phone/Mac/browser performance from H200 results. A local-device claim requires a compatible exported artifact and a separate on-device run.

## Practical execution within the evaluation and disk limits

Reserve **at most $8 for the full evaluation phase**, including baseline downloads/setup, candidate scoring, permitted optional tests, storage and transfer; the whole project must still remain below $100. The current reported Runpod credit is $8.80, not an authorization to consume it all or a guarantee of total available budget. This document launches nothing.

Use the already-authorized training GPU for a short evaluation if that avoids another setup charge, or a single compatible 24GB-or-larger GPU with an all-in quote at most $1.50/hour. This is a procurement ceiling, not a verified market price or throughput promise. Reserve $1 of the $8 envelope for storage/transfer/rounding, and stop compute before `(7 / quoted_gpu_hourly_rate)` hours, minus time already billed in evaluation. An external termination deadline and balance monitor are required; exiting Python does not stop a rental. Stop cleanly and report incomplete results if the ceiling would be exceeded.

First time 20 disjoint development examples per task on Sarvam-1. Extrapolate by observed prompt and continuation lengths, with 2x headroom for the full mandatory workload. Keep per-model evaluation response caches and item logs so a recovery does not repeat completed inference. Omit Qwen and instruction extras first if the projection exceeds the envelope; if the mandatory set does not fit, do not shrink or cherry-pick it after scores. The main suite entails roughly 7,000 answer-continuation scores per model (ARC's choice count varies), and the three tiny-model passes should be cheaper than the Sarvam pass, subject to adapter/kernel support.

With only **11 GiB free on the local Mac**, do not place baseline weights, CUDA wheels or full training datasets there. Run model loading and dataset preparation on the rented instance, with at least 25 GiB free there before evaluation (50 GiB preferable if the already-funded volume permits it). Keep one model snapshot at a time if space is tight. Sarvam's listed 5.06 GB artifact alone uses a substantial part of local headroom; an HF cache plus a separate copied snapshot wastes more. Keep local research, manifests and compressed result files below 250 MiB; inspect projected bytes before any transfer. [Sarvam inventory](https://huggingface.co/sarvamai/sarvam-1/tree/main), [HF cache structure](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

Install the pinned `lm_eval[hf]` only on the evaluation host, reusing its compatible PyTorch environment. Pin HF model/data revisions and use one experiment-specific `HF_HOME` and `LM_HARNESS_CACHE_PATH`; download only needed configs, tokenizer and safetensor shards. The newer CLI supports `lm-eval run`, `--samples` for explicit per-task indices, `--log_samples`, and separate model-response caches. `--limit` is documented as a testing convenience; it is not our registered sampler. Set `--num_fewshot 0`, `--seed 20260922`, and use manifest-selected indices with no chat template for the base suite. [Current CLI documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md).

Use pinned local task wrappers to load only required benchmark splits where stock loaders fetch unnecessary train data, preserving the exact prompts and scoring functions. Verify selected row IDs and task outputs against stock definitions. Fetch dataset file metadata first, maintain a 1 GiB evaluation-data/cache ceiling, and stream/select remotely if a loader would exceed it. Do not assume a small final sample makes the upstream download small. At completion, copy compact scores, manifests, logs and the candidate artifact only after checking local free space; then terminate funded resources according to the parent's budget controller.

## Open and reproducible release gate

The release should provide weights, tokenizer, architecture/inference and training code, pinned environment, full training-data recipe and source revisions, sampling/deduplication rules, seeds, actual compute/cost ledger, evaluation selection code, benchmark hashes, scores and failed gates. Include a runnable adaptation example and a load/inference smoke check. Do not redistribute third-party benchmark text or baseline weights beyond their terms; reproducible download instructions and hashes can be sufficient for evaluation.

Audit original training-data terms, not just a mirror's metadata. In particular, NVIDIA's Nemotron-ClimbMix is marked CC BY-NC 4.0 and research/development use, while a shuffled community copy displays MIT. That discrepancy is unresolved by the mirror label; do not present such a recipe as unconditionally open for every downstream use. Prefer training sources whose verified terms meet the intended fully open release, and record any remaining limits. [NVIDIA original](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), [community shuffle metadata](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle).

The user has requested an open release of this new project. Publish only the new project's audited artifacts, preserving third-party terms and excluding credentials and unrelated prior private work. There is no evaluation outcome yet. A small efficient English model is a reasonable experiment under this budget; broad superiority over Sarvam-1 or the measured small-model frontier is currently unsupported.
