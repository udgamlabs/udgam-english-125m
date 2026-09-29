# Reproducing the Udgam English 125M experiment

Research snapshot: **22 September 2026**. Guide updated 24 September 2026.

Pretraining, conversation training and all 12 evaluation jobs are complete for
this small English model by India-based Udgam Labs. **The performance target
failed.** On the fixed 2,500-item suite, the base scored **45.72%** and the
conversation model **46.40%**, versus **68.08% for Sarvam-1, 51.76% for
SmolLM2-135M and 57.40% for SmolLM2-360M**. Both owned models failed all task
floors. See [the completed experiment results](EXPERIMENT_RESULTS.md) for scores,
uncertainty and limits. The complete input backup, native artifact/tensor checks
and private portable provenance assembly are locally verified. No public
publication has occurred; package integrity is checked separately through its
manifest and checksums. This is a documented negative experiment, not a reliable
assistant or a benchmark victory. Server geography is unrestricted.

The candidate's **weights and tokenizer start from scratch**. No pretrained
model weights, tokenizer vocabulary, teacher logits or hidden states initialize
it. Some public training text is synthetic and has named teacher models: this
is not a claim of teacher-free data or an independently invented architecture.

## The recipe

| Part | Fixed choice |
|---|---|
| Architecture | nanochat GPT; 125,829,314 stored parameters; 12 layers, width 768, 6 query heads, 2 KV heads; `SSSL` attention pattern |
| Context and tokenizer | 2,048 tokens; owned 16,384-token Rust BPE, fitted on training text only |
| Code | nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` |
| GPU runtime | One H100, BF16 compute; PyTorch 2.9.1 / CUDA 12.8; optional model compilation enabled in the recorded run |
| Training batch | 16 rows ×2,048 targets = 32,768 targets/update; accumulation 1 |
| Random seed | `20260922` |
| Initial pilot | 1,024 updates; 33,554,432 prediction targets |
| Main pretraining | Fixed 61,035 updates; 1,999,994,880 targets, rounding the declared 2B horizon down by5,120 |
| Conversation stage | Completed one finite assistant-masked OASST2 pass: 7,237 examples, 1,810 updates, 1,892,764 supervised targets |

The exact model configuration is [nanochat_dense_reference.json](../configs/nanochat_dense_reference.json).
Its historical `reference_configuration_only` label is not the current run status;
use checkpoint/run records for completed work. The pinned architecture contains
more than these headline dimensions; reproduce its source rather than replacing
it with a similarly sized generic Transformer.

Pretraining uses the pinned upstream MuonAdamW implementation. At this batch,
the learning-rate scale is 0.25: matrix 0.005, embedding 0.075,
unembedding 0.002 and scalar 0.125. The recorded initial Muon decay is
0.07750021661995488. There are 40 warmup updates, linear decay over the last 65%
of the horizon to 5% of the initial learning rate, and the source's momentum and
cosine weight-decay schedules. See [the optimizer and step implementation](../scripts/train_dense.py)
and [public scientific provenance summary](../PUBLIC_SCIENTIFIC_PROVENANCE.json) (the original the recorded main contract remains private audit evidence).

## Public text and tokenizer lineage

The main mixture targets 60% FineWeb-Edu, 15% FinePhrase tables, 15% FinePhrase
tutorials and 10% Nemotron fact-seeking **ordinary tokens**. Actual counts and
finite tails come from the stage manifest; these percentages are not exact
document proportions. FAQ was not included in the prepared main recipe.

| Source | Pinned revision | Terms and provenance |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0; upstream web/Common Crawl terms remain |
| [FinePhrase](https://huggingface.co/datasets/HuggingFaceFW/finephrase), table/tutorial | `78cf4a5ed0099214979c094c963e699c19163838` | ODC-By 1.0; SmolLM2-1.7B-Instruct rewrites; use `rollout_results[0].text`, not the original top-level text |
| [Nemotron-Pretraining-Specialized-v1.2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2), Fact-Seeking only | `807afc1fa65c441d46ebc7d9b95295a35499a527` | CC-BY 4.0; Qwen3-30B-A3B-Instruct-2507 teacher |
| [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2), conversation stage | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | Apache 2.0; source labels and quality filters do not independently prove authorship or correctness |

The acquisition recipe preserves file/row identities, licenses and teacher
provenance. It groups related source documents, rejects normalized duplicates,
and excludes frozen validation and benchmark overlaps, including 13-word matches.
Main acquisition also excludes pilot source families, including unused pilot
rows. Unknown FineWiki identities and paraphrases remain limitations; this is
not a guarantee of zero contamination. Noncommercial ClimbMix is excluded.
Details: [source lock](data_sources.json), [public scientific provenance summary](../PUBLIC_SCIENTIFIC_PROVENANCE.json) (the original main data contract remains private audit evidence),
[public scientific provenance summary](../PUBLIC_SCIENTIFIC_PROVENANCE.json) (the original bounded v3 preparation remains private audit evidence).

The completed preparation contains 2,006,243,328 prediction targets across 217
shards, with 1,999,998,003 accepted ordinary tokens. Packed targets and ordinary
source tokens are different accounting units. This finite supply exceeds the
fixed main-training horizon; it does not extend that horizon.

During preparation, a captured benchmark-family overlap was found in the
untrained suffix. The recorded repair excluded that family and reconstructed
the suffix while preserving all 36 shards accepted by the saved main checkpoint.
The completed continuation migrated only the future-data contract in memory; it
preserved the original checkpoint file, numerical state, consumed cursor and schedule.
This incident does not establish universal absence of benchmark overlap.
The completed private provenance bundle retains the exact repair plan, receipt
and migration proof. The [public scientific provenance summary](../PUBLIC_SCIENTIFIC_PROVENANCE.json)
records their hashes and the original/completed checkpoint and stage-contract
identities; it does not publish the private operational evidence wholesale.

The recorded BPE fit used 166,412 training documents / 176,701,780 characters.
The owned JSON tokenizer SHA256 is
`1967afcf40ffc3bf110ab965dab4be7436ed2d9f7e6c77efbb1aff9d8221e034`.
Keep it fixed through main training, SFT and inference. Packed arrays are finite
`uint16` rows of 2,049 tokens: inputs are the first 2,048 and targets the last 2,048.
Do not wrap a small corpus, refill it by repetition, or fit on holdout text.

## Reproduction entrypoints

**Two different levels of reproducibility:** a verified native export can run
offline and be adapted with the [standalone JSONL tool](FINETUNE_NATIVE.md).
The historical full experiment is not a turnkey rerun from the source bundle:
it also needs the pinned data downloads, exact preparation/provenance state and
a real reviewed provider lifecycle. Links below into `runs/` and `records/` name
original private audit evidence, which is not copied wholesale into a public
release. Public scientific summaries must instead identify the relevant hashes,
counts and contract choices explicitly. No example command manufactures provider
evidence or bypasses a spending guard.

Commands below run from `scratch_20260922`. Use new output directories. Download
public data using the pinned recipe before renting a GPU. Preserve whole stopped
preparation state, including its SQLite source cursors/provenance and immutable
shards; token arrays alone cannot resume acquisition faithfully.

On a compatible GPU host, create an isolated Python 3.12 environment and install:

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r configs/gpu-requirements.txt
python -m pip check
```

The source pins `kernels==0.12.2` and FlashAttention 3 revision
`de87b9b5af06dd9984df595bef90b2eba44b181a`. The runtime injects that revision
explicitly; an unpinned kernel download or SDPA fallback is not the recorded
training setup. The complete dependency list is in
[gpu-requirements.txt](../configs/gpu-requirements.txt); evaluation adds
[baseline-eval-requirements.txt](../configs/baseline-eval-requirements.txt).

With the verified pilot corpus and its source report present, fit and pack anew:

```sh
python scripts/prepare_dense_tokens.py \
  --corpus data/english-production-001 \
  --source-report records/production-data-002.json \
  --output data/repro-tokens --vocab-size 16384 --sequence-len 2048
```

Start random-weight training with **no `--resume`**:

```sh
python scripts/train_dense.py \
  --config configs/nanochat_dense_reference.json --data data/repro-tokens \
  --output runs/repro-pilot --guard-evidence "$PILOT_GUARD" \
  --max-steps 1024 --batch-rows 16 --compute-seconds 525 --compile-model
```

`PILOT_GUARD` denotes fresh evidence from a real reviewed provider lifecycle,
not a sample JSON file. This historical entrypoint validates the pilot Runpod
profile. Do not fabricate that evidence for Modal or remove the guard. The
current Modal continuation uses its own contract and launcher; the provider
boundary is distinct from the numerical training recipe.

| Next step | Actual entrypoint and contract |
|---|---|
| Acquire/pack finite main data | [prepare_main.py](../scripts/prepare_main.py), with the frozen [v3 transport](../scripts/acquire_main_scanned.py) and bounded packer; follow the data contract above |
| Start main from the owned pilot | [train_main.py](../scripts/train_main.py), `--pilot-checkpoint`; fresh MuonAdamW once because the short pilot already annealed |
| Continue main exactly | `--resume-main`; restore weights, optimizer, RNG, consumed-row cursor and the original fixed schedule; [public scientific provenance summary](../PUBLIC_SCIENTIFIC_PROVENANCE.json) (the original Modal adapter/launch review remains private audit evidence) |
| Prepare conversation examples | [prepare_sft.py](../scripts/prepare_sft.py), pinned source receipt, owned tokenizer, whole-tree splits and benchmark/validation exclusions |
| Train one conversation pass | [train_sft_modal.py](../scripts/train_sft_modal.py); completed main parent required; [public scientific provenance summary](../PUBLIC_SCIENTIFIC_PROVENANCE.json) (the original commands and contract remains private audit evidence) |
| Final scoring/export | [prepare_final_jobs.py](../scripts/prepare_final_jobs.py) → [evaluate_modal.py](../scripts/evaluate_modal.py); real selected checkpoint hashes required |

An owned-checkpoint continuation is still part of this scratch lineage, but is
not another random initialization. Never reset the main optimizer between
sessions, extend its horizon silently, or use a public pretrained checkpoint as
the parent. Exact-resume tests use tiny CPU fixtures; identical floating-point
results across different GPUs/frameworks are not promised.

The prepared SFT training split contains 7,237 examples, 2,888,369 processed
positions and 1,892,764 supervised assistant targets. Batch 4 gives 1,810 updates.
Each assistant target appears once; earlier turns may recur as masked context.
Loss covers only assistant content plus its end token. No truncation or
quota-filling repeats. Fresh AdamW uses 1e-4→1e-5, 3% warmup, betas 0.9/0.95,
weight decay 0.01; the owned main parent and random-initialization hash survive.

## One frozen comparison, then interpretation

The [evaluation protocol](evaluation_protocol.md) fixes 2,500 items: 500 each
ARC-Challenge, HellaSwag, PIQA, BoolQ and WinoGrande. The first three use
`acc_norm`, the last two `acc`; Q is their unweighted mean. Task definitions
are pinned to harness commit `d6de81643928d653435c431bae19945d41d32520`.
Use identical zero-shot completion strings, each model's tokenizer, a 2,048-token
context, no chat template/tools and no answer repair. Unsupported examples count
incorrect rather than disappearing. Preserve per-item likelihoods and outcomes.

Evaluate our completed base and SFT plus **all three** pinned baselines:
Sarvam-1, SmolLM2-135M and SmolLM2-360M. First finish 100 disjoint development
items per model. The immutable plan estimates
`2 × (fixed loading/verification + 25 × scoring/bookkeeping) + 120s latency`
per final model, then checks the whole suite fits. Do not shrink or cherry-pick
after scores. Both owned exports and all five common latency measurements remain
separate from quality scoring. Forced 128-token latency is a microbenchmark, not
observed user-response latency.

The completed comparison and all six fixed pairwise records are retained in
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md). Cross-candidate interpretation is
exploratory; pairwise intervals are unadjusted and no winner was selected from
final scores. Both owned models also scored 0/24 on the strict automatic native
instruction checks, with all 12 human-rubric items per model unscored. One correct
colour expressed as a sentence still failed the requested single-word format;
other answers contain substantive errors and repetition. No general instruction
competence is established.

For reproduction, report every failed floor, paired bootstrap intervals, actual artifact bytes,
memory, latency and cost. The required baseline weights are private research
inputs, excluded from our release; Sarvam's research license is not MIT.
The 36 final native instruction diagnostics are limited tests, not a replacement
for the protocol's separate IFEval/instruction-baseline track, which is not yet
established. Chat weakness in an unadapted Sarvam base model is not a win.

## Run an eventual native export

After verified export, inside that standalone package:

```sh
python -m pip install -r requirements.txt
python infer.py --info
python infer.py --device cpu --prompt "A useful way to check an explanation is" --max-new-tokens 64
python infer.py --chat --max-new-tokens 80
```

The package includes owned tensors/tokenizer, pinned vendor code, notices and
`infer.py`; it needs no API key, dataset or optimizer checkpoint. Runtime uses
offline SDPA with a KV cache. Base exports should be assessed as completions;
chat formatting alone does not confer instruction ability. See
[native inference instructions](../configs/inference/README.md). This is a native
PyTorch format, not Transformers/GGUF, a quantized artifact, or a measured mobile
deployment. Preserve base and SFT exports separately.

## Cost and completion claims

The experiment ceiling is **$99 attributable cost**. Record full-price compute,
CPU data preparation, GPU setup/compilation, storage, transfer and failed attempts
even when free credits pay the bill. Separately report gross service usage,
credits consumed and new cash paid. **$0 cash does not mean $0 training cost**,
and another lab may not receive the same promotions.

The current bounded Modal plan uses a $25 gross workspace ceiling and $0 paid
ceiling. The latest additive recovery allocation is $7.75 for preparation, $7.90 for main
training, $4 for SFT and $4.35 for evaluation. Unused allowance from the stopped
main run was reassigned to conversation training; the $25/$0/$99 limits did not
increase. The failed first SFT attempt remains attributable, so this does not
reset its consumed budget. The evaluation allocation includes
the observed $0.01022273 CPU baseline preparation charge. Unexplained billing
differences remain reserved rather than counted as newly available credit.
See [public cost summary](../PUBLIC_COST_SUMMARY.json) (the original the original finite overhead policy remains private audit evidence)
and [public cost summary](../PUBLIC_COST_SUMMARY.json) (the original the additive SFT recovery allocation remains private audit evidence).
These are limits, not observed total costs or guarantees that all work will fit.
The H100 + 4 CPU + 16 GiB accounting floor used by
the execution contracts is $0.00118492/second; consult actual dated provider
observations and settled bills for a new run. A Python exit is not proof of
provider shutdown. Use real lifecycle deadlines, stopped-resource verification
and hash-verified backups before deleting remote state.

By the completed 24 September 2026 evaluation, both training stages had
completed. All 34 main-return files are locally checksum-verified. The main
checkpoint was safely reloaded on CPU and all 125,829,314 parameters were finite;
its exact 61,035-step horizon, finite cursor and recorded initialization lineage
match the returned metadata. This is integrity evidence, not a quality score.

The completed input seal contains 1,033 files / 23,869,913,206 bytes, SHA256
`a2ab933da89a9ff4d9d8daff16818a0c2b3c6bdd8560d802ae1edc5e9cde16d1`.
The complete local backup has passed the original strict sealed-input inventory.
The recovered SQLite matches its original full-file SHA256, and every other
input file matches the sealed manifest. Resumable transfer progress alone was
never treated as a complete backup or release-verification claim.

The completed main checkpoint SHA256 is
`73680732c4d2b5ce9f7335e3cfed69f4968b020a0fc456fa2a8becbf396b276d`;
its delta has 34 files / 1,994,821,004 bytes, SHA256
`e82ceb579531141f5f593ee73931fbeb8efca5b7ef73e6bc1f489b888e75f902`.
The main GPU is independently confirmed stopped.

The first conversation attempt stopped before its trainer began because serial
input copying was too slow. Its $0.74201848 observed cost and $0.05 hold remain
attributable. A full-package CPU trial then verified all 25.878 GB in 202.39
seconds. The successful recovery used the same numerical training recipe and
completed all 1,810 updates / 7,237 examples / 1,892,764 supervised targets.
It committed its result, verified a remote checkpoint reload, and stopped.
Its final checkpoint SHA256 is
`f993cf94458d2566d8aca55f1f5bdf5f68dacf9d76bbdb77af2be045a1ac61e8`.
The SFT delta has 19 files / 3,021,062,645 bytes, SHA256
`2c4041819055c08879c654e79df1244372cbb15007a27137df811de1a5d6df5e`.
The returned checkpoint and native export passed full file hashes, finite-pass
and parent-lineage checks, exact tensor equality and a relocated CPU load.
The complete sealed-input backup and private portable provenance proof are now
verified. The recovered function took 419 seconds
and used $0.49546921 observed credit plus a $0.05 unsettled hold.

All five development evaluations completed and the joint timing gate passed
with every model included and no quality-based selection. All 12 final-stage
jobs are now complete and the evaluation worker is independently stopped. Every
evaluation artifact, including both native models, was returned and verified;
the failed quality result and resource-measurement limitations are recorded in
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md). Complete training-input backup
and full portable provenance assembly are now locally verified. Publication has
not occurred. Distribution checksums are a separate packaging check, and the
original private proof remains distinct from its public scientific summary.
Dated billing records govern costs, not older account-balance snapshots; the
results report distinguishes observed gross usage from free credits and
unsettled final charges.

For a reproducible release, retain the exact source/data/tokenizer/config hashes,
seeds, finite cursors, training and billing logs, selected checkpoint and native
package, benchmark selection and item results, plus a clean-process load test.
The project's original code, tokenizer and owned weights are intended for MIT
release under [LICENSE_SCOPE.md](../LICENSE_SCOPE.md); upstream software/data
terms remain. No baseline weights, credentials, unrelated projects or raw
third-party corpus are implicitly redistributable. Public publication has not
occurred, and no Sarvam-superiority or general-assistant-quality claim is made.
