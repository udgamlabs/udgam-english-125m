# Udgam English 125M — experiment model card

**Status: training, evaluation and full local scientific verification complete; performance target missed.** This card describes a locally verified experiment. This package has not been published publicly. Last updated 24 September 2026.

Udgam Labs is an India-based lab building for a global audience. This experiment measures how useful a small English language model can be within a US$99 attributable cost ceiling. It did not meet its general-quality floors or beat Sarvam-1. No Pareto-frontier result was established.

## Model and initialization

| Item | Recorded configuration |
|---|---|
| Stored parameters | 125,829,314, including embeddings and auxiliary parameters |
| Architecture | Pinned nanochat GPT; 12 layers, width 768, 6 query heads, 2 key/value heads, SSSL attention pattern |
| Context | 2,048 tokens |
| Tokenizer | Owned 16,384-token BPE fitted on training material |
| Initialization | Random model weights; seed 20260922 |
| Pretraining completed | 33,554,432 pilot targets plus 1,999,994,880 main targets |
| Main endpoint | 61,035 updates; original optimizer, random state and finite data cursor retained across continuations |
| Conversation stage completed | One assistant-masked pass over 7,237 prepared OASST2 examples; 1,810 updates; 1,892,764 supervised assistant targets |
| Intended license | MIT for owned code, tokenizer and weights; third-party notices and terms retained |

No external pretrained weights, embeddings, tokenizer vocabulary, adapters, teacher logits or hidden states initialize the candidate. Continuing its own checkpoints preserves the random-initialization lineage. The architecture and training implementation reuse upstream open-source work, credited in the reproduction guide and vendor notices.

The completed main checkpoint has been downloaded, matched to its committed cloud SHA256, and loaded on CPU using restricted tensor deserialization. All model parameters are finite. This integrity check does not measure model quality or prove a completed release.

The exact base inference export has also passed a fresh-process load from a separate folder on this Mac and generated a bounded completion without network access. Every exported tensor matched the selected main checkpoint byte for byte. This local usability check used PyTorch 2.11 on CPU; the recorded cloud training/evaluation runtime used PyTorch 2.9.1 on CUDA. The filesystem cache was warmed by verification, so its timing is not a cold-disk latency benchmark.

The completed conversation checkpoint and native export are now local and hash-verified. Every exported tensor matches its checkpoint byte for byte; all model, optimizer and random-state tensors are finite, and the finite training counts and completed-main parent agree. The export also runs from a separate folder in a fresh CPU process without cloud access. The complete sealed input and returned artifact chain have passed the private portable provenance validator. Its sole floating-point compatibility exception permits at most four binary64 ULP when recomputing positive finite historical cosine learning rates. Stored training records, the optimizer endpoint, all other checks and bytewise checkpoint/export tensor equality remain exact. This is portable validation, not a replay of the original Linux math implementation.

## Data and method

The main mixture targets 60% FineWeb-Edu, 15% FinePhrase tables, 15% FinePhrase tutorials and 10% Nemotron fact-seeking ordinary tokens. FinePhrase and Nemotron include synthetic text; the source records disclose their teacher models. Random-weight training is not a claim that all text is human-written.

The recipe groups source families, rejects normalized duplicates and filters known validation/benchmark overlaps. During preparation, a captured benchmark-family overlap in the untrained suffix was removed. The repair preserved the 36 previously accepted shards and the saved checkpoint, changing only the future data contract. The exact repair plan, receipt and checkpoint migration proof are retained. This incident-specific check does not establish universal absence of contamination or paraphrased overlap.

The [reproduction guide](research/REPRODUCE_ENGLISH_125M.md) records source revisions, tokenizer fitting, optimizer settings, numerical schedule, software pins, hardware, and known limits. [Source records](research/data_sources.json) and [license scope](LICENSE_SCOPE.md) distinguish the open recipe from rights to redistribute underlying third-party text.

## Evaluation status

The completed main run's frozen validation loss is 2.587688 nats per prediction target. Validation loss is a training diagnostic; it is not evidence of benchmark superiority or a reliable assistant.

All 12 cloud evaluation jobs have completed and the worker is independently confirmed stopped. All evaluation artifacts, including both native weight exports, are locally hash-verified, and the fixed comparisons were independently recomputed. The complete sealed training input and its returned artifact lineage are locally verified. The frozen comparison includes the owned base and conversation models, Sarvam-1, SmolLM2-135M and SmolLM2-360M, with 500 items each from ARC-Challenge, HellaSwag, PIQA, BoolQ and WinoGrande. It requires all five models and their common latency measurements, preserves failed floors and per-item outcomes, and uses paired uncertainty estimates. The [protocol](research/evaluation_protocol.md) defines acceptance criteria before final scores are observed.

On the fixed 2,500-item suite, macro accuracy was **45.72% for the base and 46.40% for the conversation model**, versus 68.08% for Sarvam-1, 51.76% for SmolLM2-135M and 57.40% for SmolLM2-360M. Both owned models failed all five task floors and the macro floor. The base/SFT cross-candidate comparisons are exploratory; all six unadjusted pairwise intervals and failed gates are retained in the [results report](research/EXPERIMENT_RESULTS.md).

The separate 36 native instruction checks are limited diagnostics, not a full instruction-following benchmark. Both models passed 0/24 strict automatic checks; the 12 human-rubric items per model remain unscored. Some responses contained relevant content but failed exact formatting. An independent audit found no demonstrated template or loss-mask bug.

## Intended use and limits

The current artifact is suitable for research into small-model training, transparent cost accounting and reproducible evaluation. A base pretraining checkpoint is a text-completion model; chat formatting alone does not make it a competent assistant. The measured quality and strict instruction-check results do not support use as a reliable general assistant.

English is the target language. Multilingual quality, long-context behavior, factual reliability, coding ability, deployment security, quantization, mobile performance and suitability for high-stakes decisions have not been established. Generated text can be incorrect, biased or inappropriate. No tools, search service or external model is part of native inference.

The intended standalone format is native PyTorch with the owned tokenizer, pinned vendor code and offline inference entrypoint. It is not currently a Transformers, GGUF or quantized release. A verified standalone package and tested commands will accompany release completion.

## Costs and availability

The cost ceiling is US$99 attributable value, including failed attempts, preparation, setup, evaluation, storage and transfer. Modal's operating limits are US$25 gross workspace consumption and US$0 paid charges. Free credits are reported separately from compute value; zero new cash is not zero training cost. At compute close, observed Modal gross usage plus prior Runpod consumption was about $16.45; Modal paid charges were $0. A later dated account check recorded about $16.58 including backup compression, with Modal paid charges still $0. The dated public cost summary preserves both observations. Final attributable cost and storage/transfer settlement remain unresolved, and another lab's reproduction cost has not been measured. Existing Mac, internet and engineering time are unpriced.

Open publication is intended to include owned base and conversation weights, tokenizer, code, preparation recipe, provenance, evaluation outputs where permitted, and the complete successful/failed cost record. Comparison-model weights, account credentials and unrelated workspace material are excluded. No public model release is claimed by this card.
