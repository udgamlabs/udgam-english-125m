# Udgam English 125M — an open small-model experiment

**Ready to run locally.** Both model variants and their training history have been verified. The performance target failed; this is an experimental package. It has not been published to a public repository. Verify the distribution checksums before use.

Udgam Labs is an India-based lab building for a global audience. We trained a 125,829,314-parameter English model from random weights with our own 16,384-token tokenizer. The experiment completed 2.03 billion pretraining targets and one finite conversation-tuning pass.

**The performance target failed.** On the fixed 2,500-question suite, our base scored 45.72% and conversation model 46.40%, compared with 68.08% for Sarvam-1, 51.76% for SmolLM2-135M and 57.40% for SmolLM2-360M. We did not establish a Pareto-frontier result or a reliable general assistant. Both models failed the strict instruction diagnostics. We retain the negative results so others can examine the recipe and improve on it.

Read the [model card](MODEL_CARD.md), [full results](research/EXPERIMENT_RESULTS.md), [training recipe](research/REPRODUCE_ENGLISH_125M.md), and [dated cost summary](PUBLIC_COST_SUMMARY.json). Observed gross cloud usage was about $16.58 in the latest dated account check, including the final backup compression; Modal paid charges were $0. At the latest check, compute had stopped; the cloud backup and temporary access remained in place pending the owner's cleanup approval. Storage/transfer settlement is still pending. This excludes the value of the existing Mac, internet and engineering time.

## Run the native models

Both exact native models are present and have passed relocated CPU loading checks. Distribution integrity is established separately by the release manifest and checksums described below. Use Python 3.11 or 3.12 in a virtual environment. Install the model's listed runtime dependencies once:

```sh
python -m pip install -r models/base/requirements.txt
python models/base/infer.py --info
python models/base/infer.py --device cpu --prompt "A useful way to check an explanation is" --max-new-tokens 64
python models/conversation/infer.py --device cpu --chat --max-new-tokens 64
```

The distribution integrity files are `RELEASE_MANIFEST.json` and `SHA256SUMS`; verify that both are present before use. Before installing dependencies or running code, verify the unchanged download from its root with `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux. Compare the release-manifest hash against the separately supplied release hash; a checksum file inside the same download alone does not authenticate its publisher. Each native loader additionally verifies its own model inventory before loading weights.

Inference is local and makes no model, kernel or API requests. Base is a text-completion model; conversation uses the native role markers. The interface works independently of the original project, datasets and training optimizer. Output can be wrong, repetitive or fail instructions. The native format is PyTorch; it is not a Transformers, GGUF or quantized package. Portable inference uses SDPA, so the GPU FA3 microbenchmark in our report is not its measured speed.

## Adapt it

The [local fine-tuning guide](research/FINETUNE_NATIVE.md) accepts your own JSONL prompt/response pairs and creates a separate native artifact. Its example passed tiny CPU fixtures and one update on the actual 125M base, followed by a fresh CPU reload. That synthetic integration model remains private and was not scored. Full-pass resource requirements and CUDA adaptation have not been measured. Adapted weights acquire new hashes and do not inherit our benchmark results.

The acquisition, tokenizer, numerical training and evaluation source is included under `scripts/`, with exact upstream source and notices under `vendor/`. The historical full experiment is documented, but its guarded provider launchers are not a one-command training service: reproducing it requires the pinned public inputs, prepared provenance, hardware, dependencies and a real bounded provider lifecycle. Do not fabricate historical guard evidence. Standalone inference and the additive fine-tuning tool do not require those cloud records.

## What is open

Our code, documentation, tokenizer and owned model weights use MIT terms, with upstream notices retained. See [license scope](LICENSE_SCOPE.md) and [source attribution](research/data_sources.json). Public source text includes disclosed synthetic teacher output; no pretrained weights, embeddings, tokenizer vocabulary or teacher logits initialized the candidate.

The package excludes raw third-party corpora, comparison-model weights, optimizer backups, credentials and account records. Their source revisions and acquisition methods are documented; our license does not relicense third-party data. Public scientific summaries are distinct from the original private operational proof. Exact hashes identify which original evidence they summarize. The unchanged conversation attribution notice describes the original preparation artifact, where per-example source traces were present; this distribution includes their manifest hashes and acquisition metadata, but keeps those trace files and tokenized training examples in the private backup.
