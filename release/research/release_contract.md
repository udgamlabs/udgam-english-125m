# Open and reproducible release requirements

User scope, 2026-09-22: a globally useful small English model developed by
India-based Udgam Labs. Random model weights and a tokenizer fitted only to the
training split. Open weights, code, method, and cost evidence so others can train,
fine-tune, modify, and use the result.

## Release contents

- Final weights and tokenizer, hashes, architecture configuration, parameter
  count including embeddings and lookup tables, and tested inference command.
- Original project code, documentation, owned tokenizer and owned model weights
  under MIT when released, as established in `LICENSE_SCOPE.md`, with upstream
  licenses and notices preserved. Third-party data and baseline weights keep
  their own terms. A completed final model release does not yet exist.
- Exact source revisions, dependency lock, hardware and region, random seeds,
  initialization evidence, training and post-training commands, data mixing and
  processing code, checkpoint/save/resume behavior, and known reproduction limits.
- Dataset source revisions, document/source grouping, filtering and exclusions,
  training-only tokenizer fitting, teacher provenance for synthetic text,
  attribution and license inventory. Redistribute raw text only where terms allow.
- Complete successful and failed run ledger, setup, idle, evaluation, storage,
  transfer, and fees. Distinguish already owned hardware and prepaid credits from
  new cash investment; show consumed compute value even when prepaid.
- Frozen evaluation definitions and per-item outputs where redistribution permits,
  model revisions and prompts, confidence intervals, quality by task, stored size,
  measured memory and latency on common hardware, and examples of failures.
- A short model card: expected uses, limitations, no unmeasured Sarvam victory or
  worldwide Pareto-frontier claim. Synthetic teacher data is not pretrained weight
  transfer, but must remain disclosed.

## Honest boundaries

An open database license does not automatically license every underlying webpage.
The release must state whether others can download the exact data or reproduce
its acquisition and filtering, and identify any unavailable sources. Do not
describe a recipe as a fully redistributable corpus. If full raw-corpus openness
is feasible within the quality and cost target, prefer it. The existing research
on Common Pile and other explicitly licensed corpora will inform this choice.

One short run cannot establish a global Pareto frontier. Report the measured
comparison set and tradeoffs. Report complete stored parameters and resident
memory, not only active parameters. Preserve all results, including regressions.
