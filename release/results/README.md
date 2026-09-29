# Public scientific results — metadata staging

This is a results bundle, not a complete model release, full-backup proof or publication. The English 125M experiment completed training and evaluation but missed its target. Macro accuracy: base 45.72%, conversation 46.40%, Sarvam-1 68.08%, SmolLM2-135M 51.76%, SmolLM2-360M 57.40%. Both owned models failed every task floor.

## Exact and derived files

The six comparison JSONs, quality/resource JSON, five 2,500-row numeric item files and pre-scoring cluster record are exact originals. The item files contain IDs, hashes, likelihoods, predictions, correctness, token-length counts and cache-chain hashes, without benchmark text, requests, token IDs, weights or credentials. The cluster record's historical results_scored:false refers to its freeze time.

Selection, model/source identities, instruction counts and audit summary are explicitly derived with original hashes. The original selection manifest's local_file field is omitted. These derived records do not masquerade as the unchanged private originals.

The deterministic manifest classifies each file. JSON uses UTF-8, sorted keys, two-space indentation and a final newline, with no volatile timestamp. manifest.sha256 hashes manifest.json. No bootstrap was rerun, score changed, checkpoint selected, human response graded, model loaded or dataset acquired for this bundle.

## Aligning or verifying a rerun

1. Use each task's repository, immutable revision, config, split and upstream file hashes in selection/public-selection.json. Acquire permitted sources independently under their terms; no recipient acquisition is claimed here.
2. Preserve original split row order. Canonical document JSON uses sorted keys, UTF-8, no extra spaces and unescaped Unicode. SHA256 its bytes and match (task, original_split_row_index, document_sha256). Original IDs are supplemental and may be null.
3. Hash english-small-v1|20260922|task_name|original_split_row_index|canonical_document_sha256; sort ascending and take 500 distinct documents after retaining the earliest identical-document row. Exact selected order/hashes are supplied. Never redraw after observing scores.
4. Use pinned task definitions and unchanged scoring code. Match each request hash and original_requests_file_sha256 from model-identities.json; these identify formatting without bundling text. All five outcome files share identical ordered selected identities and request hashes.
5. The cluster-hash list aligns positionally with each task's selected order. HellaSwag source IDs and normalized shared BoolQ passages/WinoGrande sentences were grouped as recorded. Raw group text is absent. Canonical hashing of groups must match groups_sha256.
6. The item chain begins with canonical_evaluation_contract_sha256. For each row remove only cache_chain_sha256, append canonical row JSON to the preceding 64-character hash, then SHA256 the UTF-8 string. Verify each claimed hash. This checks recorded numerical bytes/order, not a fresh inference run or independent acquisition.

## Scope and failed gates

ARC-C/HellaSwag/PIQA use normalized accuracy; BoolQ/WinoGrande use accuracy. Macro quality weights all five 500-item tasks equally. Original paired 10,000-replicate source-cluster intervals remain unchanged; six pairwise intervals are unadjusted and cross-candidate interpretation is exploratory. No broad superiority or Pareto claim follows.

The 30 supported-prompt forced 128-token microbenchmark is not user-response latency. Owned FA3/BF16 compute differs from baseline SDPA/BF16 storage; portable native inference uses SDPA. Peak-memory capture scopes differ. Resource dominance is not established.

Both owned models scored 0/24 strict automatic instruction checks. Twelve human-rubric cases per model remain unscored. These related original templates are not IFEval. Correct content in a disallowed format can fail; many outputs also contained substantive errors and repetition.

## Ownership and upstream notice

Project MIT and license-scope notices apply to owned material; they do not relicense benchmark databases/text, upstream code or baseline weights. Original license metadata says CC-BY-SA-4.0 for ARC, other for BoolQ, and missing for HellaSwag/PIQA/WinoGrande. These are observations, not independent legal clearance. Source URLs, revisions and hashes remain attached; raw benchmark text and external weights are absent.

Task pins identify EleutherAI/lm-evaluation-harness commit `d6de81643928d653435c431bae19945d41d32520`, dated 14 September 2026. The exact upstream MIT license (Copyright 2020 EleutherAI) has now been retrieved for that revision and verified: SHA256 `a806e42547620dffbc2ec0333322c8b3523b4af11e059be14ef81aa5b1ae021f`. In the assembled public package, all 12 original hash-matching source/YAML files, original manifest, license and separate attribution notice live in [the companion task-definition directory](../research/harness-task-definitions/UPSTREAM_NOTICE.json). This results directory retains pins only; the code license does not relicense benchmark datasets. This local documentation update verified the supplied license and attribution record; it made no network request.

Private lifecycle/account records, full provenance, raw corpus, model tensors and checkpoint/export equality are outside scope. Hash references to omitted originals do not make those files public or establish a complete model release.

The earlier results manifest is preserved by its SHA256 `ee4b72aff13f97a5ecebbfd055c4dc5856ec7db52dde48ad724007595d313fb6`. This revision changes only this README and the derived task-pin license metadata; all comparison files, item outcomes and other scientific records remain byte-identical.
