#!/usr/bin/env python3
"""Offline, finite human-English SFT preparation; no downloads or model loading.

Supply a receipt for already acquired pinned OASST2 files. Each example predicts
one assistant message; previous assistant turns are context with ignored loss.
Independent variable-length examples must never be concatenated for attention.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import infer_dense as native
import prepare_dense_tokens as dense
import prepare_production as source

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'udgam-finite-human-sft-v1'
REPO = 'OpenAssistant/oasst2'
REVISION = '179dd21fc55192153d94adb0e0ce8f69e222bf75'
BENCHMARK = 'f3cbbebb1dac51e2b2fb91f4fa786dee8c0f5b1c7045b13d12fb207673d019f9'
PILOT = '1d78d58cb11d72abcb8cb8c9de58d9ae5ddf20346cbed532d89722c07ae29010'
FILES = {'train': 'data/train-00000-of-00001-88ba0162028a73fc.parquet',
         'validation': 'data/validation-00000-of-00001-1deeef95c3248fe0.parquet'}
SOURCE_COUNTS = {'train': 128575, 'validation': 6599}
sha256 = dense.sha256
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
text_hash = lambda text: hashlib.sha256(text.encode()).hexdigest()


def safe_path(directory, name):
    return native.artifact_path(Path(directory).resolve(), name)


def read_rows(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)


def inputs(receipt_path, *, tiny=False):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('repo') != REPO or receipt.get('revision') != REVISION or receipt.get('license') != 'apache-2.0':
        raise ValueError('Pinned human OASST2 source/revision/license required')
    if receipt.get('profile') != ('synthetic_test' if tiny else 'pinned_acquisition'):
        raise ValueError('Source receipt profile mismatch')
    files = receipt.get('files', [])
    if len(files) != 2 or {f['split'] for f in files} != set(FILES):
        raise ValueError('Both complete upstream train and validation files are required')
    if any(type(f.get('bytes')) is not int or f['bytes'] < 0 for f in files) or sum(f['bytes'] for f in files) > 128 * 1024 * 1024:
        raise ValueError('Pinned source-file byte bound exceeded')
    messages = {}; counts = Counter(); tree_splits = defaultdict(set)
    for entry in files:
        split = entry['split']; path = safe_path(receipt_path.parent, entry['path'])
        if not tiny and path.suffix != '.parquet': raise ValueError('Pinned production Parquet files required')
        if not tiny and (entry.get('repo_path') != FILES[split] or entry.get('observed_revision') != REVISION or
                entry.get('url') != f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILES[split]}'):
            raise ValueError('Acquisition receipt is not tied to the pinned source file')
        if sha256(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError('Source file integrity failure')
        if type(entry['rows']) is not int or not 0 <= entry['rows'] <= SOURCE_COUNTS[split]:
            raise ValueError('Source row bound exceeded')
        for row_index, row in enumerate(read_rows(path)):
            counts[split] += 1
            if counts[split] > entry['rows']: raise ValueError('Source row bound exceeded')
            mid, tid = row.get('message_id'), row.get('message_tree_id')
            if not isinstance(mid, str) or not mid or not isinstance(tid, str) or not tid:
                raise ValueError('Every source message needs stable message and tree IDs')
            if mid in messages: raise ValueError('Duplicate source message ID')
            messages[mid] = {'row': row, 'trace': {'source_split': split, 'source_file': entry['repo_path'],
                'source_file_sha256': entry['sha256'], 'row_index': row_index, 'row_sha256': text_hash(canonical(row))}}
            tree_splits[tid].add(split)
        if counts[split] != entry['rows'] or (not tiny and counts[split] != SOURCE_COUNTS[split]):
            raise ValueError('Source row count is incomplete or changed')
    if any(len(splits) != 1 for splits in tree_splits.values()):
        raise ValueError('An upstream tree crosses train/validation source files')
    return receipt, messages


def heldout_deny(pilot_data, pilot_corpus, benchmarks, *, tiny=False):
    pilot_data, pilot_corpus, benchmarks = map(Path, (pilot_data, pilot_corpus, benchmarks))
    manifest_path = pilot_data / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    pilot_hash = sha256(manifest_path)
    if pilot_hash != (pilot_data / 'manifest.sha256').read_text().strip() or (not tiny and pilot_hash != PILOT):
        raise ValueError('Frozen owned pilot manifest mismatch')
    if manifest.get('status') != 'ready' or (not tiny and manifest.get('profile') != 'production'):
        raise ValueError('Owned pilot data is incomplete')
    required = ('tokenizer.json', 'source-report.json', 'val.npy', 'source-val-provenance.jsonl.gz')
    for name in required:
        if sha256(pilot_data / name) != manifest['hashes'][name]: raise ValueError('Owned pilot artifact mismatch: ' + name)
    report = json.loads((pilot_data / 'source-report.json').read_text())
    val = pilot_corpus / 'val/shard_00000.parquet'
    if sha256(val) != report['hashes']['val/shard_00000.parquet']:
        raise ValueError('Frozen pilot validation source mismatch')
    bm = json.loads((benchmarks / 'manifest.json').read_text()); bm_hash = sha256(benchmarks / 'manifest.json')
    if bm_hash != report['benchmark_manifest_sha256'] or (not tiny and bm_hash != BENCHMARK):
        raise ValueError('Frozen benchmark exclusion contract differs')
    texts = []
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(val).iter_batches(columns=['text'], batch_size=256):
        texts.extend(batch.column(0).to_pylist())
    files = {}
    for task, info in bm['tasks'].items():
        path = benchmarks / Path(info['local_file']).name
        if sha256(path) != info['sha256']: raise ValueError('Benchmark exclusion source mismatch')
        files[path.name] = info['sha256']
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f: texts.extend(source.benchmark_fields(task, json.loads(line)))
    return source.Deny(texts), manifest, {'pilot_manifest_sha256': pilot_hash,
        'pilot_tokenizer_sha256': manifest['hashes']['tokenizer.json'],
        'pilot_validation_array_sha256': manifest['hashes']['val.npy'], 'pilot_validation_source_sha256': sha256(val),
        'pilot_source_report_sha256': manifest['hashes']['source-report.json'],
        'benchmark_manifest_sha256': bm_hash, 'benchmark_exclusion_file_sha256': files}


def eligible(row):
    return (row.get('role') in ('prompter', 'assistant') and row.get('lang') == 'en' and
        row.get('review_result') is True and type(row.get('review_count')) is int and row['review_count'] >= 1 and
        row.get('deleted') is False and row.get('synthetic') is False and row.get('model_name') in (None, '') and
        row.get('tree_state') == 'ready_for_export' and isinstance(row.get('text'), str) and bool(row['text'].strip()) and
        (row['role'] != 'assistant' or type(row.get('rank')) is int and row['rank'] == 0))


def split_for(tree_id, upstream_split, seed):
    if upstream_split == 'validation': return 'final'
    value = int(text_hash(f'{seed}:oasst2-dev:{tree_id}')[:16], 16)
    return 'dev' if value % 100 < 2 else 'train'


def path_to(mid, messages):
    result = []; seen = set(); tid = messages[mid]['row']['message_tree_id']
    while mid is not None:
        if mid in seen or mid not in messages: raise ValueError('Missing parent or cyclic tree')
        seen.add(mid); row = messages[mid]['row']
        if row['message_tree_id'] != tid: raise ValueError('Parent belongs to another tree')
        result.append(mid); mid = row.get('parent_id')
    result.reverse()
    if any(messages[m]['row']['role'] != ('prompter' if i % 2 == 0 else 'assistant') for i, m in enumerate(result)):
        raise ValueError('Malformed role alternation')
    return result


def render(path, messages, tokenizer):
    rows = [messages[mid]['row'] for mid in path]
    if rows[-1]['role'] != 'assistant': raise ValueError('Assistant target required')
    conversation = [{'role': 'user' if r['role'] == 'prompter' else 'assistant', 'content': r['text']} for r in rows[:-1]]
    prefix = tokenizer.conversation(conversation)
    target = tokenizer.encode(rows[-1]['text']) + [tokenizer.specials['<|assistant_end|>']]
    return prefix + target, [False] * len(prefix) + [True] * len(target)


def select(messages, tokenizer, deny, *, sequence_len=2048, cap=5_000_000, seed=20260922):
    """Conservative tree exclusions; no target repetition or response truncation."""
    if type(cap) is not int or not 1 <= cap <= 5_000_000 or not 2 <= sequence_len <= 2048:
        raise ValueError('Invalid finite SFT token/context cap')
    trees = defaultdict(list); stats = Counter(); tree_meta = {}; candidates = defaultdict(list)
    for mid, entry in messages.items(): trees[entry['row']['message_tree_id']].append(mid)
    for tid, mids in trees.items():
        splits = {messages[m]['trace']['source_split'] for m in mids}
        if len(splits) != 1: raise ValueError('Tree source split mismatch')
        split = split_for(tid, next(iter(splits)), seed)
        tree_meta[tid] = {'split': split, 'reason': None}
        try:
            roots = [m for m in mids if messages[m]['row'].get('parent_id') is None]
            if len(roots) != 1: raise ValueError('Tree must have one root')
            paths = {m: path_to(m, messages) for m in mids}
        except ValueError:
            tree_meta[tid]['reason'] = 'malformed_tree'; stats['malformed_tree'] += 1; continue
        if any(deny.matches(messages[m]['row'].get('text', '')) for m in mids):
            tree_meta[tid]['reason'] = 'heldout_overlap'; stats['heldout_overlap_trees'] += 1; continue
        for mid in mids:
            row = messages[mid]['row']; path = paths[mid]
            if row.get('role') != 'assistant': continue
            if not all(eligible(messages[m]['row']) for m in path): stats['filtered_assistant_paths'] += 1; continue
            ids, mask = render(path, messages, tokenizer)
            if len(ids) > sequence_len: stats['over_context_targets'] += 1; continue
            candidates[tid].append({'path': path, 'target': mid, 'ids': ids, 'mask': mask,
                                    'normalized_target_sha256': dense.normalized_hash(row['text'])})
    # Preserve final/dev trees first. Any exact-normalized message crossing split
    # boundaries removes the lower-priority entire tree, including its siblings.
    message_split = {}; target_hashes = set(); selected = {'train': [], 'dev': [], 'final': []}; total = 0
    order = sorted(trees, key=lambda t: ({'final': 0, 'dev': 1, 'train': 2}[tree_meta[t]['split']], text_hash(f'{seed}:tree:{t}')))
    for tid in order:
        meta = tree_meta[tid]; split = meta['split']
        if meta['reason']: continue
        if not candidates[tid]: meta['reason'] = 'no_eligible_target'; continue
        hashes = {dense.normalized_hash(messages[m]['row']['text']) for c in candidates[tid] for m in c['path']}
        if any(h in message_split and message_split[h] != split for h in hashes):
            meta['reason'] = 'cross_split_normalized_message'; stats['cross_split_duplicate_trees'] += 1; continue
        admitted = False
        for item in sorted(candidates[tid], key=lambda x: text_hash(f"{seed}:target:{x['target']}")):
            if item['normalized_target_sha256'] in target_hashes: stats['duplicate_target_text'] += 1; continue
            # Reserve each answer identity once even if its first occurrence is
            # above the cap; a shorter duplicate never becomes quota filler.
            target_hashes.add(item['normalized_target_sha256'])
            stats['eligible_unique_targets_before_cap'] += 1
            stats['eligible_processed_positions_before_cap'] += len(item['ids'])
            if total + len(item['ids']) > cap: stats['finite_cap_skipped_targets'] += 1; continue
            total += len(item['ids']); admitted = True
            selected[split].append({'tree_id': tid, **item})
        if admitted:
            for h in hashes: message_split[h] = split
        else: meta['reason'] = 'no_unique_target_within_cap'
    stats['source_trees'] = len(trees); stats['source_messages'] = len(messages)
    stats['processed_tokens_all_splits'] = total; stats['unused_token_allowance'] = cap - total
    return selected, dict(stats), tree_meta


def verify(directory):
    import numpy as np
    directory = Path(directory); path = directory / 'manifest.json'; manifest = json.loads(path.read_text())
    if sha256(path) != (directory / 'manifest.sha256').read_text().strip() or manifest.get('format') != FORMAT:
        raise ValueError('SFT manifest integrity failure')
    for name, digest in manifest['hashes'].items():
        if sha256(safe_path(directory, name)) != digest: raise ValueError('SFT artifact integrity failure: ' + name)
    seen = set(); tree_splits = {}; total = 0
    for split, info in manifest['splits'].items():
        ids = np.load(directory / f'{split}.tokens.npy', allow_pickle=False)
        mask = np.load(directory / f'{split}.loss-mask.npy', allow_pickle=False)
        offsets = np.load(directory / f'{split}.offsets.npy', allow_pickle=False)
        if ids.dtype != np.dtype('<u2') or mask.dtype != np.bool_ or offsets.dtype != np.dtype('<i8'):
            raise ValueError('SFT dtype mismatch')
        if ids.ndim != 1 or mask.shape != ids.shape or offsets.shape != (info['examples'] + 1,) or offsets[0] != 0 or offsets[-1] != len(ids):
            raise ValueError('SFT layout mismatch')
        if len(ids) and int(ids.max()) >= manifest['vocab_size']: raise ValueError('SFT token out of vocabulary')
        with gzip.open(directory / f'{split}.provenance.jsonl.gz', 'rt') as f: traces = [json.loads(line) for line in f]
        if len(traces) != info['examples']: raise ValueError('SFT trace count mismatch')
        for i, trace in enumerate(traces):
            lo, hi = int(offsets[i]), int(offsets[i+1]); length = hi-lo; target = trace['target_message_id']; tree = trace['tree_id']
            if not 2 <= length <= manifest['sequence_len'] or target in seen: raise ValueError('Repeated target or invalid context')
            seen.add(target)
            if tree in tree_splits and tree_splits[tree] != split: raise ValueError('SFT tree leakage')
            tree_splits[tree] = split
            first = trace['supervised_start']
            if not 1 <= first < length or mask[lo:lo+first].any() or not mask[lo+first:hi].all(): raise ValueError('Assistant-only target mask mismatch')
            if int(ids[hi-1]) != manifest['assistant_end_token']: raise ValueError('Missing assistant end target')
        if len(ids) != info['processed_tokens'] or int(mask.sum()) != info['supervised_targets']:
            raise ValueError('SFT token counts mismatch')
        total += len(ids)
    if total != manifest['counts']['processed_tokens_all_splits'] or total > manifest['processed_token_cap']:
        raise ValueError('SFT finite token bound mismatch')
    return manifest


def prepare(receipt_path, pilot_data, pilot_corpus, benchmarks, output, *, cap=5_000_000, seed=20260922, tiny=False):
    import numpy as np
    receipt_path, pilot_data, output = map(Path, (receipt_path, pilot_data, output))
    if output.exists(): raise FileExistsError('Use a fresh immutable SFT output directory')
    receipt, messages = inputs(receipt_path, tiny=tiny)
    deny, pilot, lineage = heldout_deny(pilot_data, pilot_corpus, benchmarks, tiny=tiny)
    context = pilot['sequence_len'] if tiny else 2048
    if not tiny and pilot['vocab_size'] != 16384: raise ValueError('Owned 16K tokenizer required')
    tokenizer = native.Tokenizer(pilot_data / 'tokenizer.json', pilot['vocab_size'])
    selected, counts, trees = select(messages, tokenizer, deny, sequence_len=context, cap=cap, seed=seed)
    source_plan = json.loads((ROOT / 'research/sft-plan/manifest.json').read_text())
    pinned = next(item for item in source_plan['files'] if item['repo'] == REPO)
    if pinned['revision'] != REVISION: raise ValueError('SFT source snapshot revision changed')
    snapshots = {'OpenAssistant--oasst2.README.md': pinned['card_sha256'],
                 'OpenAssistant--oasst2.json': pinned['metadata_sha256']}
    for name, expected in snapshots.items():
        if sha256(ROOT / 'research/sft-plan' / name) != expected: raise ValueError('SFT source snapshot hash changed')
    output.mkdir(parents=True); provenance_dir = output / 'source'; provenance_dir.mkdir()
    shutil.copyfile(receipt_path, provenance_dir / 'acquisition-receipt.json')
    for name in snapshots:
        shutil.copyfile(ROOT / 'research/sft-plan' / name, provenance_dir / name)
    (provenance_dir / 'SOURCE_NOTICE.md').write_text(
        f'# Training-data source notice\n\nSource: {REPO}, revision `{REVISION}`.\n\n'
        f'Official pinned card: https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md\n\n'
        'The source card declares Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 . '
        'These tokenized examples remain derived from that source; the project MIT scope does not relicense third-party data. '
        'The exact source card, metadata, acquisition receipt and per-example source traces accompany this artifact.\n')
    shutil.copyfile(pilot_data / 'tokenizer.json', output / 'tokenizer.json')
    split_info = {}
    for split, examples in selected.items():
        token_ids = []; masks = []; offsets = [0]
        with (output / f'{split}.provenance.jsonl.gz').open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
            for ex in examples:
                start = ex['mask'].index(True)
                trace = {'source_repo': REPO, 'source_revision': REVISION, 'source_license': 'apache-2.0',
                    'tree_id': ex['tree_id'], 'target_message_id': ex['target'], 'normalized_target_sha256': ex['normalized_target_sha256'],
                    'message_path': [{'message_id': m, **messages[m]['trace']} for m in ex['path']],
                    'supervised_start': start, 'processed_tokens': len(ex['ids']), 'supervised_targets': sum(ex['mask']),
                    'selection': 'human_en_reviewed_not_deleted_rank0_ancestry; final_assistant_only'}
                gz.write((canonical(trace)+'\n').encode())
                token_ids.extend(ex['ids']); masks.extend(ex['mask']); offsets.append(len(token_ids))
        np.save(output / f'{split}.tokens.npy', np.asarray(token_ids,dtype='<u2'), allow_pickle=False)
        np.save(output / f'{split}.loss-mask.npy', np.asarray(masks,dtype=np.bool_), allow_pickle=False)
        np.save(output / f'{split}.offsets.npy', np.asarray(offsets,dtype='<i8'), allow_pickle=False)
        split_info[split] = {'examples': len(examples), 'trees': len({e['tree_id'] for e in examples}),
            'processed_tokens': len(token_ids), 'supervised_targets': sum(masks),
            'maximum_unbucketed_padded_compute_positions': len(examples) * (context-1)}
    dense.write_json(output / 'tree-decisions.json', trees)
    hashes = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    manifest = {'format': FORMAT, 'status': 'prepared_for_supply_review_not_training_authorization',
        'profile': 'tiny_cpu_test' if tiny else 'production', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'repo': REPO, 'revision': REVISION, 'license': 'apache-2.0', 'receipt_sha256': sha256(receipt_path)},
        'lineage': lineage, 'vocab_size': pilot['vocab_size'], 'sequence_len': context,
        'assistant_end_token': tokenizer.specials['<|assistant_end|>'], 'processed_token_cap': cap,
        'seed': seed, 'counts': counts, 'splits': split_info, 'hashes': hashes,
        'source_code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
            ('prepare_sft.py','infer_dense.py','prepare_dense_tokens.py','prepare_production.py','prepare_english.py')},
        'loss_alignment': 'Token-aligned mask; inputs=tokens[:-1], labels=tokens[1:], loss_mask=mask[1:]; no cross-example attention.',
        'split_policy': 'Upstream validation trees -> final; stable 2% upstream training trees -> dev; remaining -> train. Cross-split normalized message conflicts drop lower-priority entire tree (final,dev,train).',
        'finite_policy': 'Each assistant target ID and normalized answer admitted once; preceding assistants may repeat as masked context. No response truncation, wrapping or quota-filling repeats. Cap counts prompt/response/markers across all splits.',
        'limitations': ['Direct normalized/word-overlap screening cannot detect every paraphrase or teacher-derived overlap.',
            'Source review/language/human labels are source annotations, not independent adjudication.',
            'Eligible supply and usefulness require review; no SFT trainer or training authorization is included.']}
    dense.write_json(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(sha256(output / 'manifest.json')+'\n')
    verify(output)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-receipt', type=Path, required=True)
    p.add_argument('--pilot-data', type=Path, default=ROOT/'data/dense-tokens-production-001')
    p.add_argument('--pilot-corpus', type=Path, default=ROOT/'data/english-production-001')
    p.add_argument('--benchmarks', type=Path, default=ROOT/'data/benchmark-exclusions-002')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--processed-token-cap', type=int, default=5_000_000)
    a=p.parse_args(); result=prepare(a.source_receipt,a.pilot_data,a.pilot_corpus,a.benchmarks,a.output,cap=a.processed_token_cap)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'splits': result['splits']}))


if __name__=='__main__': main()
