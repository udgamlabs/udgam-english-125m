"""Explicit, offline-only retirement of one proven untrained benchmark family.

Operates on a fresh restored POSIX stage, never on the immutable parent Volume.
Every stop leaves a blocked, resumable repair. No downloads or model operations.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np
import prepare_main as b
import main_scanned_store as scanned
import pack_main_bounded as bounded
import modal_transfer as common

PLAN='repair/plan.json'
RECEIPT='repair/receipt.json'
SOURCES=('repair_main_suffix.py','pack_main_repaired.py')
EVIDENCE=('receipt.json','manifest.json','evidence/hook/conflict.json','evidence/shard-exposure.json',
    'evidence/pilot-families.json','evidence/benchmark-matches.json')
MAX_DOC_BYTES=2_000_000
BATCH=64
PRODUCTION={'prefix_shards':36,'prefix_rows':97986,'prefix_positions':200773314,
    'effective_horizon_targets':1999994880,
    'checkpoint_sha256':'9fc9fdfa07b202acee33c351542448ee55968bbd5311c079fdfc55a823637fdb',
    'source_parent':{'receipt_sha256':'7749963bb51a589d689be477ac8f721ef422fc1ccd975acf332ead9cd2474297',
        'manifest_sha256':'2988d4e9fc2291f342d191894700265bddc481e2d8d1ad92825f32079d204c72',
        'stage_manifest_sha256':'7a038a60b2612d062254ae6fef38530bc2c425958676b9b9d8a4c7fd08142235'},
    'forensic':{'receipt_sha256':'f5fbe95937e09c30ec36a129d6fcea4b6ade415a20418a39dfd78cee0171d4c0',
        'manifest_sha256':'65a29a4b88191e8404ca0be12d6ab6a6562b2d9a2e7544ba31870bc9760afce1',
        'conflict_sha256':'7c8e529c47346a62dc916005caa5e171b1087c84a5864d165a70645adeb827b4'},
    'retired_document_id':1796455,'retired_positions':332,'retired_ordinary':331,
    'old_tranche':100000000,'new_tranche':2000000000,'accepted_before':999984667,'packed_before':999983083}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest_value(value):return hashlib.sha256(canonical(value)).hexdigest()


def as_stop(stop):
    if stop is None:return bounded.Stop()
    if hasattr(stop,'check'):return stop
    if not callable(stop):raise TypeError('Stop must expose check() or be a boolean callback')
    class CallbackStop:
        def check(self):
            if stop():raise bounded.PackStop('requested_stop')
    return CallbackStop()


def contract(value):
    return {'config':{k:v for k,v in value['config'].items() if k not in ('max_download_bytes','max_output_bytes','disk_reserve_bytes')},
        'lineage':{k:v for k,v in value['lineage'].items() if k not in ('frozen_validation_path','token_bytes_path')},
        'validation_policy':value['validation_policy']}


def save_once(path,value):
    data=json.dumps(value,indent=2,sort_keys=True)+'\n';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        common.regular(path.parent,path.name)
        if path.read_text()!=data:raise ValueError('Existing immutable repair artifact differs')
        return
    temporary=path.with_suffix(path.suffix+'.tmp')
    if temporary.exists():
        common.regular(temporary.parent,temporary.name)
        if not data.encode().startswith(temporary.read_bytes()):raise ValueError('Unknown repair publication temporary')
        temporary.unlink()
    with temporary.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(temporary,path);temporary.unlink();b.sync_directory(path.parent)


def source_cursors(store):
    h=hashlib.sha256()
    for name,order in [('files','id'),('jobs','file_id,rg')]:
        for row in store.db.execute('SELECT * FROM '+name+' ORDER BY '+order):
            h.update(canonical([name,[(v.hex() if isinstance(v,bytes) else v) for v in row]]));h.update(b'\n')
    h.update(canonical({'downloaded_bytes':store.get('downloaded_bytes',0),'network_inflight':store.get('network_inflight',False)}))
    return h.hexdigest()


def copy_exact(source,destination):
    item=common.describe(source.parent,source.name);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if common.describe(destination.parent,destination.name)['sha256']!=item['sha256']:raise ValueError('Repair evidence changed')
    else:
        with source.open('rb') as i,destination.open('xb') as o:shutil.copyfileobj(i,o,1024**2);o.flush();os.fsync(o.fileno())
        if common.digest(destination)!=item['sha256']:raise ValueError('Repair copy failed integrity')


def check_evidence(evidence,case):
    for path,key in [('receipt.json','receipt_sha256'),('manifest.json','manifest_sha256'),('evidence/hook/conflict.json','conflict_sha256')]:
        if common.digest(common.regular(evidence,path))!=case['forensic'][key]:raise ValueError('Exact forensic evidence required')
    manifest=common.read_json(evidence/'manifest.json');receipt=common.read_json(evidence/'receipt.json')
    if receipt.get('evidence_complete') is not True or receipt.get('committed') is not True or receipt.get('ready_for_training') is not False:
        raise ValueError('Complete observation-only forensic receipt required')
    lookup={i['path']:i for i in manifest['files']};common.check_items(manifest['files'])
    for name in EVIDENCE[2:]:
        if common.describe(evidence,name)!={**lookup[name.removeprefix('evidence/')],'path':name}:raise ValueError('Forensic evidence inventory mismatch')
    capture=common.read_json(evidence/'evidence/hook/conflict.json')
    if (capture.get('capture_complete') is not True or capture.get('combined_blocked_mask')!=4
        or capture.get('incoming_blocked_mask')!=4 or capture.get('packed_document_count')!=1
        or [d['id'] for d in capture['packed_documents']]!=[case['retired_document_id']]
        or any(capture.get('truncation',{}).values())):raise ValueError('Exact complete one-family benchmark conflict required')
    doc=capture['packed_documents'][0]
    if (doc['positions'],doc['consumed'],doc['ordinary'])!=(case['retired_positions'],case['retired_positions'],case['retired_ordinary']):
        raise ValueError('Captured retired document differs')
    for key in capture['candidate_keys']:
        if b.hkey(key['key']).hex()!=key['sha256']:raise ValueError('Candidate key preimage mismatch')
    exposure=common.read_json(evidence/'evidence/shard-exposure.json')
    if (exposure.get('mapping_complete') is not True or exposure.get('missing_document_ids') or exposure.get('coverage_mismatches')
        or any(i['shard_id']<case['prefix_shards'] for i in exposure['intervals'])
        or sum(i['document_position_stop']-i['document_position_start'] for i in exposure['intervals'])!=case['retired_positions']):
        raise ValueError('Retired document overlaps protected inventory or coverage is incomplete')
    pilot=common.read_json(evidence/'evidence/pilot-families.json')
    if pilot.get('mapping_complete') is not True or pilot.get('match_count')!=0:raise ValueError('Pilot-family relationship requires separate review')
    matches=common.read_json(evidence/'evidence/benchmark-matches.json')
    if matches.get('mapping_complete') is not True or not any(i.get('exact_words_match') for i in matches['matches']):
        raise ValueError('Observed exact benchmark match required')
    return capture


def original_path(store,relative):
    path=store.directory/relative
    retired=store.directory/'repair/original-shards'/Path(relative).name
    if path.exists() and retired.exists():raise ValueError('Ambiguous original shard copies')
    return path if path.exists() else retired


def verify_originals(store,plan,stop):
    for shard in plan['original_shards']:
        for relative,sha in shard['hashes'].items():
            stop.check();path=original_path(store,relative)
            if common.digest(common.regular(path.parent,path.name))!=sha:raise ValueError('Original shard changed')


def initialize(store,evidence,case,stop):
    plan_path=store.directory/PLAN
    if plan_path.exists():
        plan=common.read_json(common.regular(store.directory,PLAN))
        if plan['case']!=case:raise ValueError('Repair case changed on resume')
        for name,sha in plan['source_sha256'].items():
            if common.digest(Path(__file__).parent/name)!=sha:raise ValueError('Repair source changed on resume')
    else:
        store.ready()
        if store.get('network_inflight',False) or store.get('packing_incomplete',False) or (store.directory/'packing.json').exists():
            raise ValueError('Clean stopped source required')
        old=common.read_json(store.directory/'manifest.json')
        if common.digest(store.directory/'manifest.json')!=case['source_parent']['stage_manifest_sha256']:raise ValueError('Exact parent stage required')
        if old['shards']!=[json.loads(r[0]) for r in store.db.execute('SELECT manifest FROM shards ORDER BY id')]:raise ValueError('Stale parent inventory')
        protected=old['shards'][:case['prefix_shards']]
        if (len(protected)!=case['prefix_shards'] or sum(s['rows'] for s in protected)!=case['prefix_rows']
            or sum(s['encoded_positions'] for s in protected)!=case['prefix_positions']):raise ValueError('Protected full shard boundary differs')
        if old['config']['tranche_tokens']!=case['old_tranche'] or old['config']['stage_tokens']!=case['new_tranche']:
            raise ValueError('Exact deferred-acquisition schedule transition required')
        totals=store.totals()
        if sum(v['accepted'] for v in totals.values())!=case['accepted_before'] or sum(v['packed'] for v in totals.values())!=case['packed_before']:
            raise ValueError('Original ordinary token counters differ')
        capture=check_evidence(evidence,case);group=capture['packed_documents'][0]['group_id']
        actual=[dict(r) for r in store.db.execute('SELECT id,positions,consumed,ordinary,group_id FROM documents WHERE group_id=? ORDER BY id',(group,))]
        expected={k:capture['packed_documents'][0][k] for k in ('id','positions','consumed','ordinary','group_id')}
        if actual!=[expected]:raise ValueError('Unexpected additional family documents; review required')
        hashes=sorted(r[0].hex() for r in store.db.execute('SELECT key FROM identity WHERE group_id=?',(group,)))
        if hashes!=sorted(i['sha256'] for i in capture['family_identity_hashes']):raise ValueError('Captured family differs from restored parent')
        key_hashes=sorted(set(hashes)|{i['sha256'] for i in capture['candidate_keys']})
        source_hashes={name:common.digest(Path(__file__).parent/name) for name in SOURCES}
        for name in EVIDENCE:copy_exact(common.regular(evidence,name),store.directory/'repair/evidence'/name)
        for name in SOURCES:copy_exact(Path(__file__).parent/name,store.directory/'repair/source'/name)
        plan={'format':'udgam-main-suffix-repair-plan-v1','case':case,'checkpoint_sha256':case['checkpoint_sha256'],
            'source_parent':case['source_parent'],'forensic':case['forensic'],'old_stage_contract':contract(old),
            'old_stage_contract_sha256':digest_value(contract(old)),'original_shards':old['shards'],'protected_shards':protected,
            'protected_inventory_sha256':digest_value(protected),'identity_hashes':key_hashes,
            'candidate_keys':[i['key'] for i in capture['candidate_keys']],'retired_document_ids':[case['retired_document_id']],
            'source_cursor_before_sha256':source_cursors(store),'counters_before':totals,'source_sha256':source_hashes,
            'original_document_id_ceiling':store.db.execute('SELECT COALESCE(MAX(id),0) FROM documents').fetchone()[0],
            'order_policy':'protected-prefix; recovered old stream including old pending tail; new documents by original order_key,id',
            'eligibility_policy_changed':False,'schedule_migration':True}
        save_once(plan_path,plan)
    verify_originals(store,plan,stop)
    if not store.get('repair_progress'):
        with store.db:
            store.db.execute('BEGIN IMMEDIATE')
            store.db.execute('CREATE TABLE repair_order(document_id INTEGER PRIMARY KEY,rank INTEGER UNIQUE NOT NULL,original_consumed INTEGER NOT NULL,prefix_consumed INTEGER NOT NULL)')
            store.db.execute('CREATE TABLE repair_seen(document_id INTEGER PRIMARY KEY,ordinal INTEGER UNIQUE NOT NULL)')
            store.set('repair_progress',{'phase':'reset_groups','group_cursor':0,'documents_committed':0,'archive_cursor':0})
            store.set('status','repair_in_progress')
        store.snapshot();bounded.checkpoint(store)
    return plan


def guard_commit(store,count,payload=0):
    store.disk_check(bounded.dirty_reservation(store,count,payload)+16*1024**2)


def checkpoint_if_large(store):
    wal=store.directory/'state.sqlite-wal'
    if wal.exists() and wal.stat().st_size>=64*1024**2:bounded.checkpoint(store)


def reset_groups(store,stop,fault):
    progress=store.get('repair_progress')
    while progress['phase']=='reset_groups':
        stop.check();rows=store.db.execute('SELECT id FROM groups WHERE id>? ORDER BY id LIMIT 256',(progress['group_cursor'],)).fetchall()
        guard_commit(store,len(rows))
        with store.db:
            for row in rows:store.db.execute('UPDATE groups SET packed=0 WHERE id=?',(row['id'],))
            if rows:progress['group_cursor']=rows[-1]['id']
            else:progress['phase']='reconstruct'
            store.set('repair_progress',progress)
        checkpoint_if_large(store)
        if fault:fault('after_group_batch')


def stream_documents(store,plan,stop):
    """Yield exact packed document prefixes; at most one bounded document in RAM."""
    current=None;payload=bytearray();prefix=0;ordinal=0;retained=hashlib.sha256();removed=0
    for shard in plan['original_shards']:
        stop.check();sid=shard['id'];array=next(p for p in shard['hashes'] if p.endswith('.npy'))
        trace=next(p for p in shard['hashes'] if p.endswith('.jsonl.gz'))
        tokens=np.load(original_path(store,array),mmap_mode='r',allow_pickle=False)
        if tokens.dtype!=np.dtype('<u2') or list(tokens.shape)!=[shard['rows'],shard['row_width']]:raise ValueError('Original array shape differs')
        flat=tokens.reshape(-1);position=0
        with gzip.open(original_path(store,trace),'rt') as f:
            for line in f:
                row=json.loads(line);doc=row['document_id'];start,end=row['document_position_start'],row['document_position_stop']
                lo,hi=row['shard_position_start'],row['shard_position_stop']
                if lo!=position or not 0<=lo<hi<=len(flat) or hi-lo!=end-start:raise ValueError('Original provenance coverage gap')
                if doc!=current:
                    if current is not None:yield ordinal,current,bytes(payload),prefix;ordinal+=1
                    current=doc;payload=bytearray();prefix=0
                if start!=len(payload)//2:raise ValueError('Document intervals noncontiguous or repeated')
                raw=flat[lo:hi].tobytes();payload.extend(raw);position=hi
                if len(payload)>MAX_DOC_BYTES:raise ValueError('Document reconstruction exceeds explicit bound')
                if sid<plan['case']['prefix_shards']:
                    if doc in plan['retired_document_ids']:raise ValueError('Retirement touches protected shard')
                    prefix=end
                elif doc in plan['retired_document_ids']:removed+=hi-lo
                else:retained.update(raw)
        if position!=len(flat):raise ValueError('Shard provenance does not cover every stored token')
        del flat,tokens
    if current is not None:
        yield ordinal,current,bytes(payload),prefix;ordinal+=1
        # The final original document may have an unconsumed stored tail.
        doc=store.db.execute('SELECT positions,tokens FROM documents WHERE id=?',(current,)).fetchone()
        if len(payload)<doc['positions']*2:
            if doc['tokens'] is None:raise ValueError('Original partial tail is missing')
            full=zlib.decompress(doc['tokens'])
            if len(full)!=doc['positions']*2 or full[:len(payload)]!=payload:raise ValueError('Original partial overlap differs')
            if current not in plan['retired_document_ids']:retained.update(full[len(payload):])
    # Include never-packed original pending documents, in their original order.
    for doc in store.db.execute('SELECT d.* FROM documents d JOIN groups g ON g.id=d.group_id WHERE d.id<=? AND d.consumed=0 AND d.tokens IS NOT NULL AND g.blocked=0 AND NOT EXISTS(SELECT 1 FROM repair_seen s WHERE s.document_id=d.id) ORDER BY d.order_key,d.id',(plan['original_document_id_ceiling'],)):
        stop.check();full=zlib.decompress(doc['tokens'])
        if len(full)!=doc['positions']*2:raise ValueError('Original unconsumed document length differs')
        if doc['id'] not in plan['retired_document_ids']:retained.update(full)
    if removed!=plan['case']['retired_positions']:raise ValueError('Unexpected retired token coverage')
    yield None,ordinal,retained.hexdigest(),removed


def reconstruct(store,plan,stop,fault):
    progress=store.get('repair_progress');batch=[];proof=None
    if progress['phase']!='reconstruct':return
    def flush():
        nonlocal batch,progress
        if not batch:return
        stop.check();guard_commit(store,len(batch),sum(len(r[2]) for r in batch))
        with store.db:
            for ordinal,doc_id,raw,prefix in batch:
                doc=store.db.execute('SELECT * FROM documents WHERE id=?',(doc_id,)).fetchone()
                if doc is None or len(raw)!=doc['consumed']*2 or not prefix<=doc['consumed']<=doc['positions']:
                    raise ValueError('Original document coverage differs from SQLite')
                full=zlib.decompress(doc['tokens']) if doc['tokens'] is not None else raw
                if len(full)!=doc['positions']*2 or full[:len(raw)]!=raw:raise ValueError('Original token payload/packed overlap mismatch')
                store.db.execute('INSERT INTO repair_seen VALUES(?,?)',(doc_id,ordinal))
                if prefix<doc['positions']:
                    store.db.execute('INSERT INTO repair_order VALUES(?,?,?,?)',(doc_id,ordinal,doc['consumed'],prefix))
                    store.db.execute('UPDATE documents SET tokens=?,consumed=? WHERE id=?',(zlib.compress(full,1),prefix,doc_id))
                if prefix:store.db.execute('UPDATE groups SET packed=1 WHERE id=?',(doc['group_id'],))
                progress['documents_committed']=ordinal+1
            store.set('repair_progress',progress)
        batch=[];checkpoint_if_large(store)
        if fault:fault('after_reconstruction_batch')
    for ordinal,doc_id,raw,prefix in stream_documents(store,plan,stop):
        if ordinal is None:proof={'documents':doc_id,'sha256':raw,'removed_positions':prefix};break
        if ordinal<progress['documents_committed']:continue
        batch.append((ordinal,doc_id,raw,prefix))
        if len(batch)>=BATCH:flush()
    flush();stop.check()
    # Parent documents that were accepted but never packed retain their original
    # complete payload. Rank them only after the existing recorded stream.
    last=b'',0;rank=max(proof['documents'],store.db.execute('SELECT COALESCE(MAX(rank),-1)+1 FROM repair_order').fetchone()[0])
    while True:
        stop.check();rows=store.db.execute('SELECT d.* FROM documents d JOIN groups g ON g.id=d.group_id WHERE d.tokens IS NOT NULL AND g.blocked=0 AND d.consumed=0 AND NOT EXISTS(SELECT 1 FROM repair_order r WHERE r.document_id=d.id) AND (d.order_key,d.id)>(?,?) ORDER BY d.order_key,d.id LIMIT 64',last).fetchall()
        if not rows:break
        guard_commit(store,len(rows))
        with store.db:
            for doc in rows:
                raw=zlib.decompress(doc['tokens'])
                if len(raw)!=doc['positions']*2:raise ValueError('Pending tail token length differs')
                store.db.execute('INSERT INTO repair_order VALUES(?,?,0,0)',(doc['id'],rank));rank+=1
            last=rows[-1]['order_key'],rows[-1]['id']
        checkpoint_if_large(store)
    with store.db:
        progress.update(phase='quarantine',stream_proof=proof);store.set('repair_progress',progress)
    bounded.checkpoint(store)


def quarantine(store,plan,stop,fault):
    progress=store.get('repair_progress')
    if progress['phase']!='quarantine':return
    stop.check();target=plan['case']['retired_document_id'];doc=store.db.execute('SELECT * FROM documents WHERE id=?',(target,)).fetchone()
    if doc['consumed']!=0 or doc['tokens'] is None:raise ValueError('Retirement must operate on fully pending document')
    counts=Counter()
    for shard in plan['protected_shards']:counts.update(shard['ordinary_positions_by_component'])
    guard_commit(store,16,len(doc['tokens']))
    with store.db:
        for component in plan['counters_before']:store.db.execute('UPDATE components SET packed_ordinary=? WHERE name=?',(counts[component],component))
        root,blocked=store.group(plan['candidate_keys'],4)
        if blocked!=4 or store.db.execute('SELECT packed FROM groups WHERE id=?',(root,)).fetchone()[0]!=0:raise ValueError('Retired family unexpectedly touches protected prefix')
        keys=sorted(r[0].hex() for r in store.db.execute('SELECT key FROM identity WHERE group_id=?',(root,)))
        if keys!=plan['identity_hashes']:raise ValueError('Retired family closure differs')
        after=store.totals();expected=copy.deepcopy(plan['counters_before'])
        for component in expected:expected[component]['packed']=counts[component]
        expected[doc['component']]['accepted']-=doc['ordinary']
        if after!=expected:raise ValueError('Unexpected retirement counter change')
        progress.update(phase='verify_reconstruction',counters_after=after);store.set('repair_progress',progress)
    bounded.checkpoint(store)
    if fault:fault('after_quarantine')


def verify_reconstruction(store,plan,stop):
    h=hashlib.sha256();count=0
    for row in store.db.execute('SELECT d.*,r.original_consumed,r.prefix_consumed FROM repair_order r JOIN documents d ON d.id=r.document_id JOIN groups g ON g.id=d.group_id WHERE g.blocked=0 ORDER BY r.rank'):
        if count%BATCH==0:stop.check()
        raw=zlib.decompress(row['tokens'])
        if len(raw)!=row['positions']*2 or row['consumed']!=row['prefix_consumed']:raise ValueError('Recovered document cursor/length differs')
        h.update(raw[row['prefix_consumed']*2:]);count+=1
    proof=store.get('repair_progress')['stream_proof']
    if h.hexdigest()!=proof['sha256']:raise ValueError('Retained old suffix token stream differs')
    if source_cursors(store)!=plan['source_cursor_before_sha256']:raise ValueError('Source cursor or download accounting changed')
    if store.db.execute('SELECT COUNT(*) FROM documents WHERE tokens IS NOT NULL AND consumed>0').fetchone()[0]>1:
        raise ValueError('More than one protected-boundary partial document')
    return h.hexdigest()


def archive_and_finalize(store,plan,stop,fault):
    progress=store.get('repair_progress')
    if progress['phase']=='verify_reconstruction':
        sha=verify_reconstruction(store,plan,stop)
        with store.db:progress.update(phase='archive',verified_stream_sha256=sha);store.set('repair_progress',progress)
    archive=[(p,h) for s in plan['original_shards'][plan['case']['prefix_shards']:] for p,h in s['hashes'].items()]
    if progress['phase']=='archive':
        for index in range(progress['archive_cursor'],len(archive)):
            stop.check();relative,sha=archive[index];source=store.directory/relative;destination=store.directory/'repair/original-shards'/Path(relative).name
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if source.exists() or common.digest(destination)!=sha:raise ValueError('Ambiguous archived original')
            else:
                if common.digest(source)!=sha:raise ValueError('Original changed before archive')
                os.rename(source,destination);b.sync_directory(destination.parent);b.sync_directory(source.parent)
            if fault:fault('after_archive_rename')
            with store.db:progress['archive_cursor']=index+1;store.set('repair_progress',progress)
        with store.db:
            store.db.execute('DELETE FROM shards WHERE id>=?',(plan['case']['prefix_shards'],))
            config=store.get('config');config['tranche_tokens']=plan['case']['new_tranche'];store.set('config',config)
            lineage=store.get('lineage');additions=metadata_additions(store.directory)
            if any(k in lineage.get('metadata_artifact_hashes',{}) for k in additions):raise ValueError('Repair metadata collision')
            lineage['metadata_artifact_hashes']={**lineage.get('metadata_artifact_hashes',{}),**additions}
            lineage['data_repair']={'plan_path':PLAN,'plan_sha256':common.digest(store.directory/PLAN),'source_sha256':plan['source_sha256']}
            store.set('lineage',lineage);progress['phase']='publish_receipt';store.set('repair_progress',progress)
        bounded.checkpoint(store)
    if progress['phase']=='publish_receipt':
        stop.check();verify_originals(store,plan,stop);value=store.snapshot();new=contract(value);case=plan['case']
        receipt={'format':'udgam-main-suffix-repair-receipt-v1','complete':True,'ready_for_training':False,'schedule_migration':True,
            'plan_sha256':common.digest(store.directory/PLAN),'checkpoint_sha256':plan['checkpoint_sha256'],
            'source_parent':plan['source_parent'],'forensic':plan['forensic'],'old_stage_contract':plan['old_stage_contract'],
            'new_stage_contract':new,'old_stage_contract_sha256':plan['old_stage_contract_sha256'],'new_stage_contract_sha256':digest_value(new),
            'protected_shards':plan['protected_shards'],'protected_inventory_sha256':plan['protected_inventory_sha256'],
            'protected_prefix_no_overlap':True,'source_cursor_before_sha256':plan['source_cursor_before_sha256'],
            'source_cursor_after_sha256':source_cursors(store),'retained_suffix_stream_sha256_before':progress['stream_proof']['sha256'],
            'retained_suffix_stream_sha256_after':progress['verified_stream_sha256'],'retired_document_ids':plan['retired_document_ids'],
            'retired_encoded_positions':case['retired_positions'],'retired_ordinary_positions':case['retired_ordinary'],
            'identity_hashes':plan['identity_hashes'],'counters_before':plan['counters_before'],'counters_after':progress['counters_after'],
            'allowed_diff':{'config.tranche_tokens':{'before':case['old_tranche'],'after':case['new_tranche']},
                'lineage_additions':['data_repair'],'metadata_additions':metadata_additions(store.directory)}}
        if receipt['source_cursor_before_sha256']!=receipt['source_cursor_after_sha256']:raise ValueError('Source cursor changed')
        save_once(store.directory/RECEIPT,receipt)
        if fault:fault('after_receipt')
        with store.db:progress['phase']='complete';store.set('repair_progress',progress);store.set('status','initialized')
        store.snapshot();bounded.checkpoint(store)


def metadata_additions(directory):
    paths=[PLAN]+['repair/source/'+n for n in SOURCES]+['repair/evidence/'+n for n in EVIDENCE]
    return {p:common.digest(common.regular(directory,p)) for p in paths}


def repair_artifacts(directory):
    """Exact bounded existing repair files, including interrupted archive moves."""
    plan=common.read_json(common.regular(directory,PLAN));paths=list(metadata_additions(directory))
    if (directory/RECEIPT).exists():paths.append(RECEIPT)
    for shard in plan['original_shards'][plan['case']['prefix_shards']:]:
        for relative in shard['hashes']:
            name='repair/original-shards/'+Path(relative).name
            if (directory/name).exists():paths.append(name)
    return [common.describe(directory,name) for name in paths]


def repair(directory,evidence_directory,*,stop=None,fault=None,_case=None):
    """Repair only a fresh restored stage. _case is fixture injection, not CLI."""
    stop=as_stop(stop);store=scanned.ScannedStore(directory);case=PRODUCTION if _case is None else _case
    try:
        if store.get('repair_progress',{}).get('phase')=='complete':
            receipt=common.read_json(common.regular(directory,RECEIPT))
            if receipt['new_stage_contract']!=contract(store.snapshot()):raise ValueError('Completed repair contract changed')
            return {'complete':True,'status':'repair_complete','receipt_path':RECEIPT,'receipt':receipt}
        plan=initialize(store,evidence_directory,case,stop)
        reset_groups(store,stop,fault);reconstruct(store,plan,stop,fault);quarantine(store,plan,stop,fault)
        archive_and_finalize(store,plan,stop,fault)
        return {'complete':True,'status':'repair_complete','receipt_path':RECEIPT,'receipt':common.read_json(directory/RECEIPT)}
    except bounded.PackStop as exc:
        store.db.rollback();store.snapshot();bounded.checkpoint(store)
        return {'complete':False,'status':'repair_in_progress','reason':exc.reason,'progress':store.get('repair_progress'), 'ready_for_training':False}
    finally:store.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--evidence-directory',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args(argv)
    with bounded.Stop(a.max_seconds) as stop:result=repair(a.directory,a.evidence_directory,stop=stop)
    print(json.dumps(result,indent=2));return 0 if result['complete'] else 75


if __name__=='__main__':raise SystemExit(main())
