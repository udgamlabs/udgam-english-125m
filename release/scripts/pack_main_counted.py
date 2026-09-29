"""Operational pending-count memo; exact frozen packing policy and token stream.

The scientific stage contract remains the frozen reference policy. This runtime
does not claim to execute its full SUM for every snapshot: it proves the count
at phase boundaries and accounts only for committed frozen shard transactions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import main_scanned_store as scanned
import pack_main_repaired as reference
import repair_main_suffix as repair

b=reference.b
POLICY=reference.POLICY
Stop=reference.Stop
PackStop=reference.PackStop
checkpoint=reference.checkpoint
REFERENCE_SOURCES={
    'prepare_main.py':'bb1a790af02675f37da246864bb766585801fd41e154c2c410d7e03749c7bc4c',
    'main_scanned_store.py':'55edd1e60ba7b0d0add7540dd913afac8a63cb40669127c530db3409e4fc05b5',
    'pack_main_bounded.py':'70f246c49bd93b9e08ae4d8f284c252f7a268ed2ac55de6b0264057de573dd69',
    'pack_main_scanned.py':'37436ed3b74ae4133eab5ee1f4f74a238bea39709d9064f3a96cb5b5239cc4c0',
    'repair_main_suffix.py':'6e2358beea918610e772099944fef5d1f74c4a4ea47ec1a181e69eb9d215c64b',
    'pack_main_repaired.py':'a2f457424969f107dbb9588e8bcf8d8192c3d3081dd944114f1539d65f71ee5b',
}


def sha(path):return b.source.file_hash(Path(path))


def runtime_plan():
    for name,checksum in REFERENCE_SOURCES.items():
        if sha(Path(__file__).parent/name)!=checksum:raise ValueError('Frozen packing reference source changed')
    return {'format':'udgam-counted-packing-runtime-v1','runtime_source_sha256':sha(__file__),
        'reference_sources':dict(REFERENCE_SOURCES),'reference_pending_query':scanned.PENDING_QUERY,
        'policy':dict(POLICY),'scientific_stage_contract_changed':False,
        'counter_policy':'initial exact SUM minus encoded positions of newly committed shards; final independent exact SUM',
        'snapshot_policy':'same reference schema and publication, pending value from committed-ledger memo during exclusive phase',
        'authorizer_reinstalled_at_chunk_boundaries':True,
        'frozen_chunk_reused':True,'document_order_changed':False,'index_or_cache_policy_changed':False,
        'operational_source_provenance_required':True}


def ledger_hash(rows):return repair.digest_value([[sid,raw] for sid,raw in rows])


class CountedStore(scanned.ScannedStore):
    def __init__(self,directory):
        if not (Path(directory)/'state.sqlite').is_file():raise ValueError('Existing reviewed stage required')
        self._memo=None;self._inside_chunk=False;self._violation=None;self._blocked_publish=False;self._counted_attempt=False
        super().__init__(directory)
        self.db.set_authorizer(self._authorize)

    def _authorize(self,action,table,column,database,trigger):
        forbidden=(sqlite3.SQLITE_CREATE_INDEX,sqlite3.SQLITE_CREATE_TABLE,sqlite3.SQLITE_CREATE_TRIGGER,
            sqlite3.SQLITE_CREATE_VIEW,sqlite3.SQLITE_DROP_INDEX,sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_DROP_TRIGGER,sqlite3.SQLITE_DROP_VIEW,sqlite3.SQLITE_ALTER_TABLE,
            sqlite3.SQLITE_ATTACH,sqlite3.SQLITE_DETACH)
        if action in forbidden:
            self._violation='Schema or attachment mutation forbidden';return sqlite3.SQLITE_DENY
        if action not in (sqlite3.SQLITE_INSERT,sqlite3.SQLITE_UPDATE,sqlite3.SQLITE_DELETE):return sqlite3.SQLITE_OK
        if table in ('meta','events'):return sqlite3.SQLITE_OK
        allowed=(action==sqlite3.SQLITE_UPDATE and
            ((table=='documents' and column in ('consumed','tokens')) or
             (table=='groups' and column=='packed') or
             (table=='components' and column=='packed_ordinary')))
        allowed=allowed or (action==sqlite3.SQLITE_INSERT and table=='shards')
        if self._inside_chunk and allowed:return sqlite3.SQLITE_OK
        self._violation='Mutation outside frozen chunk: '+str(action)+':'+str(table)+':'+str(column)
        return sqlite3.SQLITE_DENY

    def group(self,*args,**kwargs):raise ValueError('Acquisition is forbidden in CountedStore')

    def mutation_window(self,enabled):
        self._inside_chunk=enabled
        # SQLite authorizes prepared statements at compilation. Reinstalling a
        # non-null authorizer expires cached statements at this trust boundary.
        self.db.set_authorizer(self._authorize)

    def set(self,key,value):
        if self._memo is not None and key not in ('status','packing_incomplete','repair_final_pack_started','last_bounded_pack','last_counted_pack_proof'):
            raise ValueError('Non-packing metadata mutation forbidden during counted phase')
        return super().set(key,value)

    def _rows(self):
        rows=[(row[0],row[1]) for row in self.db.execute('SELECT id,manifest FROM shards ORDER BY id')]
        if len(rows)>10000:raise ValueError('Unexpected shard ledger size')
        return rows

    def _exact(self,reference_max_seconds,absolute_deadline):
        if self.db.in_transaction:raise ValueError('Exact reference count requires stopped transaction')
        until=min(time.time()+reference_max_seconds,absolute_deadline)
        if time.time()>=until:raise TimeoutError('Reference count deadline expired')
        self.db.set_progress_handler(lambda:int(time.time()>=until),1000)
        try:return scanned.pending_positions(self)
        except sqlite3.OperationalError as error:
            if str(error)=='interrupted':raise TimeoutError('Reference count deadline reached') from error
            raise
        finally:self.db.set_progress_handler(None,0)

    def begin_counted(self,reference_max_seconds,absolute_deadline):
        if self._memo is not None:raise ValueError('Counted phase already active')
        self.ready()
        if self.get('network_inflight',False):raise ValueError('Unreconciled acquisition')
        start=time.monotonic();pending=self._exact(reference_max_seconds,absolute_deadline)
        if type(pending) is not int or pending<0:raise ValueError('Invalid exact pending count')
        rows=self._rows()
        self._memo={'initial':pending,'pending':pending,'initial_rows':rows,'rows':rows[:],
            'encoded_committed':0,'data_version':self.db.execute('PRAGMA data_version').fetchone()[0],
            'source_cursor':repair.source_cursors(self),'stage_contract':self.contract_hash(),
            'reference_initial_seconds':time.monotonic()-start}

    def contract_hash(self):
        value=json.loads((self.directory/'manifest.json').read_text())
        value.update(config=self.get('config'),lineage=self.get('lineage'))
        return repair.digest_value(repair.contract(value))

    def pending_count(self):
        if self._memo is None:
            if self._counted_attempt:raise ValueError('Counted phase lacks a bounded exact baseline')
            return scanned.pending_positions(self)
        if self.db.in_transaction:raise ValueError('Memo cannot observe uncommitted row mutations')
        if self._violation:raise ValueError(self._violation)
        memo=self._memo
        if self.db.execute('PRAGMA data_version').fetchone()[0]!=memo['data_version']:
            raise ValueError('External database writer invalidated counted phase')
        rows=self._rows();previous=memo['rows']
        if rows[:len(previous)]!=previous:raise ValueError('Committed shard prefix changed')
        expected=previous[-1][0]+1 if previous else 0
        width=self.get('config')['sequence_len']+1;delta=0
        for sid,raw in rows[len(previous):]:
            item=json.loads(raw)
            if (sid!=expected or item.get('id')!=sid or item.get('row_width')!=width or
                type(item.get('rows')) is not int or item['rows']<=0 or item.get('wrapping') is not False or
                item.get('encoded_positions')!=item['rows']*width or
                item.get('prediction_targets')!=item['rows']*(width-1)):
                raise ValueError('New committed shard accounting differs')
            delta+=item['encoded_positions'];expected+=1
        if delta>memo['pending']:raise ValueError('Committed positions exceed exact supply')
        memo['pending']-=delta;memo['encoded_committed']+=delta;memo['rows']=rows
        return memo['pending']

    def reconcile(self,reference_max_seconds,absolute_deadline):
        pending=self.pending_count();start=time.monotonic()
        exact=self._exact(reference_max_seconds,absolute_deadline)
        cursor=repair.source_cursors(self);memo=self._memo;contract=self.contract_hash()
        proof={'initial_reference_pending_positions':memo['initial'],'final_reference_pending_positions':exact,
            'final_memo_pending_positions':pending,'committed_encoded_positions':memo['encoded_committed'],
            'initial_ledger_sha256':ledger_hash(memo['initial_rows']),'final_ledger_sha256':ledger_hash(memo['rows']),
            'initial_shard_count':len(memo['initial_rows']),'final_shard_count':len(memo['rows']),
            'source_cursor_sha256_before':memo['source_cursor'],'source_cursor_sha256_after':cursor,
            'stage_contract_sha256_before':memo['stage_contract'],'stage_contract_sha256_after':contract,
            'reference_initial_seconds':memo['reference_initial_seconds'],'reference_final_seconds':time.monotonic()-start,
            'independent_final_reference_verified':exact==pending and cursor==memo['source_cursor'] and contract==memo['stage_contract']}
        return proof

    def snapshot_pending(self):
        return None if self._blocked_publish else self.pending_count()

    def snapshot(self):
        values = {'schema_version':1,'status':self.get('status'),'config':self.get('config'),
            'lineage':self.get('lineage'),'downloaded_response_bytes':self.get('downloaded_bytes',0),
            'network_request_unreconciled':self.get('network_inflight',False),'components':self.totals(),
            'shards':[json.loads(r[0]) for r in self.db.execute('SELECT manifest FROM shards ORDER BY id')],
            'remaining_pending_positions':self.snapshot_pending(),
            'validation_policy':'frozen_pilot_validation_excluded_from_new_train',
            'limitations':['Available IDs only; Nemotron lacks original FineWiki article identity.',
                'Direct benchmark n-grams do not establish paraphrase/teacher contamination clearance.',
                'All pilot train families are conservatively excluded, including any not consumed by pilot.',
                'Any packed/excluded group linkage blocks this builder; no automatic holdout retirement.',
                'Frozen intrinsic validation is group-disjoint by available identities, not proven topic-disjoint.',
                'Main preparation throughput and finite multi-shard trainer integration require validation.',
                'Raw pages retain upstream rights; compact provenance supports pinned reacquisition.']}
        b.atomic_json(self.directory/'manifest.json',values)
        checksum=b.source.file_hash(self.directory/'manifest.json')
        temporary=self.directory/'manifest.sha256.tmp'
        with temporary.open('w') as f:f.write(checksum+'\n');f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.directory/'manifest.sha256');b.sync_directory(self.directory)
        return values


def pack(store,*,max_positions=None,stop=None,fault=None,reference_max_seconds=120,absolute_deadline=None):
    if not isinstance(store,CountedStore):raise ValueError('Exclusive CountedStore required')
    if not math.isfinite(reference_max_seconds) or not 0<reference_max_seconds<=120:raise ValueError('Bounded reference scan required')
    stop=repair.as_stop(stop)
    if absolute_deadline is None:raise ValueError('Original fixed wall deadline required')
    if not math.isfinite(absolute_deadline) or absolute_deadline<=time.time():raise ValueError('Original fixed wall deadline expired')
    if max_positions is not None and max_positions<=0:raise ValueError('max-positions must be positive')
    plan=runtime_plan();started=time.monotonic();cpu=time.process_time();proof={'runtime':plan,'independent_final_reference_verified':False}
    old_cache=store.db.execute('PRAGMA cache_size').fetchone()[0]
    error=None;outcome='complete';result=None;before=store.db.execute('SELECT COUNT(*) FROM shards').fetchone()[0]
    cap=None
    store._counted_attempt=True
    try:
        store.begin_counted(reference_max_seconds,absolute_deadline)
        reference.verify_policy(store);reference.authorize_pack(store);store.ready();b.recover_pack(store)
        store.snapshot();checkpoint(store);b.verify(store)
        config=store.get('config');width=config['sequence_len']+1
        if max_positions is not None and max_positions<=0:raise ValueError('max-positions must be positive')
        cap=min(POLICY['max_shard_positions'],config['shard_positions'],POLICY['max_shard_positions'] if max_positions is None else max_positions)
        if cap<width:raise ValueError('Bounded shard cap must fit one sequence row')
        store.db.execute(f"PRAGMA cache_size=-{POLICY['sqlite_cache_bytes']//1024}")
        with store.db:
            store.set('packing_incomplete',True)
            store.event('bounded_pack_started',{'max_positions':cap,'committed_shards_before':before})
        while True:
            stop.check()
            available=store.pending_count();rows=min(cap//width,available//width)
            if not rows:break
            store.mutation_window(True)
            try:reference._pack_chunk(store,rows,width,stop,fault=fault)
            finally:store.mutation_window(False)
    except PackStop as exc:outcome=exc.reason
    except Exception as exc:
        error=exc
        outcome='resource_limit' if isinstance(exc,RuntimeError) and ('reserve' in str(exc) or 'allocation cap' in str(exc)) else 'error'
    finally:
        store.db.rollback()
        try:
            try:
                b.recover_pack(store);checkpoint(store)
                if store._memo is None:raise ValueError('No verified initial reference count')
                proof.update(store.reconcile(reference_max_seconds,absolute_deadline))
                if not proof['independent_final_reference_verified']:raise ValueError('Exact final pending/cursor proof mismatch')
                if sha(__file__)!=plan['runtime_source_sha256']:raise ValueError('Counted runtime changed during phase')
            except Exception as exc:
                outcome='recovery_required';error=exc
                proof.update(independent_final_reference_verified=False,error_class=type(exc).__name__)
            if error is not None and outcome!='resource_limit':
                store.set('status','blocked_packing_integrity')
            receipt={'outcome':outcome,'error':None if error is None else f'{type(error).__name__}: {error}',
                'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
                'committed_shards_before':before,'committed_shards_after':store.db.execute('SELECT COUNT(*) FROM shards').fetchone()[0],
                'max_positions':cap,'policy':POLICY,'counted_runtime_proof':proof}
            if outcome=='complete':store.set('packing_incomplete',False)
            store.event('bounded_pack_finished',receipt);store.set('last_bounded_pack',receipt)
            store.set('last_counted_pack_proof',proof);store.db.commit()
            # A rejected proof must never republish a memo as independently valid.
            if not proof['independent_final_reference_verified']:
                store._memo=None
                store._blocked_publish=True
            result=store.snapshot()
        finally:
            store.db.execute(f'PRAGMA cache_size={old_cache}');store._memo=None;store._counted_attempt=False
    if error is not None and outcome in ('error','recovery_required'):raise error
    return {'outcome':outcome,'receipt':receipt,'manifest':result,'counted_runtime_proof':proof}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,required=True);args=parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds<240:raise ValueError('At least240s bounded phase required')
    deadline=time.time()+args.max_seconds
    with Stop(args.max_seconds-120) as stop:
        store=CountedStore(args.directory)
        try:value=pack(store,stop=stop,absolute_deadline=deadline)
        finally:store.close()
    print(json.dumps(value['receipt'],indent=2))
    return 0 if value['outcome']=='complete' else 75


if __name__=='__main__':raise SystemExit(main())
