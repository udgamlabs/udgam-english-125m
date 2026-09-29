"""Explicit repaired-stream packing; frozen bounded chunk semantics retained."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time
import zlib
import prepare_main as b
import main_scanned_store as scanned
import pack_main_bounded as bounded
import pack_main_scanned as original
import repair_main_suffix as repair
from pack_main_bounded import Stop,PackStop,checkpoint,POLICY,AllocationGuard,dirty_reservation


def verify_policy(store):
    original.verify_policy(store)
    lineage=store.get('lineage');pin=lineage.get('data_repair',{})
    if store.get('repair_progress',{}).get('phase')!='complete':raise ValueError('Repair incomplete')
    if pin.get('plan_sha256')!=repair.common.digest(repair.common.regular(store.directory,repair.PLAN)):
        raise ValueError('Repair plan changed')
    plan=repair.common.read_json(store.directory/repair.PLAN)
    for name in repair.SOURCES:
        if pin.get('source_sha256',{}).get(name)!=repair.common.digest(Path(__file__).parent/name):raise ValueError('Repair runtime source changed')
    receipt=repair.common.read_json(repair.common.regular(store.directory,repair.RECEIPT))
    if receipt.get('complete') is not True or receipt['new_stage_contract']!=repair.contract(store.snapshot()):raise ValueError('Completed repair contract differs')
    if receipt['protected_shards']!=[json.loads(r[0]) for r in store.db.execute('SELECT manifest FROM shards ORDER BY id LIMIT ?',(plan['case']['prefix_shards'],))]:raise ValueError('Protected shard inventory changed')


def acquisition_finished(store):
    """Require the exact frozen transport quota-stop receipt, not a timer stop."""
    import acquire_main_cached as transport
    plan=repair.common.read_json(repair.common.regular(store.directory,repair.PLAN))
    target=plan['case']['new_tranche'];config=store.get('config')
    if config['tranche_tokens']!=target or config['stage_tokens']!=target or store.get('status')!='initialized' or store.get('network_inflight',False):return False
    n=store.get('cached_transport_run_sequence',0)
    folder=store.directory/'transport/runs'/f'{n:05d}'
    if not (folder/'start.json').exists() or not (folder/'finish.json').exists():return False
    start=repair.common.read_json(repair.common.regular(folder,'start.json'));finish=repair.common.read_json(repair.common.regular(folder,'finish.json'))
    if (finish.get('id')!=n or start.get('id')!=n or finish.get('outcome')!='tranche_stopped_at_target_or_document_boundary'
        or start.get('config',{}).get('tranche_tokens')!=target or start['config'].get('stage_tokens')!=target
        or finish.get('finish_cursor')!=transport.cursor_state(store)):return False
    for kind,name in [('cached_acquisition_started','start.json'),('cached_acquisition_finished','finish.json')]:
        expected=repair.common.digest(folder/name)
        events=[json.loads(r[0]) for r in store.db.execute('SELECT detail FROM events WHERE kind=? ORDER BY id DESC LIMIT 1',(kind,))]
        if len(events)!=1 or events[0].get('id')!=n or events[0].get('sha256')!=expected:return False
    width=config['sequence_len']+1
    retained=sum(json.loads(r[0])['prediction_targets'] for r in store.db.execute('SELECT manifest FROM shards'))
    required=plan['case']['effective_horizon_targets']
    return retained+(scanned.pending_positions(store)//width)*(width-1)>=required


def authorize_pack(store):
    if store.get('repair_final_pack_started',False):return
    if not acquisition_finished(store):raise ValueError('Final target boundary and sufficient finite supply required before suffix packing')
    with store.db:
        store.set('repair_final_pack_started',True)
        store.event('repair_final_pack_authorized',{'transport_run':store.get('cached_transport_run_sequence')})


def ordered_documents(store):
    plan=repair.common.read_json(store.directory/repair.PLAN)
    partial=store.db.execute('SELECT d.* FROM documents d JOIN groups g ON g.id=d.group_id WHERE d.tokens IS NOT NULL AND g.blocked=0 AND d.consumed>0').fetchall()
    if len(partial)>1:raise ValueError('More than one partial document')
    yield from partial
    rank=-1
    while True:
        rows=store.db.execute('SELECT d.*,r.rank AS repair_rank FROM repair_order r JOIN documents d ON d.id=r.document_id JOIN groups g ON g.id=d.group_id WHERE r.rank>? AND d.tokens IS NOT NULL AND d.consumed=0 AND g.blocked=0 ORDER BY r.rank LIMIT 64',(rank,)).fetchall()
        if not rows:break
        for row in rows:rank=row['repair_rank'];yield row
    last=b'',0
    while True:
        rows=store.db.execute('SELECT d.* FROM documents d JOIN groups g ON g.id=d.group_id WHERE d.id>? AND d.tokens IS NOT NULL AND d.consumed=0 AND g.blocked=0 AND (d.order_key,d.id)>(?,?) ORDER BY d.order_key,d.id LIMIT 64',(plan['original_document_id_ceiling'],*last)).fetchall()
        if not rows:break
        for row in rows:last=row['order_key'],row['id'];yield row


def _pack_chunk(store,rows,width,stop,*,fault=None):
    import numpy as np
    positions=rows*width
    # Reserve the complete output even on sparse-file filesystems, before creation.
    store.disk_check(positions*2+dirty_reservation(store,POLICY["documents_per_guard"]))
    sid=store.db.execute("SELECT COALESCE(MAX(id),-1)+1 FROM shards").fetchone()[0]
    paths=[f"shards/train_{sid:05d}.npy",f"shards/train_{sid:05d}.provenance.jsonl.gz"]
    if any((store.directory/(p+suffix)).exists() for p in paths for suffix in ("",".tmp")):
        raise FileExistsError("Refusing unrelated shard overwrite")
    b.atomic_json(store.directory/"packing.json",{"id":sid,"paths":paths})
    output,provenance=[store.directory/(p+".tmp") for p in paths]
    allocation=AllocationGuard(store,paths)
    result=None;flat=None;consumed=0;counts=Counter()
    try:
        result=np.lib.format.open_memmap(output,mode="w+",dtype="<u2",shape=(rows,width));flat=result.reshape(-1)
        documents=iter(ordered_documents(store));store.db.execute("BEGIN IMMEDIATE")
        with gzip.open(provenance,"wt",encoding="utf-8") as trace:
            while consumed<positions:
                stop.check();batch=[];planned=0
                for _ in range(POLICY["documents_per_guard"]):
                    try:doc=next(documents)
                    except StopIteration:break
                    batch.append(doc);planned+=min(doc["positions"]-doc["consumed"],positions-consumed-planned)
                    if consumed+planned==positions:break
                if not batch:raise ValueError("Packing supply changed unexpectedly")
                # Includes token output not allocated yet, gzip worst-case raw text,
                # overflow payloads and random page churn before any dirty write.
                output_unallocated=max(0,positions*2+128-output.stat().st_blocks*512)
                payload=sum(len(d["tokens"])+len(d["provenance"].encode())+1024 for d in batch)
                allocation.check(output_unallocated+dirty_reservation(store,len(batch),payload))
                updates=[]
                for doc in batch:
                    start=doc["consumed"];end=min(doc["positions"],start+positions-consumed)
                    tokens=np.frombuffer(zlib.decompress(doc["tokens"]),dtype="<u2")
                    if len(tokens)!=doc["positions"]:raise ValueError("Stored token payload length mismatch")
                    flat[consumed:consumed+end-start]=tokens[start:end]
                    counts[doc["component"]]+=end-start-int(start==0)
                    record=json.loads(doc["provenance"])
                    record.update(document_id=doc["id"],group_id=doc["group_id"],document_position_start=start,
                        document_position_stop=end,shard_position_start=consumed,shard_position_stop=consumed+end-start)
                    trace.write(b.base.canonical_json(record)+"\n")
                    updates.append((doc["id"],end,doc["group_id"]))
                    consumed+=end-start
                # Output order is unchanged; apply these independent row updates
                # in primary-key order for modest page locality within each batch.
                for doc_id,end,group_id in sorted(updates):
                    store.db.execute("UPDATE documents SET consumed=?,tokens=CASE WHEN ?=positions THEN NULL ELSE tokens END WHERE id=?",(end,end,doc_id))
                    store.db.execute("UPDATE groups SET packed=1 WHERE id=?",(group_id,))
                if fault:fault("after_batch",sid)
            stop.check()
        result.flush();flat=None;result=None
        for path in (output,provenance):
            with path.open("rb") as f:os.fsync(f.fileno())
        manifest={"id":sid,"rows":rows,"row_width":width,"dtype":"<u2","encoded_positions":positions,
            "ordinary_positions":sum(counts.values()),"prediction_targets":rows*(width-1),
            "ordinary_positions_by_component":dict(counts),"wrapping":False,
            "hashes":{paths[0]:b.source.file_hash(output),paths[1]:b.source.file_hash(provenance)}}
        allocation.check(dirty_reservation(store,1))
        for name,n in counts.items():store.db.execute("UPDATE components SET packed_ordinary=packed_ordinary+? WHERE name=?",(n,name))
        store.db.execute("INSERT INTO shards VALUES(?,?)",(sid,b.base.canonical_json(manifest)))
        for path,relative in zip((output,provenance),paths):os.replace(path,store.directory/relative)
        b.sync_directory(store.directory/"shards")
        if fault:fault("after_rename",sid)
        store.db.commit()
        if fault:fault("after_commit",sid)
        (store.directory/"packing.json").unlink();b.sync_directory(store.directory)
        if fault:fault("after_journal_removal",sid)
        checkpoint(store);store.snapshot()
        return manifest
    except BaseException:
        flat=None;result=None
        store.db.rollback()
        raise


def pack(store,*,max_positions=None,stop=None,fault=None):
    verify_policy(store);authorize_pack(store);store.ready();b.recover_pack(store)
    # A committed journal may have been recovered after publication interruption.
    # Publish before any verify and retain already committed shard identities.
    store.snapshot();checkpoint(store);b.verify(store)
    stop=repair.as_stop(stop);config=store.get("config");width=config["sequence_len"]+1
    if max_positions is not None and max_positions<=0:raise ValueError("max-positions must be positive")
    cap=min(POLICY["max_shard_positions"],config["shard_positions"],POLICY["max_shard_positions"] if max_positions is None else max_positions)
    if cap<width:raise ValueError("Bounded shard cap must fit one sequence row")
    old_cache=store.db.execute("PRAGMA cache_size").fetchone()[0]
    store.db.execute(f"PRAGMA cache_size=-{POLICY['sqlite_cache_bytes']//1024}")
    started=time.monotonic();cpu=time.process_time();before=len(store.snapshot()["shards"])
    outcome="complete";error=None
    with store.db:
        store.set("packing_incomplete",True)
        store.event("bounded_pack_started",{"max_positions":cap,"committed_shards_before":before})
    try:
        while True:
            stop.check()
            available=scanned.pending_positions(store)
            rows=min(cap//width,available//width)
            if not rows:break
            _pack_chunk(store,rows,width,stop,fault=fault)
    except PackStop as exc:outcome=exc.reason
    except Exception as exc:
        error=exc
        outcome="resource_limit" if isinstance(exc,RuntimeError) and ("reserve" in str(exc) or "allocation cap" in str(exc)) else "error"
    finally:
        store.db.rollback()
        # Recovery validates a committed rename, or removes only this journal's
        # uncommitted files. Never destroy a previous committed shard on a stop.
        try:
            try:b.recover_pack(store);checkpoint(store)
            except Exception as exc:
                outcome="recovery_required";error=exc
            if error is not None and isinstance(error,ValueError):
                store.set("status","blocked_packing_integrity")
            receipt={"outcome":outcome,"error":None if error is None else f"{type(error).__name__}: {error}",
                "wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu,
                "committed_shards_before":before,"committed_shards_after":store.db.execute("SELECT COUNT(*) FROM shards").fetchone()[0],
                "max_positions":cap,"policy":POLICY}
            if outcome=="complete":store.set("packing_incomplete",False)
            store.event("bounded_pack_finished",receipt);store.set("last_bounded_pack",receipt);store.db.commit()
            result=store.snapshot()
        finally:store.db.execute(f"PRAGMA cache_size={old_cache}")
    if error is not None and outcome in ("error","recovery_required"):raise error
    return {"outcome":outcome,"receipt":receipt,"manifest":result}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--max-positions',type=int)
    a=p.parse_args(argv)
    with Stop(a.max_seconds) as stop:
        store=scanned.ScannedStore(a.directory)
        try:
            result=pack(store,max_positions=a.max_positions,stop=stop)
            print(json.dumps(result['receipt'],indent=2));return 0 if result['outcome']=='complete' else 75
        finally:store.close()


if __name__=='__main__':raise SystemExit(main())
