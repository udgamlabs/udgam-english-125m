"""Finite small-shard packing with bounded dirty work and graceful recovery.

This additive v2 implementation preserves the original token/document stream.
It must be pinned by explicit pre-training v2 adoption before CLI use. Limits
may stop a run with pending data; complete packing before acquiring a new tranche.
"""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import signal
import shutil
import time
import zlib

import prepare_main as b

MIB=1024**2
POLICY={"max_shard_positions":10_000_000,"documents_per_guard":16,
    "sqlite_cache_bytes":16*MIB,"dirty_cache_reserve_factor":2,
    "btree_pages_per_level_per_document":12,"btree_base_pages_per_document":16,
    "metadata_reserve_bytes":2*MIB,"checkpoint_each_shard":True,
    "partial_chunk_on_stop":"rollback_current_only","stream_order":"original_pending_documents","acquire_while_pack_incomplete":"forbidden","allocation_scan":"immutable_once_per_chunk_mutable_each_batch",
    "manifest_temp_reserve":"twice_current_plus_metadata_margin"}


class PackStop(RuntimeError):
    def __init__(self,reason):self.reason=reason;super().__init__(reason)


class Stop:
    def __init__(self,max_seconds=None):
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=0):raise ValueError("max-seconds must be positive and finite")
        self.deadline=None if max_seconds is None else time.monotonic()+max_seconds
        self.signal=None;self.previous={}
    def check(self):
        if self.signal is not None:raise PackStop("signal")
        if self.deadline is not None and time.monotonic()>=self.deadline:raise PackStop("time_limit")
    def __enter__(self):
        for number in (signal.SIGINT,signal.SIGTERM):
            self.previous[number]=signal.getsignal(number)
            signal.signal(number,lambda signum,frame:setattr(self,"signal",signum))
        return self
    def __exit__(self,*args):
        for number,handler in self.previous.items():signal.signal(number,handler)


def dirty_reservation(store,documents,payload_bytes=0):
    """Conservative random B-tree churn + overflow pages + unflushed cache.

    The height bound uses binary branching (SQLite fanout is usually far larger).
    Each document covers table, partial/pending indexes and group mutation. This
    is an explicit engineering reserve, not a filesystem quota guarantee against
    unrelated writers. Actual allocation/free-space is checked before every batch.
    """
    page=store.db.execute("PRAGMA page_size").fetchone()[0]
    pages=store.db.execute("PRAGMA page_count").fetchone()[0]
    levels=max(1,math.ceil(math.log2(pages+1)))
    manifest=store.directory/"manifest.json"
    manifest_extra=2*manifest.stat().st_size if manifest.exists() else 0
    return (2*POLICY["sqlite_cache_bytes"]+POLICY["metadata_reserve_bytes"]+manifest_extra+
        documents*(16+12*levels)*page+2*payload_bytes)


class AllocationGuard:
    """Exact mutable-file allocation, with immutable files counted once.

    The exclusive preparation process.lock forbids another stage writer. Trainer
    checkpoints must live outside the stage; global filesystem free space remains
    live and is checked every batch, including unrelated external allocation.
    """
    def __init__(self,store,relative_paths):
        self.store=store;self.config=store.get("config")
        names=["state.sqlite","state.sqlite-wal","state.sqlite-shm","packing.json","packing.json.tmp",
            "manifest.json","manifest.json.tmp","manifest.sha256","manifest.sha256.tmp"]
        names += [p+suffix for p in relative_paths for suffix in ("",".tmp")]
        self.paths=[store.directory/name for name in names]
        mutable=set(self.paths)
        self.fixed=sum(path.stat().st_blocks*512 for path in store.directory.rglob("*") if path.is_file() and path not in mutable)
    def check(self,extra=0):
        if shutil.disk_usage(self.store.directory).free<self.config.get("disk_reserve_bytes",10*1024**3)+extra:
            raise RuntimeError("Disk reserve would be violated; stop bounded packing")
        allocated=self.fixed+sum(path.stat().st_blocks*512 for path in self.paths if path.exists())
        if allocated+extra>self.config.get("max_output_bytes",30_000_000_000):
            raise RuntimeError("Output-directory allocation cap would be violated during bounded packing")


def checkpoint(store):
    if store.db.in_transaction:raise RuntimeError("Cannot checkpoint an active transaction")
    value=tuple(store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
    if value!=(0,0,0):raise RuntimeError(f"WAL checkpoint did not truncate: {value}")


def verify_policy(store):
    lineage=store.get("lineage")
    if lineage.get("builder_sha256")!=b.source.file_hash(Path(b.__file__)):raise ValueError("Original builder changed")
    for module in (b.base,b.source):
        if lineage.get("helper_sha256",{}).get(Path(module.__file__).name)!=b.source.file_hash(Path(module.__file__)):
            raise ValueError("Frozen helper changed")
    active=lineage.get("active_transport",{})
    if active.get("packer_sha256")!=b.source.file_hash(Path(__file__)) or active.get("packer_policy")!=POLICY:
        raise ValueError("Bounded packer source/policy is not pinned by active v2 adoption")


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
        documents=iter(b.pending_documents(store));store.db.execute("BEGIN IMMEDIATE")
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
    store.ready();b.recover_pack(store)
    # A committed journal may have been recovered after publication interruption.
    # Publish before any verify and retain already committed shard identities.
    store.snapshot();checkpoint(store)
    stop=stop or Stop();config=store.get("config");width=config["sequence_len"]+1
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
            available=store.db.execute("SELECT COALESCE(SUM(d.positions-d.consumed),0) FROM documents d JOIN groups g ON d.group_id=g.id WHERE g.blocked=0 AND d.tokens IS NOT NULL").fetchone()[0]
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--max-seconds",type=float);p.add_argument("--max-positions",type=int)
    a=p.parse_args(argv)
    with Stop(a.max_seconds) as stop:
        store=b.Store(a.directory)
        try:
            verify_policy(store)
            if store.get("packing_incomplete",False) or (store.directory/"packing.json").exists():
                b.recover_pack(store)
                # After journal removal but before public manifest replacement,
                # the durable incomplete flag still authorizes rebuilding the view.
                for row in store.db.execute("SELECT manifest FROM shards ORDER BY id"):
                    for relative,checksum in json.loads(row[0])["hashes"].items():
                        if b.source.file_hash(store.directory/relative)!=checksum:raise ValueError("Committed shard integrity failure")
                store.snapshot()
            b.verify(store)
            result=pack(store,max_positions=a.max_positions,stop=stop)
            print(json.dumps({"outcome":result["outcome"],"receipt":result["receipt"],
                "remaining_pending_positions":result["manifest"]["remaining_pending_positions"]},indent=2))
            return 0 if result["outcome"]=="complete" else 75
        finally:store.close()


if __name__=="__main__":raise SystemExit(main())
