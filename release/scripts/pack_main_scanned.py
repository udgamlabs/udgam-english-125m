"""V3 bounded pack loop with exact sequential pending aggregation.

All output, transaction, disk, signal and recovery semantics are inherited from
source-pinned v2 helpers. Only full pending SUM/snapshot query plans differ.
"""
import argparse
import json
from pathlib import Path
import time
import pack_main_bounded as bounded
import main_scanned_store as scanned
from pack_main_bounded import Stop,PackStop,checkpoint,_pack_chunk,POLICY

b=bounded.b


def verify_policy(store):
    bounded.verify_policy(store)
    active=store.get("lineage").get("active_transport",{})
    if active.get("driver")!="cached-scanned-main-v3":raise ValueError("Explicit pre-training v3 adoption required")
    if active.get("scanned_store_sha256")!=b.source.file_hash(Path(scanned.__file__)):
        raise ValueError("Pinned v3 snapshot implementation changed")
    if active.get("scanned_packer_sha256")!=b.source.file_hash(Path(__file__)) or active.get("pending_query_policy")!=scanned.QUERY_POLICY:
        raise ValueError("Pinned v3 packer/query policy changed")


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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--max-seconds",type=float);p.add_argument("--max-positions",type=int)
    a=p.parse_args(argv)
    with Stop(a.max_seconds) as stop:
        store=scanned.ScannedStore(a.directory)
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
