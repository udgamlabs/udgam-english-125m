"""Main preparation v2: same ordered data semantics, fewer durable commits/scans.

Only transaction batching and progress-snapshot frequency differ. All adapters,
grouping, exact quotas, exclusion/conflict publication, BPE and range transport
come from the source-hashed original builder. No networking is implemented here.
"""
from collections import Counter
from contextlib import contextmanager
import json

import prepare_main as builder
import pack_main_bounded as bounded

BATCH_ROWS=1024


def reserve_batch(store, records):
    """Bound the whole next transaction, including pre-existing identity families.

    Existing families can be merged or excluded by a new candidate. Count those
    rows/payloads before mutation, rather than assuming only new text can dirty
    SQLite pages. A very large family fails closed under configured disk bounds.
    """
    keys=sorted({builder.hkey(key) for record in records for key in record["group_keys"]})
    roots=set()
    for offset in range(0,len(keys),400):
        batch=keys[offset:offset+400]
        roots.update(row[0] for row in store.db.execute(
            "SELECT DISTINCT group_id FROM identity WHERE key IN ("+",".join("?" for _ in batch)+")",batch))
    existing_rows=0;existing_bytes=0;identity_rows=0
    roots=sorted(roots)
    for offset in range(0,len(roots),400):
        batch=roots[offset:offset+400];marks=",".join("?" for _ in batch)
        row=store.db.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(length(tokens),0)+length(provenance)),0) FROM documents WHERE group_id IN ("+marks+")",batch).fetchone()
        existing_rows+=row[0];existing_bytes+=row[1]
        identity_rows+=store.db.execute("SELECT COUNT(*) FROM identity WHERE group_id IN ("+marks+")",batch).fetchone()[0]
    # Ordinary BPE cannot have more tokens than UTF-8 input bytes. Two bytes per
    # token + BOS, zlib overhead, and JSON/identity metadata are over-reserved.
    new_payload=sum(2*len(r["text"].encode())+len(builder.base.canonical_json(r).encode())+4096 for r in records)
    extra=bounded.dirty_reservation(store,len(records)+existing_rows+identity_rows,
        new_payload+existing_bytes+sum(len(key)+64 for key in keys))
    # Acquisition still uses the original builder's 64MiB cache: allow all dirty
    # pages to spill and be copied to the database during auto-checkpoint.
    extra+=2*(64*1024**2-bounded.POLICY["sqlite_cache_bytes"])
    store.disk_check(extra)
    return extra


@contextmanager
def guarded_family_mutations(store):
    """Recheck actual allocation before every existing-family merge/exclusion.

    A chain may repeatedly re-dirty the same large family within one transaction;
    the single pre-batch estimate intentionally does not pretend to cover that.
    """
    original=store.group
    def group(keys,blocked=0):
        roots=sorted({row[0] for key in {builder.hkey(k) for k in keys}
            for row in store.db.execute("SELECT group_id FROM identity WHERE key=?",(key,))})
        states=[store.db.execute("SELECT blocked,packed FROM groups WHERE id=?",(root,)).fetchone() for root in roots]
        if len(roots)>1 or (roots and (blocked or any(r["blocked"] for r in states))):
            if any(r["packed"] for r in states) and (blocked or any(r["blocked"] for r in states)):
                # Preserve the original immediate blocked_identity_conflict path,
                # even if a resource guard would otherwise reject the operation.
                return original(keys,blocked)
            rows=0;payload=0;identities=0
            for offset in range(0,len(roots),400):
                batch=roots[offset:offset+400];marks=",".join("?" for _ in batch)
                row=store.db.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(length(tokens),0)+length(provenance)),0) FROM documents WHERE group_id IN ("+marks+")",batch).fetchone()
                rows+=row[0];payload+=row[1]
                identities+=store.db.execute("SELECT COUNT(*) FROM identity WHERE group_id IN ("+marks+")",batch).fetchone()[0]
            store.disk_check(bounded.dirty_reservation(store,rows+identities,payload)+96*1024**2)
        return original(keys,blocked)
    store.group=group
    try:yield
    finally:store.group=original


def acquire(store,http,cache):
    store.ready()
    if store.get("packing_incomplete",False) or (store.directory/"packing.json").exists():
        raise RuntimeError("Previous tranche packing is incomplete; finish bounded packing before acquiring fresh documents")
    config=store.get("config")
    codec=builder.Codec(store.directory/"tokenizer.json",store.get("lineage")["tokenizer_sha256"])
    deny=builder.benchmark_deny(store);lock=store.get("source_lock")
    initial=store.totals();packed=sum(x["packed"] for x in initial.values())
    target=min(config["stage_tokens"],packed+config["tranche_tokens"])
    while sum(x["accepted"] for x in store.totals().values())<target:
        store.disk_check(128*1024**2);totals=store.totals()
        component=min(totals,key=lambda c:(totals[c]["accepted"]/totals[c]["weight"],c))
        job=store.db.execute("SELECT jobs.*,files.component,files.entry FROM jobs JOIN files ON jobs.file_id=files.id WHERE jobs.done=0 AND files.component=? ORDER BY files.rank,jobs.rank LIMIT 1",(component,)).fetchone()
        if job is None:
            file=store.db.execute("SELECT * FROM files WHERE component=? AND scanned=0 ORDER BY rank LIMIT 1",(component,)).fetchone()
            if file is None:raise RuntimeError(f"Finite pinned source exhausted for {component}; no replacement")
            name=builder.source.COMPONENTS[component][0];pinned=lock["sources"][name]
            parquet=cache.get(file["id"],pinned,json.loads(file["entry"]))
            builder.register_groups(store,file["id"],parquet);continue
        file_id=job["file_id"];name,config_name,_,_=builder.source.COMPONENTS[component];pinned=lock["sources"][name]
        parquet=cache.get(file_id,pinned,json.loads(job["entry"]))
        wanted=("text","id","url","language") if name=="fineweb_edu" else ("text","id","url","rollout_results") if name=="finephrase" else ("text","uuid","license","metadata")
        columns=[c for c in wanted if c in parquet.schema_arrow.names]
        position=0
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS,row_groups=[job["rg"]],columns=columns,use_threads=False):
            records=[];rejections=[];end=position+batch.num_rows
            if end<=job["cursor"]:position=end;continue
            for relative,row in enumerate(batch.to_pylist(),position):
                if relative<job["cursor"]:continue
                candidate={"source":name,"config":config_name,"row_index":job["start_row"]+relative,"row":row,
                    "viewer_partial":False,"viewer_indexed_rows":parquet.metadata.num_rows,"truncated_cells":[]}
                try:record=builder.base.adapt(candidate,pinned,min_words=20 if name=="nemotron_fact" else 30)
                except builder.base.Rejected as exc:
                    rejections.append((relative,"adapter_rejected:"+str(exc)));continue
                record.update(component=component,file_id=file_id,source_row_group=job["rg"],source_row_in_group=relative)
                records.append(record)
            # The original transaction and per-document quota boundary remain
            # authoritative. It still publishes blocked identity conflicts at once.
            reserve_batch(store,records)
            with guarded_family_mutations(store):
                outcome=builder.ingest_batch(store,records,codec,deny,(file_id,job["rg"]),end,end==job["rows"],target,rejections)
            position=end
            if outcome["limit_reached"]:return store.snapshot()
            if sum(x["accepted"] for x in store.totals().values())>=target:break
        # Full pending-document aggregation is deferred until the tranche ends.
        # This progress event carries only O(number-of-components) counters.
        print(json.dumps({"accepted_ordinary":sum(x["accepted"] for x in store.totals().values()),
            "target":target,"download_bytes":store.get("downloaded_bytes"),"engine":"batched-main-v2"}),flush=True)
    return store.snapshot()
