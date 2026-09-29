"""Additive cached acquisition driver; frozen original builder is not edited.

Run only after the prior preparation process exits. Adopt this hashed transport
once before main training; preserve the original builder SHA, identity, source
cursors, tokenizer, heldout, token targets, byte limits, and pending documents.
No network starts without --allow-download. SIGINT/SIGTERM request a graceful
stop after the current counted HTTP request or committed batch boundary.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
from urllib.parse import quote

import prepare_main as builder
from main_range_cache import CachedRangeFile, MIB, RangeCache
import main_range_cache


class GracefulStop(Exception):pass


class StopController:
    def __init__(self):self.requested=False;self.previous={}
    def request(self,*unused):self.requested=True
    def check(self):
        if self.requested:raise GracefulStop("Stop requested; the preceding HTTP/batch boundary is durable")
    def __enter__(self):
        for signum in (signal.SIGINT,signal.SIGTERM):
            self.previous[signum]=signal.signal(signum,self.request)
        return self
    def __exit__(self,*unused):
        for signum,previous in self.previous.items():signal.signal(signum,previous)


def cursor_state(store):
    digest=hashlib.sha256();count=0
    for row in store.db.execute("SELECT file_id,rg,cursor,done FROM jobs ORDER BY file_id,rg"):
        digest.update(builder.base.canonical_json(dict(row)).encode()+b"\n");count+=1
    return {"job_count":count,"jobs_cursor_sha256":digest.hexdigest(),
        "last_document_id":store.db.execute("SELECT COALESCE(MAX(id),0) FROM documents").fetchone()[0],
        "components":store.totals(),"downloaded_bytes":store.get("downloaded_bytes",0)}


def recorded_etags(store):
    """Carry strong validators across transport changes and process restarts."""
    result={};path=store.directory/"network.jsonl"
    if not path.exists():return result
    with path.open() as f:
        for line in f:
            row=json.loads(line)
            if row.get("status")!=206:continue
            etag=row.get("etag");url=row["url"]
            if not etag or etag.startswith("W/") or not(etag.startswith('"') and etag.endswith('"')):
                raise ValueError("Prior completed source range lacks a strong ETag")
            if url in result and result[url]!=etag:raise ValueError("Prior source receipts contain conflicting ETags")
            result[url]=etag
    return result


def adopt_transport(store,cache_config):
    store.ready()
    if store.get("network_inflight"):raise RuntimeError("Prior network bytes remain unreconciled; no cached takeover")
    lineage=store.get("lineage")
    if builder.source.file_hash(Path(builder.__file__))!=lineage["builder_sha256"]:
        raise ValueError("Original builder changed; additive driver cannot relabel its lineage")
    for module in [builder.base,builder.source]:
        if builder.source.file_hash(Path(module.__file__))!=lineage["helper_sha256"][Path(module.__file__).name]:
            raise ValueError("Frozen helper changed")
    active=lineage.get("active_transport")
    identity={"driver":"cached-main-acquisition-v1","driver_sha256":builder.source.file_hash(Path(__file__)),
        "cache_sha256":builder.source.file_hash(Path(main_range_cache.__file__)),"cache_config":cache_config}
    if active is not None:
        if any(active.get(key)!=value for key,value in identity.items()):raise ValueError("Pinned active transport changed; coordinated training contract transition required")
        return active
    directory=store.directory/"transport"
    # Initialization of an epoch is intentionally not guessed/retried after an
    # interrupted partial adoption; an existing directory requires inspection.
    paths=[Path(__file__),Path(main_range_cache.__file__),Path(builder.__file__),
           Path(builder.base.__file__),Path(builder.source.__file__)]
    store.disk_check(sum(path.stat().st_size for path in paths)+16*MIB)
    directory.mkdir(parents=True,exist_ok=False)
    hashes={}
    for path in paths:
        destination=directory/path.name;shutil.copyfile(path,destination)
        hashes[str(destination.relative_to(store.directory))]=builder.source.file_hash(destination)
    record={"schema_version":1,**identity,
        "adopted_at_utc":datetime.now(timezone.utc).isoformat(),"cutover_cursor":cursor_state(store),
        "source_hashes":hashes,"config_at_adoption":store.get("config"),
        "semantics":"same pinned file/rowgroup schedule, adapter, BPE, dedup, exclusions and token quota; transport-only change"}
    plan=directory/"plan.json";builder.atomic_json(plan,record)
    plan_path=str(plan.relative_to(store.directory));plan_hash=builder.source.file_hash(plan)
    active={**identity,"plan_path":plan_path,"plan_sha256":plan_hash,"source_snapshot_hashes":hashes}
    lineage["active_transport"]=active
    lineage.setdefault("metadata_artifact_hashes",{}).update({**hashes,plan_path:plan_hash})
    with store.db:
        store.set("lineage",lineage);store.event("active_transport_adopted",active)
    store.snapshot();return active


def begin_run(store,active,etags):
    number=store.get("cached_transport_run_sequence",0)+1
    directory=store.directory/"transport"/"runs"/f"{number:05d}"
    store.disk_check(MIB);directory.mkdir(parents=True,exist_ok=False)
    path=directory/"start.json"
    builder.atomic_json(path,{"schema_version":1,"id":number,"started_at_utc":datetime.now(timezone.utc).isoformat(),
        "active_transport_plan_sha256":active["plan_sha256"],"start_cursor":cursor_state(store),
        "prior_etag_count":len(etags),"prior_etags_sha256":builder.base.digest(builder.base.canonical_json(etags)),
        "config":store.get("config"),
        "versions":{name:importlib.metadata.version(name) for name in ["numpy","pyarrow","tiktoken","requests"]}})
    with store.db:
        store.set("cached_transport_run_sequence",number)
        store.event("cached_acquisition_started",{"id":number,"record":str(path.relative_to(store.directory)),
            "sha256":builder.source.file_hash(path)})
    return number


def finish_run(store,number,outcome,stats):
    path=store.directory/"transport"/"runs"/f"{number:05d}"/"finish.json"
    if path.exists():raise FileExistsError("Code epoch completion already exists")
    builder.atomic_json(path,{"id":number,"outcome":outcome,"finished_at_utc":datetime.now(timezone.utc).isoformat(),
        "finish_cursor":cursor_state(store),"cache_metrics":stats})
    with store.db:
        store.event("cached_acquisition_finished",{"id":number,"outcome":outcome,
            "record":str(path.relative_to(store.directory)),"sha256":builder.source.file_hash(path)})
    return store.snapshot()


def run(store,*,block_bytes=MIB,cache_bytes=128*MIB,max_fetch_bytes=32*MIB,stop=None):
    builder.verify(store)
    config=dict(block_bytes=block_bytes,max_bytes=cache_bytes,max_fetch_bytes=max_fetch_bytes,max_read_bytes=64*MIB)
    # Validate configuration before any code adoption or possible network call.
    RangeCache(None,**config)
    stop=stop or StopController();etags=recorded_etags(store)
    active=adopt_transport(store,config);number=begin_run(store,active,etags);caches={}
    original_open=builder.open_main_parquet;original_get=builder.TrackedHTTP.get;original_ingest=builder.ingest_batch
    def checked_get(http,*args,**kwargs):
        stop.check();result=original_get(http,*args,**kwargs);stop.check();return result
    def checked_ingest(*args,**kwargs):
        stop.check();result=original_ingest(*args,**kwargs);stop.check();return result
    def cached_open(http,pinned,entry):
        import pyarrow.parquet as pq
        stop.check()
        cache=caches.setdefault(id(http),None)
        if cache is None:
            cache=RangeCache(http,**config);cache.etags.update(etags);caches[id(http)]=cache
        url=f"https://huggingface.co/datasets/{pinned['repo']}/resolve/{pinned['revision']}/{quote(entry['path'],safe='/')}"
        return pq.ParquetFile(CachedRangeFile(cache,url,entry["size"]),pre_buffer=False)
    builder.open_main_parquet=cached_open;builder.TrackedHTTP.get=checked_get;builder.ingest_batch=checked_ingest
    outcome="failed"
    try:
        stop.check();builder.acquire(store);outcome="tranche_stopped_at_target_or_document_boundary"
    except GracefulStop:outcome="graceful_stop"
    finally:
        builder.open_main_parquet=original_open;builder.TrackedHTTP.get=original_get;builder.ingest_batch=original_ingest
        result=finish_run(store,number,outcome,[cache.stats() for cache in caches.values()])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True);p.add_argument("--allow-download",action="store_true")
    p.add_argument("--block-mib",type=int,default=1);p.add_argument("--cache-mib",type=int,default=128)
    p.add_argument("--max-fetch-mib",type=int,default=32)
    a=p.parse_args(argv)
    if not a.allow_download:p.error("Explicit --allow-download required; do not overlap an existing preparer")
    store=builder.Store(a.directory)
    try:
        with StopController() as stop:
            value=run(store,block_bytes=a.block_mib*MIB,cache_bytes=a.cache_mib*MIB,max_fetch_bytes=a.max_fetch_mib*MIB,stop=stop)
        print(json.dumps({"status":value["status"],"components":value["components"],"downloaded_response_bytes":value["downloaded_response_bytes"]},indent=2))
    finally:store.close()


if __name__=="__main__":main()
