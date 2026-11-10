#!/usr/bin/env python3
"""Resumable, finite main-data preparation. Network only in explicit acquire action.

The validation split is the frozen pilot validation, not a moving random sample.
All pilot identity families are excluded from new training supply. SQLite owns
global grouping, deduplication, source cursors, and committed shard identities.
Only locally trained tokenizer artifacts are loaded; no model weights are used.

Run init once, then alternate acquire --allow-download and pack for each tranche.
The immutable stage cap is fixed at init; another acquire cannot raise that cap.
Keep state.sqlite, its WAL, metadata, and shards together when moving machines.
Preparation is single-process. Initialization interruption requires a new output
directory; acquisition and packing are resumable. Measure CPU preparation speed
before committing paid GPU time. The trainer must consume these shards once.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import gzip
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import zlib

import prepare_english as base
import prepare_production as source

ROOT = Path(__file__).resolve().parents[1]
PILOT_TRAIN, PILOT_VAL, BENCHMARK = 1, 2, 4


class IdentityConflict(RuntimeError): pass


def sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def hkey(value): return hashlib.sha256(value.encode()).digest()


def encoded_bytes(ids):
    if not ids or min(ids) < 0 or max(ids) > 65535:
        raise ValueError("Token IDs must be nonempty uint16 values")
    values = array("H", ids)
    if sys.byteorder != "little": values.byteswap()
    return values.tobytes()


class Codec:
    """Exact pilot policy: encode_ordinary(text), then one explicit BOS.

    The pinned nanochat RustBPETokenizer.encode uses encode_ordinary too. Literal
    strings such as <|bos|> inside source text remain ordinary text, not control
    IDs. No policy change, pickle loading, pretrained weights, or tokenizer fit.
    """
    def __init__(self, path, expected_hash):
        import tiktoken
        import base64
        if source.file_hash(path) != expected_hash: raise ValueError("Owned tokenizer hash mismatch")
        value = json.loads(path.read_text())
        if value.get("format") != "udgam-owned-rustbpe-v1": raise ValueError("Unrecognized tokenizer lineage")
        self.enc = tiktoken.Encoding(name="udgam-main-owned-bpe", pat_str=value["pattern"],
            mergeable_ranks={base64.b64decode(k): v for k,v in value["mergeable_ranks"]},
            special_tokens=value["special_tokens"])
        self.bos = value["special_tokens"][value["bos_token"]]
        if self.enc.n_vocab != 16384: raise ValueError("Exact owned16K tokenizer required for checkpoint compatibility")
    def encode(self, text): return [self.bos, *self.enc.encode_ordinary(text)]


class Store:
    def __init__(self, directory, create=False):
        self.directory = Path(directory)
        if create: self.directory.mkdir(parents=True, exist_ok=False)
        elif not (self.directory / "state.sqlite").exists(): raise ValueError("Missing initialized main-data state")
        self.lock = (self.directory / "process.lock").open("a")
        try: fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another preparation process owns this directory") from None
        self.db = sqlite3.connect(self.directory / "state.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA cache_size=-65536")
        if create:
            self.db.executescript("""
            CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE groups(id INTEGER PRIMARY KEY,blocked INTEGER NOT NULL DEFAULT0,
                                packed INTEGER NOT NULL DEFAULT0);
            CREATE TABLE identity(key BLOB PRIMARY KEY,group_id INTEGER NOT NULL);
            CREATE INDEX identity_group ON identity(group_id);
            CREATE TABLE components(name TEXT PRIMARY KEY,weight REAL NOT NULL,
                                    accepted_ordinary INTEGER NOT NULL DEFAULT0,
                                    packed_ordinary INTEGER NOT NULL DEFAULT0);
            CREATE TABLE files(id INTEGER PRIMARY KEY,component TEXT NOT NULL,entry TEXT NOT NULL,
                               rank BLOB NOT NULL,scanned INTEGER NOT NULL DEFAULT0);
            CREATE INDEX files_order ON files(component,scanned,rank);
            CREATE TABLE jobs(file_id INTEGER NOT NULL,rg INTEGER NOT NULL,rows INTEGER NOT NULL,
                              start_row INTEGER NOT NULL,cursor INTEGER NOT NULL DEFAULT0,
                              rank BLOB NOT NULL,done INTEGER NOT NULL DEFAULT0,PRIMARY KEY(file_id,rg));
            CREATE INDEX jobs_pending ON jobs(done,file_id,rank);
            CREATE TABLE documents(id INTEGER PRIMARY KEY,fingerprint BLOB UNIQUE NOT NULL,
                group_id INTEGER NOT NULL,component TEXT NOT NULL,ordinary INTEGER NOT NULL,
                positions INTEGER NOT NULL,consumed INTEGER NOT NULL DEFAULT0,tokens BLOB,
                provenance TEXT NOT NULL,order_key BLOB NOT NULL);
            CREATE INDEX documents_group ON documents(group_id);
            CREATE INDEX documents_order ON documents(order_key);
            CREATE INDEX documents_pending_order ON documents(order_key) WHERE tokens IS NOT NULL;
            CREATE INDEX documents_partial ON documents(consumed) WHERE tokens IS NOT NULL AND consumed>0;
            CREATE TABLE shards(id INTEGER PRIMARY KEY,manifest TEXT NOT NULL);
            CREATE TABLE events(id INTEGER PRIMARY KEY,created TEXT NOT NULL,kind TEXT NOT NULL,detail TEXT NOT NULL);
            """.replace("DEFAULT0", "DEFAULT 0"))
            self.set("schema", 1); self.set("status", "initializing")
            (self.directory / "shards").mkdir()
            self.db.commit()
    def close(self):
        self.db.close()
        self.lock.close()
    def resolve(self,path):
        return Path(path) if Path(path).is_absolute() else self.directory/path
    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default
    def set(self, key, value):
        self.db.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
    def event(self, kind, detail):
        self.db.execute("INSERT INTO events(created,kind,detail) VALUES(?,?,?)", (datetime.now(timezone.utc).isoformat(),kind,json.dumps(detail)))
    def disk_check(self, extra=0):
        config=self.get("config", {})
        reserve = config.get("disk_reserve_bytes", 10 * 1024**3)
        if shutil.disk_usage(self.directory).free < reserve + extra:
            raise RuntimeError("Disk reserve would be violated; stop without downloading or packing more")
        maximum=config.get("max_output_bytes",30_000_000_000)
        allocated=sum(p.stat().st_blocks*512 for p in self.directory.rglob("*") if p.is_file())
        if allocated+extra>maximum:raise RuntimeError("Output-directory allocation cap would be violated")
    def ready(self):
        if self.get("status") != "initialized": raise RuntimeError("State is incomplete or blocked; inspect status/events")
    def totals(self):
        return {r["name"]: {"accepted": r["accepted_ordinary"], "packed": r["packed_ordinary"], "weight": r["weight"]}
                for r in self.db.execute("SELECT * FROM components ORDER BY name")}

    def group(self, keys, blocked=0):
        """Merge complete identity families transactionally before split decisions."""
        keys = sorted({hkey(k) for k in keys})
        if not keys: raise ValueError("Missing source identity keys")
        roots = {r[0] for k in keys for r in self.db.execute("SELECT group_id FROM identity WHERE key=?", (k,))}
        previous = [self.db.execute("SELECT * FROM groups WHERE id=?", (r,)).fetchone() for r in roots]
        blocked_final = blocked
        for r in previous: blocked_final |= r["blocked"]
        packed = any(r["packed"] for r in previous)
        if blocked_final and packed:
            raise IdentityConflict("A newly linked excluded family contains already-packed training; retire affected data/holdout explicitly before continuing")
        if not roots:
            root = self.db.execute("INSERT INTO groups(blocked,packed) VALUES(?,0)", (blocked_final,)).lastrowid
        else:
            root = min(roots)
            for old in roots - {root}:
                self.db.execute("UPDATE identity SET group_id=? WHERE group_id=?", (root,old))
                self.db.execute("UPDATE documents SET group_id=? WHERE group_id=?", (root,old))
                self.db.execute("DELETE FROM groups WHERE id=?", (old,))
            if blocked_final:
                # Packed conflicts were refused above. Remove newly excluded
                # pending families from both supply counters and payload storage.
                for r in self.db.execute("SELECT component,SUM(ordinary) AS n FROM documents WHERE group_id=? AND tokens IS NOT NULL GROUP BY component", (root,)).fetchall():
                    self.db.execute("UPDATE components SET accepted_ordinary=accepted_ordinary-? WHERE name=?", (r["n"],r["component"]))
                self.db.execute("UPDATE documents SET tokens=NULL WHERE group_id=?", (root,))
            self.db.execute("UPDATE groups SET blocked=?,packed=? WHERE id=?", (blocked_final,int(packed),root))
        self.db.executemany("INSERT OR IGNORE INTO identity VALUES(?,?)", [(k,root) for k in keys])
        return root, blocked_final

    def add(self, record, codec, blocked=0, ordinary_limit=None):
        root, excluded = self.group(record["group_keys"], blocked)
        if excluded: return "excluded_family", 0
        fingerprint = bytes.fromhex(record["normalized_text_sha256"])
        if self.db.execute("SELECT 1 FROM documents WHERE fingerprint=?", (fingerprint,)).fetchone(): return "duplicate", 0
        ids = codec.encode(record["text"])
        ordinary = len(ids) - 1
        if ordinary <= 0: return "empty_encoding", 0
        if ordinary_limit is not None and ordinary > ordinary_limit:return "limit_reached",0
        component = record["component"]
        provenance = {k:record.get(k) for k in ["source","component","source_id","source_url","config","revision",
            "source_row_sha256","text_sha256","normalized_text_sha256","source_document_identity_available",
            "source_row_group","source_row_in_group","row_index","file_id","row_license"]}
        self.db.execute("INSERT INTO documents(fingerprint,group_id,component,ordinary,positions,tokens,provenance,order_key) VALUES(?,?,?,?,?,?,?,?)",
            (fingerprint,root,component,ordinary,len(ids),zlib.compress(encoded_bytes(ids),1),base.canonical_json(provenance),
             hkey(f"{self.get('config')['seed']}:main-order:{record['text_sha256']}")))
        self.db.execute("UPDATE components SET accepted_ordinary=accepted_ordinary+? WHERE name=?", (ordinary,component))
        return "accepted", ordinary

    def snapshot(self):
        values = {"schema_version":1,"status":self.get("status"),"config":self.get("config"),
            "lineage":self.get("lineage"),"downloaded_response_bytes":self.get("downloaded_bytes",0),
            "network_request_unreconciled":self.get("network_inflight",False),"components":self.totals(),
            "shards":[json.loads(r[0]) for r in self.db.execute("SELECT manifest FROM shards ORDER BY id")],
            "remaining_pending_positions":self.db.execute("SELECT COALESCE(SUM(d.positions-d.consumed),0) FROM documents d JOIN groups g ON d.group_id=g.id WHERE g.blocked=0 AND d.tokens IS NOT NULL").fetchone()[0],
            "validation_policy":"frozen_pilot_validation_excluded_from_new_train",
            "limitations":["Available IDs only; Nemotron lacks original FineWiki article identity.",
                "Direct benchmark n-grams do not establish paraphrase/teacher contamination clearance.",
                "All pilot train families are conservatively excluded, including any not consumed by pilot.",
                "Any packed/excluded group linkage blocks this builder; no automatic holdout retirement.",
                "Frozen intrinsic validation is group-disjoint by available identities, not proven topic-disjoint.",
                "Main preparation throughput and finite multi-shard trainer integration require validation.",
                "Raw pages retain upstream rights; compact provenance supports pinned reacquisition."]}
        atomic_json(self.directory / "manifest.json", values)
        checksum=source.file_hash(self.directory/"manifest.json")
        temporary=self.directory/"manifest.sha256.tmp"
        with temporary.open("w") as f:f.write(checksum+"\n");f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.directory/"manifest.sha256");sync_directory(self.directory)
        return values


def initialize(directory, pilot_data, pilot_corpus, benchmark_dir, source_report, source_lock,
               *, tranche_tokens=250_000_000, stage_tokens=2_000_000_000, shard_positions=100_000_000,
               max_download_bytes=20_000_000_000, max_output_bytes=30_000_000_000,
               disk_reserve_bytes=10 * 1024**3, seed=20260922):
    if min(tranche_tokens,stage_tokens,shard_positions,max_download_bytes,max_output_bytes,disk_reserve_bytes) <= 0 or tranche_tokens > stage_tokens:
        raise ValueError("Invalid bounded stage configuration")
    pilot_data,pilot_corpus,benchmark_dir,source_report,source_lock = map(Path,(pilot_data,pilot_corpus,benchmark_dir,source_report,source_lock))
    data = json.loads((pilot_data / "manifest.json").read_text())
    if data.get("status")!="ready" or data.get("profile")!="production":raise ValueError("Completed production pilot data required")
    if source.file_hash(pilot_data / "manifest.json") != (pilot_data / "manifest.sha256").read_text().strip(): raise ValueError("Pilot token manifest hash mismatch")
    report = json.loads(source_report.read_text()); lock=json.loads(source_lock.read_text()); base.validate_lock(lock,"2026-09-22")
    if report.get("status")!="corpus_prepared_token_supply_unmeasured" or report.get("errors"):raise ValueError("Pilot corpus preparation is incomplete")
    if source.file_hash(source_report) != data["hashes"]["source-report.json"]: raise ValueError("Pilot source report mismatch")
    if source.file_hash(source_lock) != report["source_lock_sha256"]:raise ValueError("Pilot source lock mismatch")
    if report["source_revisions"] != {name:lock["sources"][name]["revision"] for name in report["source_revisions"]}:
        raise ValueError("Source schedule revisions differ from source lock")
    if set(report["inventories"]) != set(source.COMPONENTS):raise ValueError("Incomplete component inventory")
    for split in ("train","val"):
        path=pilot_corpus/split/"provenance.jsonl.gz"
        if source.file_hash(path) != report["hashes"][f"{split}/provenance.jsonl.gz"]: raise ValueError("Pilot identity provenance mismatch")
    benchmark=json.loads((benchmark_dir/"manifest.json").read_text())
    if source.file_hash(benchmark_dir/"manifest.json") != report["benchmark_manifest_sha256"] or set(benchmark["tasks"]) != set(source.BENCHMARKS):
        raise ValueError("Frozen benchmark set differs from pilot exclusion contract")
    if data.get("tokenizer_fit_split") != "train" or data.get("vocab_size") != 16384: raise ValueError("Owned train-only16K tokenizer required")
    store=Store(directory,create=True)
    try:
        store.set("config",dict(tranche_tokens=tranche_tokens,stage_tokens=stage_tokens,shard_positions=shard_positions,
            max_download_bytes=max_download_bytes,max_output_bytes=max_output_bytes,disk_reserve_bytes=disk_reserve_bytes,seed=seed,sequence_len=2048))
        store.disk_check()
        for name,(_,_,_,weight) in source.COMPONENTS.items(): store.db.execute("INSERT INTO components(name,weight) VALUES(?,?)",(name,weight))
        store.set("source_lock",lock); store.set("benchmark_dir","frozen/benchmarks")
        store.set("benchmark_manifest",benchmark); store.set("downloaded_bytes",0)
        store.set("network_inflight",False)
        store.set("lineage",{"pilot_manifest_sha256":source.file_hash(pilot_data/"manifest.json"),
            "source_report_sha256":source.file_hash(source_report),"source_lock_sha256":source.file_hash(source_lock),
            "benchmark_manifest_sha256":source.file_hash(benchmark_dir/"manifest.json"),
            "tokenizer_sha256":data["hashes"]["tokenizer.json"],"frozen_validation_path":"frozen/val.npy",
            "frozen_validation_sha256":data["hashes"]["val.npy"],"builder_sha256":source.file_hash(Path(__file__)),
            "token_bytes_path":"frozen/token_bytes.npy","token_bytes_sha256":data["hashes"]["token_bytes.npy"],
            "helper_sha256":{Path(module.__file__).name:source.file_hash(Path(module.__file__)) for module in [base,source]}})
        if source.file_hash(pilot_data/"tokenizer.json") != data["hashes"]["tokenizer.json"]: raise ValueError("Tokenizer mismatch")
        if source.file_hash(pilot_data/"val.npy") != data["hashes"]["val.npy"]: raise ValueError("Validation array mismatch")
        shutil.copyfile(pilot_data/"tokenizer.json",store.directory/"tokenizer.json")
        frozen=store.directory/"frozen";frozen.mkdir();(frozen/"benchmarks").mkdir()
        copied_inputs={"frozen/val.npy":(pilot_data/"val.npy",data["hashes"]["val.npy"]),
            "frozen/token_bytes.npy":(pilot_data/"token_bytes.npy",data["hashes"]["token_bytes.npy"]),
            "frozen/benchmarks/manifest.json":(benchmark_dir/"manifest.json",report["benchmark_manifest_sha256"])}
        for info in benchmark["tasks"].values():
            name=Path(info["local_file"]).name
            copied_inputs["frozen/benchmarks/"+name]=(benchmark_dir/name,info["sha256"])
        for name,(original,checksum) in copied_inputs.items():
            store.disk_check(original.stat().st_size+1024**2)
            shutil.copyfile(original,store.directory/name)
            if source.file_hash(store.directory/name)!=checksum:raise ValueError("Frozen input copy hash mismatch")
        for component,inventory in report["inventories"].items():
            if inventory["inventory_partial"]: raise ValueError("Main source schedule requires complete pinned file inventory")
            store.db.executemany("INSERT INTO files(component,entry,rank) VALUES(?,?,?)",[(component,base.canonical_json(e),hkey(f"{seed}:{component}:{e['path']}")) for e in inventory["files"]])
        store.db.commit()
        atomic_json(store.directory/"source-lock.json",lock)
        atomic_json(store.directory/"benchmark-selection.json",benchmark)
        with gzip.open(store.directory/"source-files.jsonl.gz","wt") as f:
            for row in store.db.execute("SELECT id,component,entry FROM files ORDER BY id"):
                f.write(base.canonical_json({"id":row["id"],"component":row["component"],**json.loads(row["entry"])})+"\n")
        lineage=store.get("lineage")
        lineage["metadata_artifact_hashes"]={name:source.file_hash(store.directory/name) for name in ["source-lock.json","source-files.jsonl.gz","benchmark-selection.json"]}
        lineage["metadata_artifact_hashes"].update({name:checksum for name,(_,checksum) in copied_inputs.items()})
        store.set("lineage",lineage);store.db.commit()
        for split,flag in (("val",PILOT_VAL),("train",PILOT_TRAIN)):
            with gzip.open(pilot_corpus/split/"provenance.jsonl.gz","rt") as f:
                for i,line in enumerate(f):
                    record=json.loads(line);store.group(record["group_keys"],flag)
                    if i % 1000 == 999:
                        store.db.commit();store.disk_check(16*1024**2)
            store.db.commit()
        store.set("status","initialized");store.event("initialized",{"network_downloads":0});store.db.commit()
        store.snapshot()
        return store
    except Exception as e:
        store.db.rollback();store.set("status","initialization_failed");store.event("initialization_failed",str(e));store.db.commit()
        raise


class TrackedHTTP(source.HTTP):
    """Persist each request's accounting; abrupt in-flight death fails closed."""
    def __init__(self,store):
        store.ready()
        if store.get("network_inflight"):raise RuntimeError("Interrupted network request has unreconciled bytes; inspect before authorizing more downloads")
        remaining=store.get("config")["max_download_bytes"]-store.get("downloaded_bytes",0)
        super().__init__(remaining);self.store=store
    def get(self,*args,**kwargs):
        self.store.disk_check(128*1024**2)
        self.store.set("network_inflight",True);self.store.db.commit()
        before,count=self.received,len(self.requests)
        try:return super().get(*args,**kwargs)
        finally:
            received=self.received-before
            self.store.set("downloaded_bytes",self.store.get("downloaded_bytes",0)+received)
            self.store.set("network_inflight",False)
            self.store.db.commit()
            with (self.store.directory/"network.jsonl").open("a") as f:
                for request in self.requests[count:]:f.write(base.canonical_json(request)+"\n")
                f.flush();os.fsync(f.fileno())


def benchmark_deny(store):
    directory=store.resolve(store.get("benchmark_dir"));manifest=store.get("benchmark_manifest")
    if source.file_hash(directory/"manifest.json") != store.get("lineage")["benchmark_manifest_sha256"]:raise ValueError("Benchmark manifest changed")
    texts=[]
    for task,info in manifest["tasks"].items():
        path=directory/Path(info["local_file"]).name
        if source.file_hash(path) != info["sha256"]:raise ValueError("Benchmark exclusion file changed")
        with gzip.open(path,"rt") as f:
            for line in f:texts.extend(source.benchmark_fields(task,json.loads(line)))
    return source.Deny(texts)


def register_groups(store,file_id,parquet):
    row=store.db.execute("SELECT * FROM files WHERE id=?",(file_id,)).fetchone();seed=store.get("config")["seed"]
    offset=0
    with store.db:
        for rg in range(parquet.num_row_groups):
            count=parquet.metadata.row_group(rg).num_rows
            store.db.execute("INSERT OR IGNORE INTO jobs(file_id,rg,rows,start_row,rank) VALUES(?,?,?,?,?)",
                (file_id,rg,count,offset,hkey(f"{seed}:{row['component']}:{json.loads(row['entry'])['path']}:{rg}")))
            offset+=count
        store.db.execute("UPDATE files SET scanned=1 WHERE id=?",(file_id,))


def open_main_parquet(http,pinned,entry):
    import pyarrow.parquet as pq
    from urllib.parse import quote
    url=f"https://huggingface.co/datasets/{pinned['repo']}/resolve/{pinned['revision']}/{quote(entry['path'],safe='/')}"
    # SQLite request accounting is synchronous on the owner thread. Disable
    # PyArrow's background prebuffer I/O rather than share a connection unsafely.
    return pq.ParquetFile(source.RangeFile(http,url,entry["size"]),pre_buffer=False)


class ParquetCache:
    """Bounded live file handles/footers; no unbounded source-byte cache."""
    def __init__(self,http,capacity=4):self.http,self.capacity,self.files=http,capacity,OrderedDict()
    def get(self,file_id,pinned,entry):
        if file_id in self.files:
            self.files.move_to_end(file_id)
            return self.files[file_id]
        value=open_main_parquet(self.http,pinned,entry)
        self.files[file_id]=value
        if len(self.files)>self.capacity:self.files.popitem(last=False)[1].close()
        return value
    def close(self):
        for value in self.files.values():value.close()
        self.files.clear()


def ingest_batch(store,candidates,codec,deny,job=None,next_cursor=None,done=False,total_limit=None,input_rejections=None):
    """Atomic group/data updates and source cursor; fixtures use no network."""
    outcomes=Counter()
    try:
        with store.db:
            for record in candidates:
                blocked=BENCHMARK if deny.matches(record["text"]) or deny.matches(record.get("original_text")) else 0
                remaining=None if total_limit is None else max(0,total_limit-sum(r["accepted"] for r in store.totals().values()))
                result,_=store.add(record,codec,blocked,remaining);outcomes[result]+=1
                if result=="limit_reached":
                    next_cursor=record["source_row_in_group"];done=False
                    break
            outcomes.update(reason for position,reason in (input_rejections or [])
                            if next_cursor is None or position<next_cursor)
            if job is not None:
                store.db.execute("UPDATE jobs SET cursor=?,done=? WHERE file_id=? AND rg=?",(next_cursor,int(done),*job))
            store.event("ingest_batch",dict(outcomes))
    except IdentityConflict as e:
        store.db.rollback();store.set("status","blocked_identity_conflict");store.event("identity_conflict",str(e));store.db.commit()
        store.snapshot()
        raise
    return outcomes


def _acquire(store,http,cache):
    """One explicit bounded tranche, stopping at source/document boundaries."""
    store.ready();config=store.get("config")
    codec=Codec(store.directory/"tokenizer.json",store.get("lineage")["tokenizer_sha256"])
    deny=benchmark_deny(store);lock=store.get("source_lock")
    initial=store.totals();packed=sum(x["packed"] for x in initial.values())
    target=min(config["stage_tokens"],packed+config["tranche_tokens"])
    while sum(x["accepted"] for x in store.totals().values()) < target:
        store.disk_check(128*1024**2)
        totals=store.totals()
        # Token-based source quotas; source failures never silently reweight.
        component=min(totals,key=lambda c:(totals[c]["accepted"]/totals[c]["weight"],c))
        job=store.db.execute("SELECT jobs.*,files.component,files.entry FROM jobs JOIN files ON jobs.file_id=files.id WHERE jobs.done=0 AND files.component=? ORDER BY files.rank,jobs.rank LIMIT1".replace("LIMIT1","LIMIT 1"),(component,)).fetchone()
        if job is None:
            file=store.db.execute("SELECT * FROM files WHERE component=? AND scanned=0 ORDER BY rank LIMIT1".replace("LIMIT1","LIMIT 1"),(component,)).fetchone()
            if file is None:raise RuntimeError(f"Finite pinned source exhausted for {component}; no replacement")
            name=source.COMPONENTS[component][0];pinned=lock["sources"][name]
            parquet=cache.get(file["id"],pinned,json.loads(file["entry"]))
            register_groups(store,file["id"],parquet);continue
        file_id=job["file_id"];name,config_name,_,_=source.COMPONENTS[component];pinned=lock["sources"][name]
        parquet=cache.get(file_id,pinned,json.loads(job["entry"]))
        wanted=("text","id","url","language") if name=="fineweb_edu" else ("text","id","url","rollout_results") if name=="finephrase" else ("text","uuid","license","metadata")
        columns=[c for c in wanted if c in parquet.schema_arrow.names]
        position=0
        for batch in parquet.iter_batches(batch_size=128,row_groups=[job["rg"]],columns=columns,use_threads=False):
            records=[];rejections=[];end=position+batch.num_rows
            if end<=job["cursor"]:position=end;continue
            for relative,row in enumerate(batch.to_pylist(),position):
                if relative<job["cursor"]:continue
                candidate={"source":name,"config":config_name,"row_index":job["start_row"]+relative,"row":row,
                    "viewer_partial":False,"viewer_indexed_rows":parquet.metadata.num_rows,"truncated_cells":[]}
                try:record=base.adapt(candidate,pinned,min_words=20 if name=="nemotron_fact" else 30)
                except base.Rejected as exc:
                    rejections.append((relative,"adapter_rejected:"+str(exc)))
                    continue
                record.update(component=component,file_id=file_id,source_row_group=job["rg"],source_row_in_group=relative)
                records.append(record)
            outcome=ingest_batch(store,records,codec,deny,(file_id,job["rg"]),end,end==job["rows"],target,rejections)
            position=end
            if outcome["limit_reached"]:return store.snapshot()
            if sum(x["accepted"] for x in store.totals().values())>=target:break
        store.snapshot()
        print(json.dumps({"accepted_ordinary":sum(x["accepted"] for x in store.totals().values()),"target":target,"download_bytes":store.get("downloaded_bytes")}),flush=True)
    return store.snapshot()


def acquire(store):
    started=time.monotonic();before=store.totals();http=TrackedHTTP(store);cache=ParquetCache(http)
    try:return _acquire(store,http,cache)
    finally:
        cache.close();after=store.totals()
        store.event("acquisition_run",{"wall_seconds":time.monotonic()-started,
            "received_body_bytes":http.received,
            "accepted_ordinary_delta":sum(after[c]["accepted"]-before[c]["accepted"] for c in before)})
        store.db.commit()


def recover_pack(store):
    journal=store.directory/"packing.json"
    if not journal.exists():return
    value=json.loads(journal.read_text());sid=value["id"]
    expected=[f"shards/train_{sid:05d}.npy",f"shards/train_{sid:05d}.provenance.jsonl.gz"]
    if value.get("paths")!=expected:raise ValueError("Unrecognized packing journal; do not remove files")
    row=store.db.execute("SELECT manifest FROM shards WHERE id=?",(sid,)).fetchone()
    if row:
        manifest=json.loads(row[0])
        for relative,checksum in manifest["hashes"].items():
            if source.file_hash(store.directory/relative)!=checksum:raise ValueError("Committed shard integrity failure")
    else:
        # Only this builder's journaled uncommitted outputs are disposable.
        for relative in expected:
            for suffix in ("",".tmp"):
                path=store.directory/(relative+suffix)
                if path.exists():path.unlink()
    journal.unlink();store.event("pack_recovered",{"id":sid,"committed":bool(row)});store.db.commit()


def pending_documents(store):
    """Bounded keyset batches; no live SELECT cursor is mutated by packing."""
    partial=store.db.execute("SELECT * FROM documents WHERE tokens IS NOT NULL AND consumed>0").fetchall()
    if len(partial)>1:raise ValueError("More than one partial document violates finite packing cursor")
    # Continue a split document before introducing a new tranche's shuffled docs.
    yield from partial
    last_key,last_id=b"",0
    while True:
        batch=store.db.execute("SELECT d.* FROM documents d JOIN groups g ON d.group_id=g.id WHERE g.blocked=0 AND d.tokens IS NOT NULL AND d.consumed=0 AND (d.order_key,d.id)>(?,?) ORDER BY d.order_key,d.id LIMIT64".replace("LIMIT64","LIMIT 64"),(last_key,last_id)).fetchall()
        if not batch:return
        for row in batch:
            last_key,last_id=row["order_key"],row["id"]
            yield row


def pack(store,crash_after_rename=False):
    """Emit verified finite shards; cursor updates commit with shard identities."""
    import numpy as np
    store.ready();recover_pack(store)
    # recover_pack only re-verifies the single shard named in packing.json's
    # journal (the shard mid-flight at the last crash/stop). Re-check every
    # previously committed shard too on every pack() call, so a shard
    # corrupted between separate pack() invocations (bad copy, disk fault,
    # partial rsync/scp while moving state.sqlite+shards between machines)
    # is caught instead of silently packed on top of; mirrors the resume-time
    # full-ledger recheck in pack_main_bounded.py/pack_main_scanned.py/
    # pack_main_repaired.py.
    for row in store.db.execute("SELECT manifest FROM shards ORDER BY id"):
        for relative,checksum in json.loads(row[0])["hashes"].items():
            if source.file_hash(store.directory/relative)!=checksum:raise ValueError("Committed shard integrity failure")
    config=store.get("config");width=config["sequence_len"]+1
    max_rows=config["shard_positions"]//width
    if max_rows<1:raise ValueError("Shard cap must fit one sequence row")
    while True:
        available=store.db.execute("SELECT COALESCE(SUM(d.positions-d.consumed),0) FROM documents d JOIN groups g ON d.group_id=g.id WHERE g.blocked=0 AND d.tokens IS NOT NULL").fetchone()[0]
        rows=min(max_rows,available//width)
        if not rows:break
        positions=rows*width;store.disk_check(positions*2+16*1024**2)
        sid=store.db.execute("SELECT COALESCE(MAX(id),-1)+1 FROM shards").fetchone()[0]
        paths=[f"shards/train_{sid:05d}.npy",f"shards/train_{sid:05d}.provenance.jsonl.gz"]
        if any((store.directory/x).exists() for x in paths):raise FileExistsError("Refusing unrelated shard overwrite")
        atomic_json(store.directory/"packing.json",{"id":sid,"paths":paths})
        output,provenance=[store.directory/(x+".tmp") for x in paths]
        result=np.lib.format.open_memmap(output,mode="w+",dtype="<u2",shape=(rows,width));flat=result.reshape(-1)
        consumed=0;counts=Counter()
        store.db.execute("BEGIN IMMEDIATE")
        try:
            with gzip.open(provenance,"wt",encoding="utf-8") as trace:
                for document in pending_documents(store):
                    start=document["consumed"];stop=min(document["positions"],start+positions-consumed)
                    tokens=np.frombuffer(zlib.decompress(document["tokens"]),dtype="<u2")
                    flat[consumed:consumed+stop-start]=tokens[start:stop]
                    ordinary=stop-start-int(start==0)
                    counts[document["component"]]+=ordinary
                    record=json.loads(document["provenance"])
                    record.update(document_id=document["id"],group_id=document["group_id"],document_position_start=start,
                                  document_position_stop=stop,shard_position_start=consumed,shard_position_stop=consumed+stop-start)
                    trace.write(base.canonical_json(record)+"\n")
                    store.db.execute("UPDATE documents SET consumed=?,tokens=CASE WHEN ?=positions THEN NULL ELSE tokens END WHERE id=?",(stop,stop,document["id"]))
                    store.db.execute("UPDATE groups SET packed=1 WHERE id=?",(document["group_id"],))
                    consumed+=stop-start
                    if consumed==positions:break
            if consumed!=positions:raise RuntimeError("Packing supply changed unexpectedly")
            result.flush();del flat;del result
            for path in (output,provenance):
                with path.open("rb") as f:os.fsync(f.fileno())
            manifest={"id":sid,"rows":rows,"row_width":width,"dtype":"<u2","encoded_positions":positions,
                "ordinary_positions":sum(counts.values()),"prediction_targets":rows*config["sequence_len"],
                "ordinary_positions_by_component":dict(counts),"wrapping":False,
                "hashes":{paths[0]:source.file_hash(output),paths[1]:source.file_hash(provenance)}}
            for name,n in counts.items():store.db.execute("UPDATE components SET packed_ordinary=packed_ordinary+? WHERE name=?",(n,name))
            store.db.execute("INSERT INTO shards VALUES(?,?)",(sid,base.canonical_json(manifest)))
            os.replace(output,store.directory/paths[0]);os.replace(provenance,store.directory/paths[1])
            sync_directory(store.directory/"shards")
            if crash_after_rename:raise RuntimeError("Injected crash before SQLite commit")
            store.db.commit();(store.directory/"packing.json").unlink()
            store.snapshot()
        except BaseException:
            store.db.rollback();raise
    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return store.snapshot()


def verify(store):
    store.ready()
    manifest=store.directory/"manifest.json"
    if source.file_hash(manifest)!=(store.directory/"manifest.sha256").read_text().strip():raise ValueError("Stage manifest hash mismatch")
    value=json.loads(manifest.read_text());lineage=store.get("lineage")
    if value["shards"]!=[json.loads(r[0]) for r in store.db.execute("SELECT manifest FROM shards ORDER BY id")]:raise ValueError("Manifest view is stale relative to committed state; inspect/rebuild status before use")
    for name,checksum in lineage.get("metadata_artifact_hashes",{}).items():
        if source.file_hash(store.directory/name)!=checksum:raise ValueError("Metadata artifact hash mismatch")
    if source.file_hash(store.directory/"tokenizer.json")!=lineage["tokenizer_sha256"]:raise ValueError("Tokenizer hash mismatch")
    if source.file_hash(store.resolve(lineage["frozen_validation_path"]))!=lineage["frozen_validation_sha256"]:raise ValueError("Frozen validation changed")
    for shard in value["shards"]:
        for path,checksum in shard["hashes"].items():
            if source.file_hash(store.directory/path)!=checksum:raise ValueError("Committed shard hash mismatch")
    return value


def update_limits(store,*,max_download_bytes,max_output_bytes,disk_reserve_bytes):
    """Explicit budget amendment; no implicit spending or token-stage expansion."""
    store.ready();previous=store.get("config");updated=dict(previous)
    changes=dict(max_download_bytes=max_download_bytes,max_output_bytes=max_output_bytes,disk_reserve_bytes=disk_reserve_bytes)
    if min(changes.values())<=0 or max_download_bytes<store.get("downloaded_bytes",0):raise ValueError("Invalid revised byte limits")
    updated.update(changes)
    with store.db:
        store.set("config",updated);store.disk_check()
        store.event("explicit_limit_amendment",{"before":previous,"after":updated})
    return store.snapshot()


def relocate_inputs(store,pilot_data,benchmark_dir):
    """Remap byte-identical frozen inputs after moving the entire stage directory."""
    store.ready();lineage=store.get("lineage");validation=Path(pilot_data)/"val.npy"
    if source.file_hash(validation)!=lineage["frozen_validation_sha256"]:raise ValueError("Relocated validation hash mismatch")
    with store.db:
        store.set("benchmark_dir",str(Path(benchmark_dir).resolve()))
        benchmark_deny(store)
        lineage["frozen_validation_path"]=str(validation.resolve());store.set("lineage",lineage)
        store.event("inputs_relocated",{"validation":str(validation.resolve()),"benchmarks":str(Path(benchmark_dir).resolve())})
    return store.snapshot()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["init","acquire","pack","verify","status","update-limits","relocate"]);p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--allow-download",action="store_true")
    p.add_argument("--pilot-data",type=Path,default=ROOT/"data/dense-tokens-production-001")
    p.add_argument("--pilot-corpus",type=Path,default=ROOT/"data/english-production-001")
    p.add_argument("--benchmark-dir",type=Path,default=ROOT/"data/benchmark-exclusions-002")
    p.add_argument("--source-report",type=Path,default=ROOT/"records/production-data-002.json")
    p.add_argument("--source-lock",type=Path,default=ROOT/"research/data_sources.json")
    p.add_argument("--tranche-tokens",type=int,default=250_000_000);p.add_argument("--stage-tokens",type=int,default=2_000_000_000)
    p.add_argument("--shard-positions",type=int,default=100_000_000);p.add_argument("--max-download-bytes",type=int)
    p.add_argument("--max-output-bytes",type=int);p.add_argument("--disk-reserve-bytes",type=int)
    a=p.parse_args(argv)
    if a.action=="acquire" and not a.allow_download:p.error("acquire requires explicit --allow-download; init/pack/status are offline")
    if a.action=="update-limits" and None in (a.max_download_bytes,a.max_output_bytes,a.disk_reserve_bytes):
        p.error("update-limits requires all three explicit byte limits")
    if a.action=="init":store=initialize(a.directory,a.pilot_data,a.pilot_corpus,a.benchmark_dir,a.source_report,a.source_lock,
        tranche_tokens=a.tranche_tokens,stage_tokens=a.stage_tokens,shard_positions=a.shard_positions,
        max_download_bytes=20_000_000_000 if a.max_download_bytes is None else a.max_download_bytes,
        max_output_bytes=30_000_000_000 if a.max_output_bytes is None else a.max_output_bytes,
        disk_reserve_bytes=10*1024**3 if a.disk_reserve_bytes is None else a.disk_reserve_bytes)
    else:store=Store(a.directory)
    if a.action=="acquire":
        acquire(store)
    elif a.action=="pack":pack(store)
    elif a.action=="verify":verify(store)
    elif a.action=="update-limits":update_limits(store,max_download_bytes=a.max_download_bytes,max_output_bytes=a.max_output_bytes,disk_reserve_bytes=a.disk_reserve_bytes)
    elif a.action=="relocate":relocate_inputs(store,a.pilot_data,a.benchmark_dir)
    result=store.snapshot()
    print(json.dumps({k:result[k] for k in ["status","components","downloaded_response_bytes","remaining_pending_positions"]},indent=2))
    store.close()


if __name__=="__main__":main()
