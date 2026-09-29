"""Explicit pre-training adoption of sequential-scan main preparation v3.

The running v1 files/state are never modified by merely importing this module.
Adoption requires --adopt-v3-before-training: an operator assertion that no main
checkpoint has pinned the prior lineage. The earlier pilot is unaffected. Old
transport snapshots/run history remain verifiable; new runs use frozen v3 code.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import shutil

import acquire_main_cached as transport
import main_batch_engine as engine
import pack_main_bounded as packer
import acquire_main_batched as previous
import main_scanned_store as scanned
import pack_main_scanned as scanned_packer

builder=transport.builder
MIB=transport.MIB


def adopt(store,cache_config,*,allow_transition=False):
    store.ready()
    if store.get("network_inflight"):raise RuntimeError("Prior network request remains unreconciled")
    lineage=store.get("lineage")
    if builder.source.file_hash(Path(builder.__file__))!=lineage["builder_sha256"]:raise ValueError("Original builder changed")
    for module in [builder.base,builder.source]:
        if builder.source.file_hash(Path(module.__file__))!=lineage["helper_sha256"][Path(module.__file__).name]:raise ValueError("Frozen helper changed")
    identity={"driver":"cached-scanned-main-v3","driver_sha256":builder.source.file_hash(Path(__file__)),
        "engine_sha256":builder.source.file_hash(Path(engine.__file__)),
        "v2_driver_sha256":builder.source.file_hash(Path(previous.__file__)),
        "scanned_store_sha256":builder.source.file_hash(Path(scanned.__file__)),
        "scanned_packer_sha256":builder.source.file_hash(Path(scanned_packer.__file__)),
        "pending_query_policy":scanned.QUERY_POLICY,
        "transport_driver_sha256":builder.source.file_hash(Path(transport.__file__)),
        "cache_sha256":builder.source.file_hash(Path(transport.main_range_cache.__file__)),
        "packer_sha256":builder.source.file_hash(Path(packer.__file__)),"packer_policy":packer.POLICY,
        "cache_config":cache_config,"batch_rows":engine.BATCH_ROWS,"snapshot_policy":"once_per_tranche_and_failure"}
    current=lineage.get("active_transport")
    if current and current.get("driver")==identity["driver"]:
        if any(current.get(k)!=v for k,v in identity.items()):raise ValueError("Pinned v3 code or policy changed")
        return current
    if not allow_transition:raise ValueError("Explicit --adopt-v3-before-training required; cannot silently change a training contract")
    if current is None or current.get("driver")!="cached-batched-main-v2":raise ValueError("Expected recorded v2 transport before v3 transition")
    previous.adopt(store,cache_config)  # validates all code/policy still used from v2
    directory=store.directory/"transport_v3"
    paths=[Path(__file__),Path(scanned.__file__),Path(scanned_packer.__file__),Path(previous.__file__),Path(engine.__file__),Path(packer.__file__),Path(transport.__file__),Path(transport.main_range_cache.__file__),
        Path(builder.__file__),Path(builder.base.__file__),Path(builder.source.__file__)]
    store.disk_check(sum(p.stat().st_size for p in paths)+16*MIB);directory.mkdir(exist_ok=False)
    hashes={}
    for path in paths:
        destination=directory/path.name;shutil.copyfile(path,destination)
        hashes[str(destination.relative_to(store.directory))]=builder.source.file_hash(destination)
    at=datetime.now(timezone.utc).isoformat()
    plan={"schema_version":3,**identity,"adopted_at_utc":at,"previous_active_transport":current,
        "cutover_cursor":transport.cursor_state(store),"config_at_adoption":store.get("config"),
        "source_hashes":hashes,"operator_assertion":"adopted before any main-training checkpoint",
        "semantics":"same source order, eligibility, grouping, per-document quota and packed token bytes; 1024-row commits with family-aware disk reservations, deferred full progress aggregation, bounded pack chunks preserving concatenated stream; exact pending SUM with documents NOT INDEXED"}
    path=directory/"plan.json";builder.atomic_json(path,plan)
    relative=str(path.relative_to(store.directory));checksum=builder.source.file_hash(path)
    active={**identity,"plan_path":relative,"plan_sha256":checksum,"source_snapshot_hashes":hashes}
    lineage.setdefault("previous_active_transports",[]).append({"replaced_at_utc":at,"active_transport":current})
    lineage["active_transport"]=active
    lineage["metadata_artifact_hashes"].update({**hashes,relative:checksum})
    with store.db:
        store.set("lineage",lineage);store.event("explicit_transport_v3_adoption",{"previous":current,"active":active})
    store.snapshot();return active


def guarded_engine(store,http,cache):
    try:return engine.acquire(store,http,cache)
    except transport.GracefulStop:raise
    except Exception as exc:
        # IdentityConflict already set/published its more specific blocked state
        # in the original transaction. Never overwrite or defer that signal.
        with store.db:
            if isinstance(exc,ValueError) and store.get("status")=="initialized":
                store.set("status","blocked_acquisition_integrity")
            store.event("batched_acquisition_failed",{"class":type(exc).__name__,"message":str(exc)[:2048],
                "status":store.get("status")})
        store.snapshot()
        raise


def run(store,*,allow_transition=False,stop=None,block_bytes=MIB,cache_bytes=128*MIB,max_fetch_bytes=32*MIB):
    original_engine=builder._acquire;original_adopt=transport.adopt_transport
    builder._acquire=guarded_engine
    transport.adopt_transport=partial(adopt,allow_transition=allow_transition)
    try:return transport.run(store,stop=stop,block_bytes=block_bytes,cache_bytes=cache_bytes,max_fetch_bytes=max_fetch_bytes)
    finally:builder._acquire=original_engine;transport.adopt_transport=original_adopt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--allow-download",action="store_true")
    p.add_argument("--adopt-v3-before-training",action="store_true")
    p.add_argument("--adopt-only",action="store_true")
    a=p.parse_args(argv)
    if not a.adopt_only and not a.allow_download:p.error("Acquisition requires explicit --allow-download")
    if a.adopt_only and not a.adopt_v3_before_training:p.error("Adopt-only requires explicit --adopt-v3-before-training")
    store=scanned.ScannedStore(a.directory)
    try:
        builder.verify(store)
        if a.adopt_only:
            adopt(store,{"block_bytes":MIB,"max_bytes":128*MIB,"max_fetch_bytes":32*MIB,"max_read_bytes":64*MIB},allow_transition=True)
        else:
            with transport.StopController() as stop:run(store,allow_transition=a.adopt_v3_before_training,stop=stop)
    finally:store.close()


if __name__=="__main__":main()
