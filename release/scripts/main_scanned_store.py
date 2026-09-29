"""Additive v3: exact original pending SUM with forced sequential table scan.

The original partial order index is appropriate for shuffled keyset reads but
causes random table seeks when scanning every pending document. This changes
only that query plan; it introduces no counter assumptions or new manifest fields.
"""
import json
import os
import prepare_main as b
from prepare_main import atomic_json,source,sync_directory

PENDING_QUERY='SELECT COALESCE(SUM(d.positions-d.consumed),0) FROM documents d NOT INDEXED JOIN groups g ON d.group_id=g.id WHERE g.blocked=0 AND d.tokens IS NOT NULL'
QUERY_POLICY={"version":3,"query":PENDING_QUERY,"semantics":"original SUM and group filter; documents NOT INDEXED"}


def pending_positions(store):
    return store.db.execute(PENDING_QUERY).fetchone()[0]


class ScannedStore(b.Store):
    """Original snapshot body, with only the pending aggregation query changed."""
    def snapshot(self):
        values = {"schema_version":1,"status":self.get("status"),"config":self.get("config"),
            "lineage":self.get("lineage"),"downloaded_response_bytes":self.get("downloaded_bytes",0),
            "network_request_unreconciled":self.get("network_inflight",False),"components":self.totals(),
            "shards":[json.loads(r[0]) for r in self.db.execute("SELECT manifest FROM shards ORDER BY id")],
            "remaining_pending_positions":pending_positions(self),
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

