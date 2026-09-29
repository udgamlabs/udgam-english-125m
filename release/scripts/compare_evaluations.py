#!/usr/bin/env python3
"""Paired, task-stratified source-cluster bootstrap for the frozen final suite.

No model loading, training, checkpoint selection, paid calls or benchmark scoring.
Quality conclusions concern this suite only. Resource dominance is not inferred.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import evaluate_dense as evaluation


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def clusters(suite):
    """Use supplied source IDs, or shared passages, without changing selection."""
    output = {}
    for task, rows in suite.items():
        values = []
        for row in rows:
            doc = row["document"]
            if task == "hellaswag" and doc.get("source_id") is not None:
                identity = ["source_id", str(doc["source_id"])]
            elif task == "boolq":
                identity = ["passage", normalized(doc["passage"])]
            elif task == "winogrande":
                identity = ["sentence", normalized(doc["sentence"])]
            else:
                identity = ["document", row["document_sha256"]]
            values.append(evaluation.digest(evaluation.canonical([task, identity])))
        output[task] = values
    return output


def read_complete(directory, requests, benchmark_hash):
    directory = Path(directory)
    contract = json.loads((directory / "evaluation-contract.json").read_text())
    if (contract.get("policy") != evaluation.POLICY or
            contract.get("benchmark_manifest_sha256") != benchmark_hash or
            contract.get("suite_kind") != "frozen_final_suite"):
        raise ValueError("Only the unchanged frozen final scoring policy is comparable")
    expected_text = "".join(evaluation.canonical(r) + "\n" for r in requests)
    if contract.get("requests_sha256") != evaluation.digest(expected_text):
        raise ValueError("Final request strings differ")
    if (directory / "requests.jsonl").read_text() != expected_text:
        raise ValueError("Saved final requests differ")
    items, _ = evaluation.cached_items(directory / "items.jsonl", requests, contract)
    summary = json.loads((directory / "summary.json").read_text())
    if (len(items) != len(requests) or summary.get("complete") is not True or
            summary.get("items_sha256") != evaluation.sha256(directory / "items.jsonl")):
        raise ValueError("Partial or changed results cannot establish final superiority")
    for item, request in zip(items, requests, strict=True):
        for key in ("task", "gold", "primary_metric", "document_sha256", "original_split_row_index"):
            if item.get(key) != request[key]:
                raise ValueError("Scored item identity differs from the fixed request")
        if type(item.get("primary_correct")) is not int or item["primary_correct"] not in (0, 1):
            raise ValueError("Correctness must be a binary observation")
        if not item["supported"] and item["primary_correct"]:
            raise ValueError("Unsupported observations must count as incorrect")
        metric = item["acc_norm"] if request["primary_metric"] == "acc_norm" else item["acc"]
        if item["primary_correct"] != metric:
            raise ValueError("Primary task metric changed")
    calculated = evaluation.summarize(items, requests)
    for key in ("Q", "quality_floors_pass"):
        if summary.get(key) != calculated[key]:
            raise ValueError("Summary does not match item outcomes")
    return items, calculated, contract


def paired_bootstrap(candidate, baseline, group_ids, *, repeats=10000, seed=20260922):
    if set(candidate) != set(evaluation.TASKS) or set(candidate) != set(baseline) or set(candidate) != set(group_ids):
        raise ValueError("All five task strata are required")
    rng = np.random.default_rng(seed)
    task_samples, output = [], {}
    for task in evaluation.TASKS:
        left, right = np.asarray(candidate[task]), np.asarray(baseline[task])
        if left.ndim != 1 or len(left) == 0 or left.shape != right.shape or len(left) != len(group_ids[task]):
            raise ValueError("Paired task item counts differ")
        if not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
            raise ValueError("Only binary correctness can be bootstrapped")
        differences = left.astype(np.float64) - right.astype(np.float64)
        groups = defaultdict(list)
        for index, identity in enumerate(group_ids[task]): groups[identity].append(index)
        sizes = np.asarray([len(indices) for indices in groups.values()])
        sums = np.asarray([differences[indices].sum() for indices in groups.values()])
        count = len(groups)
        samples = np.empty(repeats, dtype=np.float64)
        # Bound temporary memory while preserving one seeded sequential stream.
        for begin in range(0, repeats, 256):
            end = min(begin + 256, repeats)
            draws = rng.integers(count, size=(end - begin, count))
            samples[begin:end] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        task_samples.append(samples)
        output[task] = {"items": len(left), "source_clusters": count,
                        "difference_percentage_points": float(differences.mean() * 100),
                        "ci95_percentage_points": (np.quantile(samples, [.025, .975]) * 100).tolist(),
                        "individual_task_inference": "exploratory; no multiplicity correction"}
    macro = np.mean(task_samples, axis=0)
    return {"replicates": repeats, "seed": seed, "method": "paired source-cluster bootstrap within each task; unweighted task macro",
            "macro_difference_percentage_points": float(np.mean([x["difference_percentage_points"] for x in output.values()])),
            "macro_ci95_percentage_points": (np.quantile(macro, [.025, .975]) * 100).tolist(), "tasks": output}


def quality_gates(stats, candidate_summary):
    floors = candidate_summary["quality_floors_pass"] is True
    no_large_regression = all(t["difference_percentage_points"] >= -5 - 1e-9 for t in stats["tasks"].values())
    better = (floors and no_large_regression and
              stats["macro_difference_percentage_points"] >= 2 - 1e-9 and
              stats["macro_ci95_percentage_points"][0] > 0)
    noninferior = floors and no_large_regression and stats["macro_ci95_percentage_points"][0] >= -3
    return {"quality_floors_pass": floors, "no_task_regression_over_five_points": no_large_regression,
            "better_on_this_2500_item_suite": better,
            "noninferior_quality_gate_for_smaller_tradeoff": noninferior,
            "useful_smaller_tradeoff_established": False,
            "resource_gate_status": "Requires separately matched artifact-size and peak-memory evidence plus latency/cost reporting",
            "pareto_frontier_established": False}


def compare(candidate_dir, baseline_dir, benchmark_dir):
    _, suite = evaluation.load_frozen_suite(benchmark_dir)
    requests = evaluation.make_requests(suite)
    benchmark_hash = evaluation.sha256(Path(benchmark_dir) / "manifest.json")
    left, summary, left_contract = read_complete(candidate_dir, requests, benchmark_hash)
    right, _, right_contract = read_complete(baseline_dir, requests, benchmark_hash)
    if left_contract["task_source_manifest_sha256"] != right_contract["task_source_manifest_sha256"]:
        raise ValueError("Task definition snapshots differ")
    vectors = lambda rows: {task: [r["primary_correct"] for r in rows if r["task"] == task] for task in evaluation.TASKS}
    groups = clusters(suite)
    cluster_record = Path(__file__).resolve().parents[1] / "records/evaluation-clusters-001.json"
    frozen = json.loads(cluster_record.read_text())
    if (frozen.get("status") != "frozen_before_confirmatory_scoring" or
            frozen.get("seed") != 20260922 or frozen.get("replicates") != 10000 or
            frozen.get("benchmark_manifest_sha256") != benchmark_hash or frozen.get("groups") != groups or
            frozen.get("groups_sha256") != evaluation.digest(evaluation.canonical(groups))):
        raise ValueError("Source cluster policy differs from the record frozen before scoring")
    stats = paired_bootstrap(vectors(left), vectors(right), groups)
    return {"schema_version": 1, "benchmark_manifest_sha256": benchmark_hash,
            "comparison_script_sha256": evaluation.sha256(Path(__file__)),
            "numpy_version": np.__version__,
            "frozen_cluster_record_sha256": evaluation.sha256(cluster_record),
            "candidate_contract_sha256": evaluation.sha256(Path(candidate_dir) / "evaluation-contract.json"),
            "baseline_contract_sha256": evaluation.sha256(Path(baseline_dir) / "evaluation-contract.json"),
            "cluster_map_sha256": evaluation.digest(evaluation.canonical(groups)),
            "statistics": stats, "gates": quality_gates(stats, summary),
            "limitations": ["Conditional on this fixed checkpoint and suite; no training-seed variability measured",
                "Known HellaSwag source IDs and repeated BoolQ passages are clustered; unprovided upstream links cannot be recovered",
                "No claim about all real-world tasks, all Sarvam models, or a global Pareto frontier"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError("Preserve an existing comparison record")
    result = compare(a.candidate, a.baseline, a.benchmarks)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__": main()
