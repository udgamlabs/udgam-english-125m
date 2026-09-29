#!/usr/bin/env python3
"""Freeze and score original English instruction checks; no model/provider calls.

This small diagnostic is not a representative benchmark or a training dataset.
Use its final split once after checkpoint selection. Human rubric items remain
unscored until independently reviewed; automatic checks never invent grades.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

FORMAT = "udgam-original-instruction-checks-v1"
POLICY = {
    "purpose": "small original diagnostic, not a general capability leaderboard",
    "generation": {"temperature": 0, "top_k": 0, "max_new_tokens": 128, "seed": 42},
    "tools": False,
    "final_use": "once after final checkpoint selection; no score-based training changes",
    "chat_rendering": "same native user/assistant markers for owned base and SFT",
    "automatic_scoring": "exact specified output after Unicode NFC and outer whitespace removal",
    "human_scoring": "unscored unless separately blinded reviewers assess fidelity, relevance, and constraints",
    "missing_responses": "incomplete report, never an aggregate success score",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(text):
    return unicodedata.normalize("NFC", text).strip()


def examples():
    """Original fixed toy scenarios, deliberately free of outside factual answers."""
    rows = []
    for split, indices in (("development", range(2)), ("final", range(2, 8))):
        for i in indices:
            code = f"Vela-{731 + i * 37}"
            n, m = 13 + 7 * i, 5 + 3 * i
            cases = [
                ("extraction", f'Return only a JSON object with keys "parcel" and "count". '
                 f'Memo: parcel {code} contains {n} sample cards. The shelf number is {m}. '
                 'Use the parcel identifier and card count, not the shelf number.',
                 {"kind": "json", "value": {"parcel": code, "count": n}}),
                ("grounding", f'Use only this log: {code} is marked violet. Runo-{812+i} is marked amber. '
                 f'What mark does {code} have? Reply with the single lowercase colour word.',
                 {"kind": "exact", "value": "violet"}),
                ("formatting", f'Reverse the order of these labels: {code}, Eno-{n}, Tavi-{m}. '
                 'Return only the reversed labels, separated by a comma and one space.',
                 {"kind": "exact", "value": f"Tavi-{m}, Eno-{n}, {code}"}),
                ("arithmetic", f'A box labelled {code} starts with {n} counters. '
                 f'{m} counters are added and 4 are removed. How many remain? Return only the integer.',
                 {"kind": "exact", "value": str(n + m - 4)}),
            ]
            for category, prompt, validator in cases:
                rows.append({"id": f"{split}-{category}-{i:02d}", "split": split, "category": category,
                    "messages": [{"role": "user", "content": prompt}], "validator": validator})
            rows.extend([
                {"id": f"{split}-rewrite-{i:02d}", "split": split, "category": "rewrite",
                 "messages": [{"role": "user", "content":
                    f'Rewrite this notice in friendly, clear English in no more than two sentences. '
                    f'Keep every fact: The {code} reading room opens at 09:15 on Friday; entry is free; '
                    f'visitors must bring their reservation code. Do not add any promises.'}],
                 "validator": {"kind": "human", "facts": [code, "09:15", "Friday", "free entry", "bring reservation code"],
                    "constraints": ["at most two sentences", "no added promises", "clear friendly rewrite"]}},
                {"id": f"{split}-clarification-{i:02d}", "split": split, "category": "clarification",
                 "messages": [{"role": "user", "content": f'Help me write a short invitation for event {code}.'},
                    {"role": "assistant", "content": "What date and venue should I include?"},
                    {"role": "user", "content": 'The date is 14 October. I have not chosen a venue. '
                     'Ask one short question about the missing detail. Do not invent it or repeat the date question.'}],
                 "validator": {"kind": "human", "facts": ["date already supplied", "venue unknown"],
                    "constraints": ["one short venue question", "no invented venue", "do not ask for the date"]}},
            ])
    for row in rows:
        row["prompt_sha256"] = digest(canonical(row["messages"]))
    return rows


def validate_rows(rows):
    if not rows or len(rows) > 100 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Missing, duplicate or excessive instruction cases")
    hashes = set()
    for row in rows:
        messages = row["messages"]
        if row["split"] not in {"development", "final"} or not messages or len(messages) % 2 != 1:
            raise ValueError("Invalid split/conversation")
        for i, message in enumerate(messages):
            if (set(message) != {"role", "content"} or message["role"] != ("user" if i % 2 == 0 else "assistant") or
                    not isinstance(message["content"], str) or not message["content"] or len(message["content"]) > 4096):
                raise ValueError("Invalid message")
        checksum = digest(canonical(messages))
        if checksum != row["prompt_sha256"] or checksum in hashes:
            raise ValueError("Changed or duplicate prompt")
        hashes.add(checksum)
        if row["validator"]["kind"] not in {"exact", "json", "human"}:
            raise ValueError("Unknown validator")
    return rows


def freeze(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Use a new immutable suite directory")
    rows = validate_rows(examples())
    body = "".join(canonical(r) + "\n" for r in rows)
    manifest = {"format": FORMAT, "policy": POLICY, "source": "Original deterministic Udgam synthetic diagnostic prompts; no teacher calls",
        "license": "MIT", "contains_training_examples": False, "prepared_before_sft": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "items_sha256": digest(body), "counts": dict(Counter(r["split"] for r in rows)),
        "limitations": ["Template-related development/final items are not independent task families.",
            "Only four narrow automatic tasks and two human-reviewed skills are covered.",
            "No claim of corpus-wide paraphrase decontamination or global representativeness.",
            "Manual rubric items have no automatic correctness labels."]}
    directory.mkdir(parents=True)
    (directory / "items.jsonl").write_text(body)
    text = canonical(manifest) + "\n"
    (directory / "manifest.json").write_text(text)
    (directory / "manifest.sha256").write_text(digest(text) + "\n")
    return manifest


def load(directory, expected_sha256):
    directory = Path(directory)
    raw = (directory / "manifest.json").read_text()
    if digest(raw) != expected_sha256 or expected_sha256 != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Instruction suite differs from frozen identity")
    manifest = json.loads(raw)
    body = (directory / "items.jsonl").read_text()
    if manifest.get("format") != FORMAT or manifest.get("policy") != POLICY or digest(body) != manifest["items_sha256"]:
        raise ValueError("Instruction policy or cases changed")
    rows = validate_rows([json.loads(line) for line in body.splitlines()])
    if dict(Counter(r["split"] for r in rows)) != manifest["counts"]:
        raise ValueError("Instruction counts changed")
    return manifest, rows


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def assess(row, text):
    if not isinstance(text, str) or len(text) > 32768:
        raise ValueError("Invalid bounded response text")
    rule = row["validator"]
    if rule["kind"] == "human":
        return None
    if rule["kind"] == "exact":
        return normalized(text) == rule["value"]
    try:
        answer = json.loads(normalized(text), object_pairs_hook=unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        return canonical(answer) == canonical(rule["value"])
    except (ValueError, TypeError):
        return False


def score(rows, responses, *, split, suite_sha256, checkpoint_sha256):
    validate_rows(rows)
    if split not in {"development", "final"} or any(not re.fullmatch(r"[a-f0-9]{64}", s) for s in (suite_sha256, checkpoint_sha256)):
        raise ValueError("Exact split/suite/checkpoint identity required")
    selected = {r["id"]: r for r in rows if r["split"] == split}
    seen = {}; items = []
    for response in responses:
        key = response["id"]
        if key not in selected or key in seen:
            raise ValueError("Unknown, cross-split or repeated response")
        row = selected[key]
        if (response.get("prompt_sha256") != row["prompt_sha256"] or response.get("suite_sha256") != suite_sha256 or
                response.get("checkpoint_sha256") != checkpoint_sha256 or response.get("generation") != POLICY["generation"] or
                response.get("prompt_format") != "native_chat" or response.get("external_tools") is not False):
            raise ValueError("Response belongs to different prompt/model/generation policy")
        correct = assess(row, response["text"])
        items.append({"id": key, "category": row["category"], "correct": correct,
            "human_review_required": correct is None})
        seen[key] = response
    automatic = [r for r in selected.values() if r["validator"]["kind"] != "human"]
    completed = len(seen) == len(selected)
    correct = sum(i["correct"] is True for i in items)
    return {"format": FORMAT, "split": split, "suite_sha256": suite_sha256, "checkpoint_sha256": checkpoint_sha256,
        "responses_complete": completed, "received": len(seen), "expected": len(selected),
        "automatic_correct": correct, "automatic_total": len(automatic),
        "automatic_accuracy": correct / len(automatic) if completed else None,
        "human_review_complete": False, "human_score": None, "items": items,
        "scope": POLICY["purpose"], "not_general_capability_or_sarvam_superiority_evidence": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "score")); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--expected-suite-sha256"); p.add_argument("--checkpoint-sha256")
    p.add_argument("--split", choices=("development", "final")); p.add_argument("--responses", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.action == "freeze":
        result = freeze(a.suite)
    else:
        if not all((a.expected_suite_sha256, a.checkpoint_sha256, a.split, a.responses, a.output)):
            p.error("Scoring requires explicit frozen suite, checkpoint, split, response file and new output")
        if a.output.exists(): raise FileExistsError("Immutable scoring output required")
        _, rows = load(a.suite, a.expected_suite_sha256)
        if a.responses.stat().st_size > 4 * 1024**2: raise ValueError("Response file too large")
        responses = [json.loads(s) for s in a.responses.read_text().splitlines() if s.strip()]
        result = score(rows, responses, split=a.split, suite_sha256=a.expected_suite_sha256, checkpoint_sha256=a.checkpoint_sha256)
        a.output.write_text(canonical(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
