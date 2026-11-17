#!/usr/bin/env python3
"""Compare question-only, retrieved-evidence, and gold-evidence probes."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> dict[int, dict]:
    with path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    indexed = {int(row["idx"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return indexed


def paired(a: dict[int, dict], b: dict[int, dict], ids: list[int]) -> dict:
    gained = lost = both_correct = both_wrong = 0
    for idx in ids:
        a_ok = a[idx]["prediction"] == a[idx]["gold_answer"]
        b_ok = b[idx]["prediction"] == b[idx]["gold_answer"]
        gained += not a_ok and b_ok
        lost += a_ok and not b_ok
        both_correct += a_ok and b_ok
        both_wrong += not a_ok and not b_ok
    return {
        "a_wrong_b_correct": gained,
        "a_correct_b_wrong": lost,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_correct_gain": gained - lost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--question-predictions", type=Path, required=True)
    parser.add_argument("--gold-predictions", type=Path, required=True)
    parser.add_argument("--retrieved-predictions", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    ids = [int(idx) for idx in split["test_ids"]]
    q = read_jsonl(args.question_predictions)
    g = read_jsonl(args.gold_predictions)
    r = read_jsonl(args.retrieved_predictions)
    retrieval = read_jsonl(args.retrieval)
    required = set(ids)
    for name, data in (("question", q), ("gold", g), ("retrieved", r), ("retrieval", retrieval)):
        missing = required - set(data)
        if missing:
            raise ValueError(f"{name} missing {len(missing)} test IDs")
    for idx in ids:
        if {q[idx]["gold_answer"], g[idx]["gold_answer"], r[idx]["gold_answer"]} != {q[idx]["gold_answer"]}:
            raise ValueError(f"Gold answer mismatch for {idx}")
        if any(data[idx]["partition"] != "test" for data in (q, g, r)):
            raise ValueError(f"Non-test prediction for {idx}")
    conditions = {}
    for name, data in (("question_only", q), ("gold_evidence", g), ("bm25_evidence", r)):
        correct = sum(data[idx]["prediction"] == data[idx]["gold_answer"] for idx in ids)
        conditions[name] = {"correct": correct, "n": len(ids), "accuracy": correct / len(ids)}
    retrieval_stats = {}
    for partition in ("train", "test"):
        part_ids = [int(idx) for idx in split[f"{partition}_ids"]]
        retrieval_stats[partition] = {
            "n": len(part_ids),
            "at_least_one_annotated_in_top10": sum(retrieval[idx]["gold_any"] for idx in part_ids),
            "all_annotated_in_top10": sum(retrieval[idx]["gold_all"] for idx in part_ids),
        }
        retrieval_stats[partition]["any_rate"] = (
            retrieval_stats[partition]["at_least_one_annotated_in_top10"] / len(part_ids)
        )
        retrieval_stats[partition]["all_rate"] = (
            retrieval_stats[partition]["all_annotated_in_top10"] / len(part_ids)
        )
    answer_by_hit = {}
    for hit in (True, False):
        hit_ids = [idx for idx in ids if bool(retrieval[idx]["gold_any"]) is hit]
        correct = sum(r[idx]["prediction"] == r[idx]["gold_answer"] for idx in hit_ids)
        answer_by_hit[str(hit).lower()] = {
            "n": len(hit_ids), "correct": correct,
            "accuracy": correct / len(hit_ids) if hit_ids else None,
        }
    report = {
        "sample": "course-internal question-group-disjoint test split",
        "n_test": len(ids),
        "conditions": conditions,
        "paired_test": {
            "question_to_bm25": paired(q, r, ids),
            "question_to_gold": paired(q, g, ids),
            "bm25_to_gold": paired(r, g, ids),
        },
        "retrieval": retrieval_stats,
        "bm25_answer_accuracy_by_annotated_hit": answer_by_hit,
        "caveat": "Gold-evidence is an oracle diagnostic; annotated ID overlap does not imply all decisive text was present. Evidence inputs are capped to ten passages of 400 characters each and 1536 total tokens.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
