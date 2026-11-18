#!/usr/bin/env python3
"""Measure whether second-hop retrieval improves evidence coverage and answers."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    out = {int(row["idx"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return out


def paired_flags(a: dict[int, bool], b: dict[int, bool], ids: list[int]) -> dict:
    gain = sum(not a[idx] and b[idx] for idx in ids)
    loss = sum(a[idx] and not b[idx] for idx in ids)
    return {"gained": gain, "lost": loss, "net_gain": gain - loss,
            "a_count": sum(a[idx] for idx in ids), "b_count": sum(b[idx] for idx in ids),
            "n": len(ids)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--one-shot-retrieval", type=Path, required=True)
    p.add_argument("--iterative-retrieval", type=Path, required=True)
    p.add_argument("--one-shot-predictions", type=Path, required=True)
    p.add_argument("--iterative-predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ids = [int(idx) for idx in json.loads(args.split.read_text())["test_ids"]]
    expected = set(ids)
    one = read_jsonl(args.one_shot_retrieval)
    two = read_jsonl(args.iterative_retrieval)
    answer_one = read_jsonl(args.one_shot_predictions)
    answer_two = read_jsonl(args.iterative_predictions)
    for name, data in (("one-shot retrieval", one), ("iterative retrieval", two),
                       ("one-shot answers", answer_one), ("iterative answers", answer_two)):
        if expected - set(data):
            raise ValueError(f"{name} is missing test questions")
    for idx in ids:
        if answer_one[idx]["gold_answer"] != answer_two[idx]["gold_answer"]:
            raise ValueError(f"Gold answer mismatch for {idx}")
        if one[idx]["gold_statute_ids"] != two[idx]["gold_statute_ids"]:
            raise ValueError(f"Gold statute mismatch for {idx}")
    any_one = {idx: bool(one[idx]["gold_any"]) for idx in ids}
    any_two = {idx: bool(two[idx]["gold_any"]) for idx in ids}
    all_one = {idx: bool(one[idx]["gold_all"]) for idx in ids}
    all_two = {idx: bool(two[idx]["gold_all"]) for idx in ids}
    correct_one = {idx: answer_one[idx]["prediction"] == answer_one[idx]["gold_answer"] for idx in ids}
    correct_two = {idx: answer_two[idx]["prediction"] == answer_two[idx]["gold_answer"] for idx in ids}
    answer_by_retrieval_change = {}
    for label, subset in (
        ("newly_any_hit", [idx for idx in ids if not any_one[idx] and any_two[idx]]),
        ("lost_any_hit", [idx for idx in ids if any_one[idx] and not any_two[idx]]),
        ("unchanged_any_hit", [idx for idx in ids if any_one[idx] == any_two[idx]]),
    ):
        answer_by_retrieval_change[label] = {
            "n": len(subset),
            "one_shot_correct": sum(correct_one[idx] for idx in subset),
            "iterative_correct": sum(correct_two[idx] for idx in subset),
        }
    report = {
        "sample": "course-internal group-disjoint test split",
        "retrieval_any_annotated_id": paired_flags(any_one, any_two, ids),
        "retrieval_all_annotated_ids": paired_flags(all_one, all_two, ids),
        "answer_accuracy": paired_flags(correct_one, correct_two, ids),
        "answer_by_retrieval_change": answer_by_retrieval_change,
        "method": "First 5 BM25 statutes, LLM generated follow-up query from those statutes, 5 new BM25 statutes, then structured answer over up to 10 statutes.",
        "caveat": "Annotated ID overlap is an imperfect proxy for decisive evidence. Both final answer conditions use the same structured prompt and 400-character passage cap."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
