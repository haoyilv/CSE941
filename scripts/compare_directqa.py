#!/usr/bin/env python3
"""Paired evaluation of direct and structured Gemma HousingQA runs."""

import argparse
import json
import re
from pathlib import Path

CONDITIONS = [
    "none_direct", "bm25_direct", "gold_direct",
    "bm25_structured", "gold_structured",
]


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    out = {int(row["idx"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate question IDs in {path}")
    return out


def paired(a: dict[int, dict], b: dict[int, dict], ids: list[int]) -> dict:
    gain = loss = both_correct = both_wrong = 0
    for idx in ids:
        a_ok = a[idx]["prediction"] == a[idx]["gold_answer"]
        b_ok = b[idx]["prediction"] == b[idx]["gold_answer"]
        gain += not a_ok and b_ok
        loss += a_ok and not b_ok
        both_correct += a_ok and b_ok
        both_wrong += not a_ok and not b_ok
    return {"a_wrong_b_correct": gain, "a_correct_b_wrong": loss,
            "both_correct": both_correct, "both_wrong": both_wrong,
            "net_gain": gain - loss,
            "accuracy_point_change": 100 * (gain - loss) / len(ids)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--retrieval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ids = [int(idx) for idx in json.loads(args.split.read_text())["test_ids"]]
    expected = set(ids)
    results = {name: read_jsonl(args.results_dir / f"directqa_{name}.jsonl")
               for name in CONDITIONS}
    for name, rows in results.items():
        if set(rows) != expected:
            raise ValueError(f"{name} missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}")
        for idx in ids:
            row = rows[idx]
            if row["gold_answer"] != results["none_direct"][idx]["gold_answer"]:
                raise ValueError(f"Answer mismatch for {idx}")
            if f"{row['evidence_mode']}_{row['prompt_style']}" != name:
                raise ValueError(f"Condition mismatch in {name} for {idx}")
    retrieval = read_jsonl(args.retrieval)
    if expected - set(retrieval):
        raise ValueError("Retrieval file is missing test IDs")
    table = {}
    for name, rows in results.items():
        correct = sum(rows[idx]["prediction"] == rows[idx]["gold_answer"] for idx in ids)
        invalid = sum(rows[idx]["prediction"] not in {"Yes", "No"} for idx in ids)
        cap = 384 if "structured" in name else 64
        capped = sum(rows[idx]["output_tokens"] >= cap for idx in ids)
        table[name] = {"n": len(ids), "correct": correct, "accuracy": correct/len(ids),
                       "invalid": invalid, "at_generation_cap": capped,
                       "mean_input_tokens": sum(rows[idx]["input_tokens"] for idx in ids)/len(ids),
                       "mean_output_tokens": sum(rows[idx]["output_tokens"] for idx in ids)/len(ids)}
        if "structured" in name:
            valid_citation = unsupported_number = uncertainty = 0
            for idx in ids:
                row = rows[idx]
                numbers = [int(x) for x in re.findall(r"\[(\d+)\]", row["raw_response"])]
                valid_citation += any(1 <= n <= len(row["evidence_ids"]) for n in numbers)
                unsupported_number += any(n < 1 or n > len(row["evidence_ids"]) for n in numbers)
                uncertainty += bool(re.search(r"Uncertainty:\s*(?!none\b)\S", row["raw_response"], re.I))
            table[name].update({"has_valid_numbered_citation": valid_citation,
                                "has_out_of_range_citation_number": unsupported_number,
                                "declares_uncertainty": uncertainty})
    hit = {idx: bool(retrieval[idx]["gold_any"]) for idx in ids}
    answer_by_hit = {}
    for name in ("bm25_direct", "bm25_structured"):
        answer_by_hit[name] = {}
        for flag in (True, False):
            group = [idx for idx in ids if hit[idx] == flag]
            correct = sum(results[name][idx]["prediction"] == results[name][idx]["gold_answer"] for idx in group)
            answer_by_hit[name][str(flag).lower()] = {
                "n": len(group), "correct": correct, "accuracy": correct/len(group)
            }
    comparisons = {
        "none_to_bm25_direct": paired(results["none_direct"], results["bm25_direct"], ids),
        "none_to_gold_direct": paired(results["none_direct"], results["gold_direct"], ids),
        "bm25_to_gold_direct": paired(results["bm25_direct"], results["gold_direct"], ids),
        "bm25_direct_to_structured": paired(results["bm25_direct"], results["bm25_structured"], ids),
        "gold_direct_to_structured": paired(results["gold_direct"], results["gold_structured"], ids),
        "bm25_structured_to_gold_structured": paired(results["bm25_structured"], results["gold_structured"], ids),
    }
    report = {"sample": "course-internal group-disjoint test split",
              "conditions": table, "paired": comparisons,
              "bm25_answer_accuracy_by_gold_id_hit": answer_by_hit,
              "caveat": "Citation-number presence is a syntax check, not legal support verification. Gold evidence consists of complete released excerpts; BM25 evidence uses top ten passages, each capped to 400 characters."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
