#!/usr/bin/env python3
"""Score predictions without treating missing or malformed answers as correct."""
import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--split", type=Path, help="Verify exact held-out test ID coverage")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.predictions.open() if line.strip()]
    by_id = {row["idx"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("Duplicate question IDs in predictions")
    if args.split:
        expected = set(json.loads(args.split.read_text())["test_ids"])
        if set(by_id) != expected:
            raise SystemExit(f"Test ID mismatch: missing={len(expected-set(by_id))}, extra={len(set(by_id)-expected)}")
    n = len(rows)
    correct = sum(r["prediction"] == r["gold_answer"] for r in rows)
    invalid = sum(r["prediction"] not in {"Yes", "No"} for r in rows)
    labels = collections.Counter(r["gold_answer"] for r in rows)
    tp = sum(r["gold_answer"] == "Yes" and r["prediction"] == "Yes" for r in rows)
    tn = sum(r["gold_answer"] == "No" and r["prediction"] == "No" for r in rows)
    fp = sum(r["gold_answer"] == "No" and r["prediction"] == "Yes" for r in rows)
    fn = sum(r["gold_answer"] == "Yes" and r["prediction"] != "Yes" for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {"n": n, "correct": correct, "accuracy": correct / n if n else None,
              "invalid_or_missing": invalid, "label_counts": dict(labels),
              "model_requested": sorted(set(r["model_requested"] for r in rows)),
              "mode": sorted(set(r.get("mode", f"{r.get('retrieval_strategy', 'one_shot')}/{r.get('evidence_mode')}/{r.get('prompt_style')}") for r in rows)),
              "yes_precision": precision, "yes_recall": recall,
              "yes_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
              "balanced_accuracy": (
                  (recall + tn / labels["No"]) / 2
                  if labels.get("No") and labels.get("Yes") else None
              ),
              "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn_or_invalid_yes": fn}}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
