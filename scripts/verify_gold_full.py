#!/usr/bin/env python3
"""Verify complete-gold features cover every question without input truncation."""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    args = p.parse_args()
    expected = {int(r["idx"]): int(r["tokens"])
                for r in json.loads(args.audit.read_text())["rows"]}
    observed = {}
    with args.features.open() as source:
        for line in source:
            row = json.loads(line)
            idx = int(row["idx"])
            if idx in observed:
                raise ValueError(f"Duplicate feature for question {idx}")
            if row["evidence_mode"] != "gold" or not row.get("strict_untruncated"):
                raise ValueError(f"Question {idx} was not extracted with strict full-gold settings")
            if row["max_passages"] < 15 or row["max_passage_chars"] < 15825:
                raise ValueError(f"Question {idx} used passage caps that may omit evidence")
            observed[idx] = int(row["input_tokens"])
    if set(expected) != set(observed):
        raise ValueError(f"Question coverage mismatch: missing={len(set(expected)-set(observed))}, "
                         f"extra={len(set(observed)-set(expected))}")
    mismatches = [(idx, expected[idx], observed[idx]) for idx in expected
                  if expected[idx] != observed[idx]]
    if mismatches:
        raise ValueError(f"Input-token mismatch in {len(mismatches)} examples: {mismatches[:5]}")
    print(json.dumps({"verified_questions": len(expected),
                      "complete_excerpts": True, "truncated_inputs": 0,
                      "maximum_input_tokens": max(observed.values())}, indent=2))


if __name__ == "__main__":
    main()
