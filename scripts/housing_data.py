#!/usr/bin/env python3
"""Read HousingQA questions and make a reproducible group-disjoint split.

The public release calls its question collection a test split. Our split is an
*internal* train/test split for the course assignment, not the official paper
evaluation. Questions sharing a question_group are kept on one side to reduce
template leakage across jurisdictions.
"""

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def read_examples(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        raw = json.load(archive.open("questions.json"))
    examples = []
    seen_ids = set()
    for row in raw:
        idx = int(row["idx"])
        if idx in seen_ids:
            raise ValueError(f"Duplicate question ID: {idx}")
        seen_ids.add(idx)
        answer = str(row["answer"]).capitalize()
        if answer not in {"Yes", "No"}:
            raise ValueError(f"Unexpected answer for {idx}: {answer}")
        examples.append({
            "idx": idx,
            "state": str(row["state"]),
            "question": str(row["question"]),
            "question_group": str(row["question_group"]),
            "label": 1 if answer == "Yes" else 0,
            "gold_statutes": row.get("statutes", []),
        })
    return examples


def input_text(example: dict) -> str:
    """X: jurisdiction and question only; gold statutes are excluded."""
    return (
        "Answer a question about United States housing law around 2021.\n"
        f"Jurisdiction: {example['state']}\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )


def make_group_split(examples: list[dict], seed: int, test_fraction: float) -> dict:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    groups = defaultdict(list)
    for example in examples:
        groups[example["question_group"]].append(example)
    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    target = round(len(examples) * test_fraction)
    test_groups = set()
    test_size = 0
    for group in group_names:
        if test_size >= target:
            break
        test_groups.add(group)
        test_size += len(groups[group])
    train = [row for row in examples if row["question_group"] not in test_groups]
    test = [row for row in examples if row["question_group"] in test_groups]
    if not train or not test:
        raise ValueError("Split produced an empty partition")
    assert {row["question_group"] for row in train}.isdisjoint(
        {row["question_group"] for row in test}
    )
    return {
        "source": "reglab/housing_qa questions release (official test collection)",
        "purpose": "course-internal train/test split; not comparable to published test results",
        "seed": seed,
        "test_fraction_requested": test_fraction,
        "group_key": "question_group",
        "input": "2021 jurisdiction + question text; no statute evidence",
        "target": "Yes=1, No=0",
        "n_total": len(examples),
        "train_ids": [row["idx"] for row in train],
        "test_ids": [row["idx"] for row in test],
        "train_groups": len(groups) - len(test_groups),
        "test_groups": len(test_groups),
        "train_labels": dict(Counter("Yes" if row["label"] else "No" for row in train)),
        "test_labels": dict(Counter("Yes" if row["label"] else "No" for row in test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    examples = read_examples(args.questions_zip)
    split = make_group_split(examples, args.seed, args.test_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps({k: v for k, v in split.items() if not k.endswith("_ids")}, indent=2))
    print("Example X:", input_text(examples[0]))
    print("Example Y:", examples[0]["label"])


if __name__ == "__main__":
    main()
