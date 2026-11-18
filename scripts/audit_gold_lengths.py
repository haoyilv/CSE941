#!/usr/bin/env python3
"""Count tokens in complete HousingQA annotated excerpts before GPU inference."""

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from extract_gemma_features import evidence_text
from housing_data import read_examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions-zip", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    examples = read_examples(args.questions_zip)
    tokenizer = AutoProcessor.from_pretrained(args.model_path).tokenizer
    rows = []
    for example in examples:
        text = evidence_text(example, example["gold_statutes"], 100, 100_000)
        rows.append({
            "idx": example["idx"],
            "passages": len(example["gold_statutes"]),
            "tokens": len(tokenizer.encode(text)),
            "characters": len(text),
        })
    values = sorted(row["tokens"] for row in rows)
    summary = {
        "n": len(rows),
        "max_passages": max(row["passages"] for row in rows),
        "mean_tokens": sum(values) / len(values),
        "p50_tokens": values[int(.50 * (len(values) - 1))],
        "p90_tokens": values[int(.90 * (len(values) - 1))],
        "p95_tokens": values[int(.95 * (len(values) - 1))],
        "p99_tokens": values[int(.99 * (len(values) - 1))],
        "max_tokens": values[-1],
        "over_4096": sum(v > 4096 for v in values),
        "over_8192": sum(v > 8192 for v in values),
        "over_16384": sum(v > 16384 for v in values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
