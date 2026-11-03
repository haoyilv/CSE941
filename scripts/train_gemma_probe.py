#!/usr/bin/env python3
"""Train an initial Yes/No classifier on frozen Gemma 4 question vectors."""

import argparse
import base64
import json
import struct
from pathlib import Path

import torch

from housing_data import read_examples


def read_vectors(path: Path) -> dict[int, torch.Tensor]:
    vectors = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            if idx in vectors:
                raise ValueError(f"Duplicate feature ID {idx}")
            width = int(row["width"])
            packed = base64.b64decode(row["vector_f16_b64"], validate=True)
            if len(packed) != width * 2:
                raise ValueError(f"Wrong feature length for {idx}")
            vectors[idx] = torch.tensor(struct.unpack(f"<{width}e", packed), dtype=torch.float32)
    return vectors


def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        prediction = (torch.sigmoid(model(x).squeeze(1)) >= 0.5).long()
    true = y.long()
    tp = int(((prediction == 1) & (true == 1)).sum())
    tn = int(((prediction == 0) & (true == 0)).sum())
    fp = int(((prediction == 1) & (true == 0)).sum())
    fn = int(((prediction == 0) & (true == 1)).sum())
    n = len(true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": n,
        "yes": tp + fn,
        "no": tn + fp,
        "correct": tp + tn,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (recall + specificity) / 2,
        "yes_precision": precision,
        "yes_recall": recall,
        "yes_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "majority_class_accuracy": max(tp + fn, tn + fp) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-zip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    torch.manual_seed(20261103)
    examples = {row["idx"]: row for row in read_examples(args.questions_zip)}
    split = json.loads(args.split.read_text())
    vectors = read_vectors(args.features)
    ids = [int(i) for i in split["train_ids"] + split["test_ids"]]
    missing = set(ids) - set(vectors)
    if missing:
        raise ValueError(f"Missing features for {len(missing)} IDs, e.g. {sorted(missing)[:5]}")
    if set(split["train_ids"]) & set(split["test_ids"]):
        raise ValueError("Train/test IDs overlap")
    width = {vectors[i].numel() for i in ids}
    if len(width) != 1:
        raise ValueError("Feature widths differ")

    def make_tensors(partition: str) -> tuple[torch.Tensor, torch.Tensor]:
        part_ids = [int(i) for i in split[f"{partition}_ids"]]
        x = torch.stack([vectors[i] for i in part_ids])
        y = torch.tensor([examples[i]["label"] for i in part_ids], dtype=torch.float32)
        return x, y

    train_x, train_y = make_tensors("train")
    test_x, test_y = make_tensors("test")
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_x).squeeze(1), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    report = {
        "method": "frozen Gemma 4 26B A4B final-token question vector + trained linear classifier",
        "input": split["input"],
        "target": split["target"],
        "split_seed": split["seed"],
        "split_group_key": split["group_key"],
        "feature_width": train_x.shape[1],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "final_train_loss": float(loss_function(model(train_x).squeeze(1), train_y)),
        "train": evaluate(model, train_x, train_y),
        "test": evaluate(model, test_x, test_y),
        "comparison_warning": "Internal split of the public benchmark test collection; not comparable to published test accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
