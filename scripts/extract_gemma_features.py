#!/usr/bin/env python3
"""Cache frozen Gemma 4 final-token vectors for HousingQA questions.

The model is not trained here. Vectors are written incrementally so an HPCC
job can resume safely. Labels are never given to Gemma. Annotated statutes are
included only when --evidence-mode=gold is explicitly selected.
"""

import argparse
import base64
import json
import struct
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from housing_data import input_text, read_examples


def encode_vector(values: torch.Tensor) -> str:
    flat = values.detach().float().cpu().tolist()
    packed = struct.pack(f"<{len(flat)}e", *flat)
    return base64.b64encode(packed).decode("ascii")


def load_retrieval(path: Path | None) -> dict[int, list[dict]]:
    if path is None:
        return {}
    with path.open() as source:
        return {
            int(row["idx"]): row["retrieved"]
            for line in source if (row := json.loads(line))
        }


def evidence_text(example: dict, passages: list[dict], max_passages: int, max_chars: int) -> str:
    start = input_text(example).removesuffix("Answer:")
    pieces = [start, "Statutory evidence:\n"]
    for i, passage in enumerate(passages[:max_passages], 1):
        body = passage.get("excerpt", passage.get("text", ""))
        pieces.append(f"[{i}] {passage['citation']}\n{body[:max_chars]}\n\n")
    pieces.append("Answer:")
    return "".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-zip", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evidence-mode", choices=["none", "gold", "retrieved"], default="none")
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--max-passages", type=int, default=10)
    parser.add_argument("--max-passage-chars", type=int, default=400)
    parser.add_argument("--strict-untruncated", action="store_true",
                        help="Fail if the assembled input exceeds max-input-tokens; never truncate it")
    parser.add_argument("--limit", type=int, help="smoke test on first N examples")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for Gemma feature extraction")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.evidence_mode == "retrieved" and args.retrieval is None:
        raise SystemExit("--retrieval is required for retrieved evidence")
    if args.max_passages < 1 or args.max_passage_chars < 1:
        raise SystemExit("Passage limits must be positive")
    examples = read_examples(args.questions_zip)
    if args.limit is not None:
        examples = examples[: args.limit]
    retrieval = load_retrieval(args.retrieval)
    if args.evidence_mode == "retrieved":
        missing = [row["idx"] for row in examples if row["idx"] not in retrieval]
        if missing:
            raise SystemExit(f"Missing retrieved evidence for {len(missing)} questions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.output.exists():
        with args.output.open() as source:
            for line in source:
                if line.strip():
                    seen.add(int(json.loads(line)["idx"]))
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    started = time.monotonic()
    pending = [example for example in examples if example["idx"] not in seen]
    with args.output.open("a") as output, torch.inference_mode():
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            texts = []
            for example in batch:
                if args.evidence_mode == "none":
                    texts.append(input_text(example))
                else:
                    passages = (
                        example["gold_statutes"] if args.evidence_mode == "gold"
                        else retrieval[example["idx"]]
                    )
                    texts.append(evidence_text(
                        example, passages, args.max_passages, args.max_passage_chars
                    ))
            if args.strict_untruncated:
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
                lengths = inputs["attention_mask"].sum(dim=1).tolist()
                if any(length > args.max_input_tokens for length in lengths):
                    bad = [(example["idx"], length) for example, length in zip(batch, lengths)
                           if length > args.max_input_tokens]
                    raise ValueError(f"Inputs exceed {args.max_input_tokens} tokens: {bad}")
            else:
                inputs = tokenizer(
                    texts, return_tensors="pt",
                    padding=True, truncation=True, max_length=args.max_input_tokens,
                )
            inputs = inputs.to(model.device)
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            attention = inputs["attention_mask"]
            last_position = attention.shape[1] - 1 - attention.flip(dims=[1]).argmax(dim=1)
            batch_index = torch.arange(len(batch), device=hidden.device)
            vectors = hidden[batch_index, last_position]
            for example, vector, mask in zip(batch, vectors, attention):
                record = {
                    "idx": example["idx"],
                    "vector_f16_b64": encode_vector(vector),
                    "width": vector.numel(),
                    "input_tokens": int(mask.sum()),
                    "evidence_mode": args.evidence_mode,
                    "max_input_tokens": args.max_input_tokens,
                    "max_passages": args.max_passages if args.evidence_mode != "none" else 0,
                    "max_passage_chars": args.max_passage_chars if args.evidence_mode != "none" else 0,
                    "strict_untruncated": args.strict_untruncated,
                }
                output.write(json.dumps(record) + "\n")
            output.flush()
            completed = len(seen) + offset + len(batch)
            if completed // 100 > (completed - len(batch)) // 100 or completed == len(examples):
                print(f"{completed}/{len(examples)} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
