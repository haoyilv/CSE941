#!/usr/bin/env python3
"""Cache frozen Gemma 4 final-token vectors for HousingQA questions.

The model is not trained here. Vectors are written incrementally so an HPCC
job can resume safely. Labels and annotated statutes are never given to Gemma.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-zip", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="smoke test on first N examples")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for Gemma feature extraction")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    examples = read_examples(args.questions_zip)
    if args.limit is not None:
        examples = examples[: args.limit]
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
            inputs = tokenizer(
                [input_text(example) for example in batch], return_tensors="pt",
                padding=True, truncation=True, max_length=args.max_input_tokens,
            ).to(model.device)
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
                }
                output.write(json.dumps(record) + "\n")
            output.flush()
            completed = len(seen) + offset + len(batch)
            if completed // 100 > (completed - len(batch)) // 100 or completed == len(examples):
                print(f"{completed}/{len(examples)} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
