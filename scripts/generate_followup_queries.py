#!/usr/bin/env python3
"""Generate second-hop statute searches after reading first-hop evidence."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from qa_common import load_questions, load_retrieval


def prompt_for(question: dict, initial: dict, first_k: int, chars: int) -> str:
    text = (
        f"You are researching statutory housing law in {question['state']} around 2021.\n"
        f"Question: {question['question']}\n"
        "The first search found these statutes:\n"
    )
    for i, passage in enumerate(initial["retrieved"][:first_k], 1):
        text += f"[{i}] {passage['citation']}\n{passage['text'][:chars]}\n\n"
    text += (
        "Identify one legal dependency that may still need checking, such as a definition, "
        "exception, notice period, procedural condition, or cross-reference. "
        "Write a short targeted search query for the next statute search in this state. "
        "Use statutory terms where possible. Do not answer the Yes/No question. "
        "Respond in exactly one line: Search query: <terms>"
    )
    return text


def parse_query(raw: str, fallback: str) -> str:
    match = re.search(r"Search query\s*:\s*(.+)", raw, flags=re.I)
    query = match.group(1).splitlines()[0].strip() if match else ""
    query = re.sub(r"[\r\n\t]+", " ", query)[:240].strip(" .:;\"'")
    return query if len(query.split()) >= 2 else fallback


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions-zip", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--initial-retrieval", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first-k", type=int, default=5)
    p.add_argument("--max-passage-chars", type=int, default=400)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-input-tokens", type=int, default=8192)
    args = p.parse_args()
    selected = {int(idx) for idx in json.loads(args.split.read_text())["test_ids"]}
    questions = [q for q in load_questions(args.questions_zip) if q["idx"] in selected]
    if len(questions) != len(selected):
        raise ValueError("Question ID coverage mismatch")
    initial = load_retrieval(args.initial_retrieval)
    if selected - set(initial):
        raise ValueError("Initial retrieval is missing test questions")
    seen = set()
    if args.output.exists():
        with args.output.open() as source:
            for line in source:
                idx = json.loads(line)["idx"]
                if idx in seen:
                    raise ValueError("Duplicate ID in existing follow-up output")
                seen.add(idx)
    processor = AutoProcessor.from_pretrained(args.model_path)
    pending = []
    for question in questions:
        if question["idx"] in seen:
            continue
        prompt = prompt_for(question, initial[question["idx"]], args.first_k, args.max_passage_chars)
        messages = [
            {"role": "system", "content": "Formulate a targeted legal research query; do not answer the question."},
            {"role": "user", "content": prompt},
        ]
        tokens = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        length = len(tokens[0])
        if length > args.max_input_tokens:
            raise ValueError(f"Follow-up prompt for {question['idx']} is too long: {length}")
        pending.append((question, messages, length))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with args.output.open("a") as out, torch.inference_mode():
        for i, (question, messages, length) in enumerate(pending, 1):
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False
            ).to(model.device)
            if inputs["input_ids"].shape[-1] != length:
                raise ValueError("Prompt length changed between preflight and inference")
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            raw = processor.decode(generated[0][length:], skip_special_tokens=True)
            record = {
                "idx": question["idx"], "state": question["state"],
                "question": question["question"],
                "first_hop_ids": [p["idx"] for p in initial[question["idx"]]["retrieved"][:args.first_k]],
                "raw_response": raw,
                "followup_query": parse_query(raw, question["question"]),
                "input_tokens": length,
                "output_tokens": int(generated.shape[-1] - length),
                "model_requested": "google/gemma-4-26B-A4B-it",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(pending):
                print(f"{len(seen)+i}/{len(questions)} elapsed={time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
