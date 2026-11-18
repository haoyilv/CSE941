#!/usr/bin/env python3
"""Direct Gemma QA on a fixed HousingQA split with auditable evidence inputs."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from qa_common import load_questions, load_retrieval, parse_answer


def build_prompt(question: dict, evidence_mode: str, prompt_style: str,
                 retrieval: dict[int, dict], top_k: int, bm25_chars: int) -> tuple[str, list[dict]]:
    start = (
        f"Consider statutory housing law in {question['state']} around 2021.\n"
        f"Question: {question['question']}\n"
    )
    passages = []
    if evidence_mode == "gold":
        passages = [
            {"idx": str(p["statute_idx"]), "citation": p["citation"],
             "text": p["excerpt"]}
            for p in question["statutes"]
        ]
    elif evidence_mode == "bm25":
        passages = [
            {"idx": str(p["idx"]), "citation": p["citation"],
             "text": p["text"][:bm25_chars]}
            for p in retrieval[question["idx"]]["retrieved"][:top_k]
        ]
    if evidence_mode != "none":
        start += "Statutory evidence:\n"
        for i, passage in enumerate(passages, 1):
            start += f"[{i}] {passage['citation']}\n{passage['text']}\n\n"
    if prompt_style == "direct":
        instruction = "Respond with exactly one line: 'Final answer: Yes' or 'Final answer: No'. Do not explain."
    else:
        instruction = (
            "Answer the binary benchmark question. Your FIRST line must be exactly "
            "'Final answer: Yes' or 'Final answer: No'. Never put 'Uncertain' in that line. "
            "Then give a short, evidence-grounded explanation (at most 120 words total). "
            "Identify the controlling rule, relevant definition, exception, or cross-reference "
            "when present, and apply it to the exact question. Cite supplied excerpt numbers "
            "such as [1]. Do not invent a legal rule. If the excerpts are insufficient, "
            "say so in the Evidence gap line, while still making a tentative Yes/No prediction. "
            "Use this format:\n"
            "Final answer: <Yes or No>\n"
            "Rule: <rule and excerpt citation, or missing>\n"
            "Application: <conditions and exceptions applied to the question>\n"
            "Evidence gap: <none or missing legal support>"
        )
    return start + instruction, passages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions-zip", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--retrieval", type=Path)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--evidence-mode", choices=["none", "gold", "bm25"], required=True)
    p.add_argument("--prompt-style", choices=["direct", "structured"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bm25-chars", type=int, default=400)
    p.add_argument("--retrieval-strategy", choices=["one_shot", "iterative"],
                   default="one_shot")
    p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--limit", type=int)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.evidence_mode == "bm25" and args.retrieval is None:
        raise SystemExit("--retrieval is required for BM25 evidence")
    split = json.loads(args.split.read_text())
    selected = {int(idx) for idx in split["test_ids"]}
    questions = [q for q in load_questions(args.questions_zip) if q["idx"] in selected]
    if len(questions) != len(selected):
        raise ValueError("Some test IDs are absent from the questions file")
    if args.limit is not None:
        questions = questions[:args.limit]
    retrieval = load_retrieval(args.retrieval) if args.evidence_mode == "bm25" else {}
    if args.evidence_mode == "bm25" and any(q["idx"] not in retrieval for q in questions):
        raise ValueError("Missing BM25 result for one or more test questions")
    processor = AutoProcessor.from_pretrained(args.model_path)
    pending = []
    seen = set()
    if args.output.exists() and not args.preflight_only:
        with args.output.open() as source:
            for line in source:
                row = json.loads(line)
                if row["idx"] in seen:
                    raise ValueError("Duplicate question ID in existing output")
                if (row["evidence_mode"] != args.evidence_mode
                    or row["prompt_style"] != args.prompt_style
                    or row.get("retrieval_strategy", "one_shot") != args.retrieval_strategy):
                    raise ValueError("Existing output uses a different experiment condition")
                seen.add(row["idx"])
    lengths = []
    for q in questions:
        prompt, passages = build_prompt(
            q, args.evidence_mode, args.prompt_style, retrieval, args.top_k, args.bm25_chars
        )
        messages = [
            {"role": "system", "content": "Answer the legal benchmark question carefully. Distinguish the specified jurisdiction from other jurisdictions."},
            {"role": "user", "content": prompt},
        ]
        input_ids = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        length = len(input_ids[0])
        lengths.append(length)
        if length > args.max_input_tokens:
            raise ValueError(f"Question {q['idx']} has {length} tokens, above {args.max_input_tokens}")
        if q["idx"] not in seen:
            pending.append((q, messages, passages, length))
    if args.preflight_only:
        print(json.dumps({"n": len(questions), "min_tokens": min(lengths),
                          "max_tokens": max(lengths), "mean_tokens": sum(lengths)/len(lengths),
                          "evidence_mode": args.evidence_mode,
                          "prompt_style": args.prompt_style}, indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for generation")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with args.output.open("a") as out, torch.inference_mode():
        for i, (q, messages, passages, length) in enumerate(pending, 1):
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False,
            ).to(model.device)
            if inputs["input_ids"].shape[-1] != length:
                raise ValueError(f"Prompt length changed for question {q['idx']}")
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            raw = processor.decode(output_ids[0][length:], skip_special_tokens=True)
            record = {
                "idx": q["idx"], "state": q["state"], "question": q["question"],
                "gold_answer": q["answer"], "prediction": parse_answer(raw),
                "raw_response": raw,
                "model_requested": "google/gemma-4-26B-A4B-it",
                "evidence_mode": args.evidence_mode, "prompt_style": args.prompt_style,
                "retrieval_strategy": args.retrieval_strategy,
                "evidence_ids": [p["idx"] for p in passages],
                "evidence_citations": [p["citation"] for p in passages],
                "bm25_chars": args.bm25_chars if args.evidence_mode == "bm25" else None,
                "input_tokens": length,
                "output_tokens": int(output_ids.shape[-1] - length),
                "error": None,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(pending):
                print(f"{len(seen)+i}/{len(questions)} elapsed={time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
