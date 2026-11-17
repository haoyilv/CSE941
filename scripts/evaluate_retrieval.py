#!/usr/bin/env python3
"""Evaluate state-filtered SQLite FTS5 BM25 retrieval on HousingQA."""
import argparse
import json
import re
import sqlite3
import zipfile
from pathlib import Path


def query_expression(state: str, question: str) -> str:
    words = re.findall(r"[a-z0-9]+", question.lower())
    words = [w for w in words if len(w) > 2 and w not in {"the", "and", "for", "are", "can", "does", "there", "state", "territory", "law"}]
    words = list(dict.fromkeys(words))[:128]
    if not words:
        words = ["housing"]
    state_tokens = re.findall(r"[a-z0-9]+", state.lower())
    state_phrase = " ".join(state_tokens)
    return f'state:"{state_phrase}" AND (' + " OR ".join(f'"{w}"' for w in words) + ")"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions-zip", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ids-file", type=Path)
    p.add_argument("--expansions", type=Path, help="JSONL output from run_query_expansion.py")
    args = p.parse_args()
    with zipfile.ZipFile(args.questions_zip) as zf:
        questions = json.load(zf.open("questions.json"))
    if args.ids_file:
        selected = set(json.loads(args.ids_file.read_text()))
        questions = [q for q in questions if q["idx"] in selected]
    expansions = {}
    if args.expansions:
        expansions = {r["idx"]: r["expansion"] for line in args.expansions.open() if (r := json.loads(line))}
        missing = [q["idx"] for q in questions if not expansions.get(q["idx"])]
        if missing:
            raise SystemExit(f"Missing nonempty expansions for {len(missing)} selected questions")
    if args.limit:
        questions = questions[: args.limit]
    con = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hit_any = hit_all = 0
    with args.output.open("w") as out:
        for i, q in enumerate(questions, 1):
            expanded_query = q["question"] + " " + expansions.get(q["idx"], "")
            expr = query_expression(q["state"], expanded_query)
            rows = con.execute(
                "SELECT idx,citation,path,text,bm25(statutes) AS score FROM statutes WHERE statutes MATCH ? ORDER BY score LIMIT ?",
                (expr, args.top_k),
            ).fetchall()
            gold = {str(s["statute_idx"]) for s in q["statutes"]}
            got = {r[0] for r in rows}
            hit_any += bool(gold & got)
            hit_all += gold <= got
            out.write(json.dumps({
                "idx": q["idx"], "state": q["state"], "question": q["question"],
                "answer": q["answer"], "gold_statute_ids": sorted(gold),
                "expansion": expansions.get(q["idx"]),
                "retrieved": [{"idx": r[0], "citation": r[1], "path": r[2], "text": r[3], "score": r[4]} for r in rows],
                "gold_any": bool(gold & got), "gold_all": gold <= got,
            }, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(questions)} any={hit_any/i:.4f} all={hit_all/i:.4f}", flush=True)
    print(json.dumps({"n": len(questions), "top_k": args.top_k, "recall_any": hit_any / len(questions), "recall_all": hit_all / len(questions)}))


if __name__ == "__main__":
    main()
