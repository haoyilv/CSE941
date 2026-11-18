#!/usr/bin/env python3
"""Combine first-hop BM25 with BM25 driven by a model's follow-up query."""

import argparse
import json
import sqlite3
from pathlib import Path

from evaluate_retrieval import query_expression


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    out = {int(row["idx"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--initial-retrieval", type=Path, required=True)
    p.add_argument("--followups", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first-k", type=int, default=5)
    p.add_argument("--second-k", type=int, default=5)
    args = p.parse_args()
    test_ids = [int(idx) for idx in json.loads(args.split.read_text())["test_ids"]]
    initial = read_jsonl(args.initial_retrieval)
    followups = read_jsonl(args.followups)
    if set(test_ids) - set(initial) or set(test_ids) - set(followups):
        raise ValueError("Initial or follow-up retrieval is missing test questions")
    con = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    any_count = all_count = newly_found = 0
    with args.output.open("w") as out:
        for i, idx in enumerate(test_ids, 1):
            original = initial[idx]
            followup = followups[idx]
            first = original["retrieved"][:args.first_k]
            if [str(r["idx"]) for r in first] != [str(x) for x in followup["first_hop_ids"]]:
                raise ValueError(f"First-hop evidence mismatch for question {idx}")
            expression = query_expression(original["state"], followup["followup_query"])
            raw_second = con.execute(
                "SELECT idx,citation,path,text,bm25(statutes) AS score "
                "FROM statutes WHERE statutes MATCH ? ORDER BY score LIMIT ?",
                (expression, max(args.second_k * 2, 10)),
            ).fetchall()
            second = [
                {"idx": row[0], "citation": row[1], "path": row[2],
                 "text": row[3], "score": row[4]}
                for row in raw_second
            ]
            combined = list(first)
            used = {str(row["idx"]) for row in combined}
            new = []
            for row in second:
                if str(row["idx"]) not in used:
                    combined.append(row)
                    new.append(row)
                    used.add(str(row["idx"]))
                if len(new) >= args.second_k:
                    break
            for row in original["retrieved"]:
                if len(combined) >= args.first_k + args.second_k:
                    break
                if str(row["idx"]) not in used:
                    combined.append(row)
                    used.add(str(row["idx"]))
            gold = set(map(str, original["gold_statute_ids"]))
            found = {str(row["idx"]) for row in combined}
            first_found = {str(row["idx"]) for row in first}
            gold_any = bool(gold & found)
            gold_all = gold <= found
            any_count += gold_any
            all_count += gold_all
            newly_found += bool((gold & found) - first_found)
            out.write(json.dumps({
                "idx": idx, "state": original["state"], "question": original["question"],
                "answer": original["answer"], "gold_statute_ids": original["gold_statute_ids"],
                "retrieved": combined, "gold_any": gold_any, "gold_all": gold_all,
                "strategy": "first5_then_followup5",
                "followup_query": followup["followup_query"],
                "first_hop_ids": [row["idx"] for row in first],
                "second_hop_ids": [row["idx"] for row in new],
                "new_annotated_id_after_second_hop": bool((gold & found) - first_found),
            }, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(test_ids)} any={any_count/i:.4f} all={all_count/i:.4f}", flush=True)
    print(json.dumps({"n": len(test_ids), "any_rate": any_count/len(test_ids),
                      "all_rate": all_count/len(test_ids),
                      "new_annotated_id_after_second_hop": newly_found}))


if __name__ == "__main__":
    main()
