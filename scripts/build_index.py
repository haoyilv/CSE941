#!/usr/bin/env python3
"""Build a local FTS5 index from the official HousingQA statutes ZIP."""
import argparse
import csv
import io
import sqlite3
import sys
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--statutes-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    con = sqlite3.connect(args.output)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE VIRTUAL TABLE statutes USING fts5(state, citation, path, text, idx UNINDEXED, tokenize='porter unicode61')")
    with zipfile.ZipFile(args.statutes_zip) as zf:
        with zf.open("statutes.tsv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""), delimiter="\t")
            batch = []
            n = 0
            for row in reader:
                batch.append((row["state"].lower(), row["citation"], row["path"], row["text"], row["idx"]))
                if len(batch) == 2000:
                    con.executemany("INSERT INTO statutes(state,citation,path,text,idx) VALUES (?,?,?,?,?)", batch)
                    con.commit()
                    n += len(batch)
                    batch.clear()
                    if n % 100000 == 0:
                        print(f"indexed {n:,}", flush=True)
            if batch:
                con.executemany("INSERT INTO statutes(state,citation,path,text,idx) VALUES (?,?,?,?,?)", batch)
                con.commit()
                n += len(batch)
    print(f"indexed {n:,} statutes", flush=True)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()


if __name__ == "__main__":
    main()
