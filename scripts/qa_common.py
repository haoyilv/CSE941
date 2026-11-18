"""Small shared readers and answer parser for local HousingQA experiments."""

import json
import re
import zipfile
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        return json.load(archive.open("questions.json"))


def load_retrieval(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    with path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    out = {int(row["idx"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return out


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"Final answer\s*:\s*(Yes|No)\b", text, flags=re.I)
    if matches:
        return matches[-1].capitalize()
    bare = re.fullmatch(r"\s*(Yes|No)[.!\s]*", text, flags=re.I)
    return bare.group(1).capitalize() if bare else None
