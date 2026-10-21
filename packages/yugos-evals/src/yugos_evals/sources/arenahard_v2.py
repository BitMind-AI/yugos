"""Arena-Hard v2.0 questions and per-category baseline answers at a pinned upstream commit.

Files come from lmarena/arena-hard-auto at REVISION and are verified by
SHA-256 before any row is read. Each question's category selects its baseline
model; the baseline answer is the sample target and never reaches the model.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from inspect_ai.dataset import Sample

from ._common import LoadedSource
from ._fetch import fetch, file_record, verified

REPOSITORY = "lmarena/arena-hard-auto"
REVISION = "196f6b826783b3da7310e361a805fa36f0be83f3"
QUESTIONS_SHA256 = "a75cef6623db6b27aec810497b059f0659705d7ff48dcbccc5bd5b130728ac73"
CATEGORIES = ("hard_prompt", "creative_writing")
# Baseline model and SHA-256 of its upstream model_answer file, by question category.
BASELINES = {
    "hard_prompt": (
        "o3-mini-2025-01-31",
        "c23a7c255eead99f7f86ad48aa1d8ccdb5b3334f2097e0e1e09184df87265e03",
    ),
    "creative_writing": (
        "gemini-2.0-flash-001",
        "70f614e1345f787a1da98c239d27f6df3c1d26b411cf894949f934a3039d6f0b",
    ),
}
_BASE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/data/arena-hard-v2.0"
_MAX_BYTES = 64 * 1024 * 1024


def _file(name: str, sha256: str, data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        return verified(Path(data_dir) / name, sha256)
    return fetch(f"{_BASE_URL}/{name}", name=Path(name).name, max_bytes=_MAX_BYTES, sha256=sha256)


def _rows(path: Path) -> list[dict]:
    # Split on newlines only: prompts contain Unicode line separators.
    with path.open(encoding="utf-8", newline="") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _baseline_answer(row: dict) -> tuple[str, str]:
    try:
        uid, answer = row["uid"], row["messages"][-1]["content"]["answer"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected Arena-Hard baseline answer record") from None
    if not isinstance(uid, str) or not isinstance(answer, str) or not answer.strip():
        raise ValueError("Arena-Hard baseline answers need a UID and a nonempty answer")
    return uid, answer


def load_arenahard_v2(
    categories: Collection[str] = CATEGORIES, *, data_dir: str | Path | None = None
) -> LoadedSource:
    """Questions of `categories`, each bound to its category's pinned baseline answer.

    `data_dir` holds staged copies as `question.jsonl` and `model_answer/<model>.jsonl`.
    A question whose baseline file lacks its answer is an error, not an exclusion.
    """
    unknown = set(categories) - set(CATEGORIES)
    if not categories or unknown:
        raise ValueError(f"unknown Arena-Hard v2.0 categories: {sorted(unknown)}")
    baselines, files = {}, []
    for category in CATEGORIES:
        if category not in categories:
            continue
        model, digest = BASELINES[category]
        name = f"model_answer/{model}.jsonl"
        path = _file(name, digest, data_dir)
        files.append(file_record(path, source=REPOSITORY, revision=REVISION, filename=name))
        baselines[category] = dict(_baseline_answer(row) for row in _rows(path))
    path = _file("question.jsonl", QUESTIONS_SHA256, data_dir)
    files.append(file_record(path, source=REPOSITORY, revision=REVISION, filename="question.jsonl"))
    samples = []
    for row in _rows(path):
        category = row.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"unexpected Arena-Hard v2.0 category: {category!r}")
        if category not in categories:
            continue
        uid, prompt = row.get("uid"), row.get("prompt")
        if not isinstance(uid, str) or not uid.strip() or not isinstance(prompt, str):
            raise ValueError("Arena-Hard questions need a UID and a prompt")
        if not prompt.strip():
            raise ValueError(f"Arena-Hard question {uid} has an empty prompt")
        model, digest = BASELINES[category]
        if uid not in baselines[category]:
            raise ValueError(f"baseline {model} has no answer for question {uid}")
        samples.append(
            Sample(
                id=f"arenahard-v2-{uid}",
                input=prompt,
                target=baselines[category][uid],
                metadata={
                    "benchmark": f"arenahard_v2_{category}",
                    "uid": uid,
                    "category": category,
                    "subcategory": row.get("subcategory"),
                    "baseline_model": model,
                    "baseline_sha256": digest,
                    "revision": REVISION,
                },
            )
        )
    return LoadedSource(samples, [], files)
