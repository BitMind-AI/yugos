"""BenchEvolver LiveCodeBench-Plus medium/hard at a pinned commit, minus LiveCodeBench overlap.

Every Plus problem is stdin/stdout; all public and hidden tests are kept, public
first. Problems already in the excluded LiveCodeBench releases (by default the
v6 additions) are dropped: an original keeps its AtCoder ID, and an evolved
problem overlaps when its normalized prompt matches.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path

from inspect_ai.dataset import Sample

from ._common import LoadedSource
from ._fetch import file_record, hf_file
from .livecodebench import jsonl, load_livecodebench, prompt, statement_key, tests_target

DATASET = "BenchEvolver/livecodebench-plus"
REVISION = "856b9fdef2d8c5d368db7dff610e1b545bef546a"
SPLITS = ("medium", "hard")
FILE_SHA256 = {
    "medium.jsonl": "101e054c6b0c8becf2f6b15bbccfa063ad98ef1951bd00960de7e3466d82eed2",
    "hard.jsonl": "bbc2947d15e05e6cf51ac38739de1c3039e52e487efac4c3139966f263143953",
}
_MAX_BYTES = 512 * 1024 * 1024
_FIELDS = {
    "problem_id",
    "title",
    "difficulty",
    "problem_statement",
    "public_tests",
    "test_cases",
    "solution_code",
    "mutation_type",
    "approach",
    "mutation_notes",
    "seed_problem",
}


def _cases(row: dict) -> tuple[list[dict], int]:
    cases, public = row["test_cases"], row["public_tests"]
    if (
        not isinstance(cases, list)
        or not isinstance(public, list)
        or any(
            not isinstance(case, dict)
            or set(case) != {"input", "output", "tier"}
            or not isinstance(case["input"], str)
            or not isinstance(case["output"], str)
            for case in cases
        )
    ):
        raise ValueError("unsupported LiveCodeBench-Plus test case")
    tiers = [case["tier"] for case in cases]
    count = tiers.count("public")
    if (
        not public
        or cases[:count] != public
        or tiers != ["public"] * count + ["private"] * (len(cases) - count)
        or count == len(cases)
    ):
        raise ValueError("LiveCodeBench-Plus requires public tests followed by hidden tests")
    converted = [{"input": c["input"], "output": c["output"], "testtype": "stdin"} for c in cases]
    return converted, count


def livecodebench_plus_sample(row: dict, *, split: str) -> Sample:
    """One Plus problem as a stdin sample, as Ycore's adapter builds it."""
    if set(row) != _FIELDS:
        raise ValueError("unexpected LiveCodeBench-Plus fields")
    source_id, statement = row["problem_id"], row["problem_statement"]
    evolved = row["seed_problem"] is not None
    if (
        not isinstance(source_id, str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", source_id)
        or row["difficulty"] != split
        or not isinstance(statement, str)
        or not statement.strip()
        or evolved != bool(row["mutation_type"])
        or evolved != bool(row["solution_code"])
        or (not evolved and not re.fullmatch(r"(abc|arc|agc)[0-9]+_[a-z][0-9]*", source_id))
    ):
        raise ValueError("invalid LiveCodeBench-Plus problem identity")
    cases, public = _cases(row)
    return Sample(
        id=f"livecodebench-plus-{source_id}",
        input=prompt(statement, ""),
        target=tests_target(cases, ""),
        metadata={
            "benchmark": "livecodebench-plus",
            "split": split,
            "origin": "benchevolver" if evolved else "livecodebench",
            "mutation_type": row["mutation_type"],
            "public_test_count": public,
            "private_test_count": len(cases) - public,
        },
    )


def load_livecodebench_plus(
    splits: Collection[str] = SPLITS,
    *,
    exclude_releases: Collection[str] = ("v6",),
    data_dir: str | Path | None = None,
    lcb_data_dir: str | Path | None = None,
) -> LoadedSource:
    """Plus problems of `splits`, excluding any already in LiveCodeBench `exclude_releases`."""
    if not splits or set(splits) - set(SPLITS):
        raise ValueError("LiveCodeBench-Plus splits are medium and hard")
    overlap = (
        load_livecodebench(exclude_releases, data_dir=lcb_data_dir)
        if exclude_releases
        else LoadedSource([])
    )
    overlap_ids = {str(sample.id) for sample in overlap.samples}
    overlap_prompts = {statement_key(sample.input): str(sample.id) for sample in overlap.samples}
    samples, excluded, files = [], [], list(overlap.files)
    for split in SPLITS:
        if split not in splits:
            continue
        name = f"{split}.jsonl"
        path = hf_file(
            DATASET,
            name,
            revision=REVISION,
            sha256=FILE_SHA256[name],
            max_bytes=_MAX_BYTES,
            data_dir=data_dir,
        )
        files.append(file_record(path, source=DATASET, revision=REVISION, filename=name))
        for row in jsonl(path):
            sample = livecodebench_plus_sample(row, split=split)
            original = f"livecodebench-atcoder-{row['problem_id']}"
            matched = (
                original
                if row["seed_problem"] is None and original in overlap_ids
                else overlap_prompts.get(statement_key(sample.input))
            )
            if matched:
                excluded.append(
                    {
                        "id": str(sample.id),
                        "reason": f"overlap: already in LiveCodeBench as {matched}",
                    }
                )
                continue
            samples.append(sample)
    return LoadedSource(samples, excluded, files)
