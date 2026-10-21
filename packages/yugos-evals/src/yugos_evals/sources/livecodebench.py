"""LiveCodeBench code generation (Python) at a pinned commit, by release file and contest date.

Each upstream release adds one file; `release_vN` is the union of v1..vN, so
`releases=("v6",)` is the v6 additions only. The prompt adapts upstream's
Python instructions from public fields. The target holds every public and
private test (inputs and expected outputs) as JSON; it never reaches the model.
Test payloads are decoded without unpickling.
"""

from __future__ import annotations

import base64
import json
import pickletools
import re
import unicodedata
import zlib
from collections.abc import Collection, Iterator
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path

from inspect_ai.dataset import Sample

from ._common import LoadedSource
from ._fetch import file_record, hf_file

DATASET = "livecodebench/code_generation_lite"
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_FILES = {f"v{n}": "test.jsonl" if n == 1 else f"test{n}.jsonl" for n in range(1, 7)}
# Digests reviewed in Ycore's acquisition configs; other files are pinned by the commit.
FILE_SHA256 = {"test6.jsonl": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"}
_MAX_BYTES = 512 * 1024 * 1024
_MAX_DECOMPRESSED = 128 * 1024 * 1024


def prompt(content: str, starter: str) -> str:
    """Ycore's adaptation of upstream's Python instructions, from public fields only."""
    text = f"Write a Python solution to the following problem.\n\n{content}\n\n"
    if starter:
        text += (
            "Complete this starter code, preserving its entry point:\n"
            f"```python\n{starter}\n```\n\n"
        )
    else:
        text += (
            "Read the problem input from standard input and print the result to "
            "standard output. Your program must run when invoked; do not hardcode "
            "the sample inputs.\n\n"
        )
    return text + "Return your solution in a fenced Python code block."


def decode_test_cases(value: str) -> list[dict]:
    """Decode upstream JSON or compressed pickled JSON, refusing executable objects."""
    try:
        cases = json.loads(value)
    except json.JSONDecodeError:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(value, validate=True), _MAX_DECOMPRESSED)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("invalid or oversized LiveCodeBench test payload") from None
        # Upstream pickles a JSON string. Read literal opcodes only; never unpickle.
        text, end = None, 0
        for op, arg, position in pickletools.genops(raw):
            if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"} and text is None:
                text = arg
            elif op.name not in {"PROTO", "FRAME", "MEMOIZE", "STOP"}:
                raise ValueError("LiveCodeBench test payload must be a literal JSON string")
            if op.name == "STOP":
                end = position + 1
        if not isinstance(text, str) or end != len(raw):
            raise ValueError("LiveCodeBench test payload must contain one JSON string")
        cases = json.loads(text)
    if not isinstance(cases, list) or not cases:
        raise ValueError("LiveCodeBench requires nonempty public and private tests")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"input", "output", "testtype"}
            or not isinstance(case["input"], str)
            or not isinstance(case["output"], str)
            or case["testtype"] not in {"stdin", "functional"}
        ):
            raise ValueError("unsupported LiveCodeBench test case")
    return cases


def tests_target(cases: list[dict], fn_name: str) -> str:
    """The canonical test reference `code_tests_scorer` grades against."""
    return json.dumps(
        {"kind": "lcb", "cases": cases, "fn_name": fn_name},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def statement_key(text: str) -> str:
    """Rendered-prompt identity after Unicode and whitespace normalization."""
    return sha256(" ".join(unicodedata.normalize("NFKC", text).split()).encode()).hexdigest()


def jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def livecodebench_sample(row: dict, *, release: str) -> Sample:
    """One upstream problem as a sample, as Ycore's adapter builds it."""
    source_id, platform = row["question_id"], row["platform"]
    content, starter = row["question_content"], row["starter_code"]
    if (
        not isinstance(source_id, str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", source_id)
        or platform not in {"leetcode", "atcoder", "codeforces"}
        or not isinstance(content, str)
        or not content.strip()
        or not isinstance(starter, str)
    ):
        raise ValueError("invalid LiveCodeBench problem identity or prompt")
    metadata = json.loads(row["metadata"])
    if not isinstance(metadata, dict):
        raise TypeError("LiveCodeBench metadata must be an object")
    public = decode_test_cases(row["public_test_cases"])
    private = decode_test_cases(row["private_test_cases"])
    cases = public + private
    modes = {case["testtype"] for case in cases}
    fn_name = metadata.get("func_name")
    functional_ok = (
        isinstance(fn_name, str)
        and fn_name.isidentifier()
        and re.search(r"^\s*def\s+" + re.escape(fn_name) + r"\s*\(", starter, re.MULTILINE)
    )
    if (
        len(modes) != 1
        or (modes == {"functional"} and not functional_ok)
        or (modes == {"stdin"} and (fn_name or starter.strip()))
    ):
        raise ValueError("LiveCodeBench entry point, starter and test modes disagree")
    return Sample(
        id=f"livecodebench-{platform}-{source_id}",
        input=prompt(content, starter),
        target=tests_target(cases, fn_name or ""),
        metadata={
            "benchmark": f"livecodebench-{release}",
            "platform": platform,
            "contest_date": str(row["contest_date"]),
            "upstream_difficulty": row.get("difficulty"),
            "test_mode": next(iter(modes)),
            "public_test_count": len(public),
            "private_test_count": len(private),
        },
    )


def _date(value: str | date | None) -> date | None:
    return date.fromisoformat(value) if isinstance(value, str) else value


def load_livecodebench(
    releases: Collection[str] = ("v6",),
    *,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    data_dir: str | Path | None = None,
) -> LoadedSource:
    """Problems of the given release files whose contest date is in [start, end]."""
    unknown = set(releases) - RELEASE_FILES.keys()
    if not releases or unknown:
        raise ValueError(f"unknown LiveCodeBench releases: {sorted(unknown)}")
    start, end = _date(start_date), _date(end_date)
    samples, files = [], []
    for release in sorted(set(releases), key=lambda name: int(name[1:])):
        name = RELEASE_FILES[release]
        path = hf_file(
            DATASET,
            name,
            revision=REVISION,
            sha256=FILE_SHA256.get(name),
            max_bytes=_MAX_BYTES,
            data_dir=data_dir,
        )
        files.append(file_record(path, source=DATASET, revision=REVISION, filename=name))
        for row in jsonl(path):
            contest = datetime.fromisoformat(row["contest_date"]).date()
            if (start and contest < start) or (end and contest > end):
                continue
            samples.append(livecodebench_sample(row, release=release))
    return LoadedSource(samples, [], files)
