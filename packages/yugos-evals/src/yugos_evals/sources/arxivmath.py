"""MathArena ArXivMath monthly releases, pinned by commit and file digest.

Each month is one Hugging Face dataset with a single parquet file. Problems are
rendered with the boxed final-answer prompt; references Math-Verify cannot
check faithfully are excluded with a reason. Licence: CC BY-SA 4.0 (attribute
MathArena and the source papers).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from ._common import LoadedSource
from ._fetch import file_record, hf_file
from ._math import math_samples

DATA_FILE = "data/train-00000-of-00001.parquet"
# Columns across the published months; a new column must be reviewed, not ignored.
_REQUIRED = {"problem_idx", "answer", "problem", "source"}
_OPTIONAL = {"problem_type", "title", "authors"}


@dataclass(frozen=True)
class ArxivMathMonth:
    """One monthly release: dataset, commit, parquet SHA-256 and size."""

    dataset: str
    revision: str
    sha256: str
    size: int

    @property
    def yymm(self) -> str:
        """Upstream names months MMYY; identifiers use YYMM so they sort chronologically."""
        mmyy = self.dataset.rsplit("-", 1)[1]
        return mmyy[2:] + mmyy[:2]


# The months reviewed in Ycore's acquisition config. July 2026 was not published.
MONTHS = tuple(
    ArxivMathMonth(f"MathArena/arxivmath-{mmyy}", revision, digest, size)
    for mmyy, revision, digest, size in (
        ("1225", "90c60aa1eadbb72168640169d82154b079cdeaac", "e95cc6bd7c5677077ad6cc8b28d1ce5702728265b2bd2fde3877a2ec2e71146e", 12584),
        ("0126", "bac939ba603c93f905aa48072980ea681a2824cd", "44a802bcc18e4a2154d73f2917c82bf7353ca719a9313aa08d07d4483f1cbd29", 13465),
        ("0226", "a038fd1b01405ea1102e165b6ef43693fd16ca6b", "f838d0d95f1a4b113961ea9d7dd0c810900a87f56c945bd9087e5639f60aecff", 14136),
        ("0326", "14ff32983ae4c72b75fb64eba750f3c27f46607a", "ba4e5c97ba84976970b5505ea42fb10b62411691d23132b39d6f403bc549e0e8", 18523),
        ("0426", "fc2cb715b5855c100142e4c05a8042351c21f3c8", "9e7b84a2aaa05800d37ff3b22a40b3f64451cb6f10d16b41a5538fb19a58cc5a", 20156),
        ("0526", "69f701c7f266bfba235ef302ff9b19093f7455bb", "2c2375ae41cb434bc876505c3ae347e4e0bef421358e435807606bd312bde52b", 20763),
        ("0626", "dca0d771ab20c1b8e8c50d0921bf693f7bfe663d", "67e038c57192b6c2aaf5131d6877ce82684551ae44282c17c3906cef68de7446", 22770),
        ("0826", "977a3a0083198e11ac4f20fe79288351cebd6281", "a635931866406bbe7eb5cee8a4e47e188634b3c25b0b0b83dc06d36435e75609", 55180),
    )
)  # fmt: skip


def _arxiv_id(value) -> str:
    """Early months store arXiv IDs as floats, which drop trailing zeros."""
    text = f"{value:.5f}" if isinstance(value, float) else str(value)
    if not re.fullmatch(r"[0-9]{4}\.[0-9]{5}", text):
        raise ValueError(f"invalid ArXivMath source paper ID: {value!r}")
    return text


def _rows(month: ArxivMathMonth, path: Path):
    import pyarrow.parquet as pq

    indices = set()
    for row in pq.read_table(path).to_pylist():
        index = row.get("problem_idx")
        if (
            not _REQUIRED <= set(row) <= _REQUIRED | _OPTIONAL
            or type(index) is not int
            or index <= 0
            or index in indices
            or not isinstance(row["answer"], str)
            or not isinstance(row["problem"], str)
        ):
            raise ValueError(f"invalid ArXivMath row in {month.dataset}")
        indices.add(index)
        release = f"20{month.yymm[:2]}-{month.yymm[2:]}"
        metadata = {
            "benchmark": f"arxivmath-{month.yymm}",
            "release_month": release,
            "dataset": month.dataset,
            "revision": month.revision,
            "arxiv_id": _arxiv_id(row["source"]),
        }
        yield f"arxivmath-{month.yymm}-{index}", row["problem"], row["answer"], metadata


def load_arxivmath(
    months: Collection[str] | None = None, *, data_dir: str | Path | None = None
) -> LoadedSource:
    """Samples from the pinned months (`YYMM`, default all), and the references excluded.

    `data_dir` holds staged copies as `<dataset>/<file>`, e.g.
    `MathArena/arxivmath-1225/data/train-00000-of-00001.parquet`.
    """
    selected = [m for m in MONTHS if months is None or m.yymm in months]
    if months is not None and (not months or len(selected) != len(set(months))):
        raise ValueError(f"unknown ArXivMath months: {sorted(set(months))}")
    samples, excluded, files = [], [], []
    for month in selected:
        path = hf_file(
            month.dataset,
            DATA_FILE,
            revision=month.revision,
            sha256=month.sha256,
            max_bytes=month.size,
            data_dir=None if data_dir is None else Path(data_dir) / month.dataset,
        )
        files.append(
            file_record(path, source=month.dataset, revision=month.revision, filename=DATA_FILE)
        )
        kept, dropped = math_samples(_rows(month, path))
        samples += kept
        excluded += dropped
    return LoadedSource(samples, excluded, files)
