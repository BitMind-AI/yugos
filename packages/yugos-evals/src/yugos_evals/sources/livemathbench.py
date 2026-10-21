"""OpenCompass LiveMathBench releases at a pinned commit (a gated Hugging Face dataset).

Accept the dataset's terms on the Hub and set `HF_TOKEN`, or stage the files
under `data_dir`. Problems use the boxed final-answer prompt; references
Math-Verify cannot check faithfully are excluded with a reason.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection
from pathlib import Path

from ._common import LoadedSource
from ._fetch import SourceAccessError, file_record, hf_file
from ._math import math_samples

DATASET = "opencompass/LiveMathBench"
REVISION = "ea05b95ab431f3ef3900ce86f864451cf5bcd699"
# `<release>/<file stem>` subsets at REVISION.
SUBSETS = frozenset(
    {"202505/all_en"}
    | {
        f"202412/{name}_{lang}"
        for name in ("AMC", "CCEE", "CNMO", "WLPMC", "hard")
        for lang in ("en", "cn")
    }
)
DEFAULT_SUBSETS = ("202505/all_en",)
_MAX_BYTES = 16 * 1024 * 1024
_GATED = (
    f"{DATASET} is gated: accept its terms on the Hugging Face Hub and set HF_TOKEN, "
    "or stage the files with data_dir"
)


def load_livemathbench(
    subsets: Collection[str] = DEFAULT_SUBSETS, *, data_dir: str | Path | None = None
) -> LoadedSource:
    """Samples from LiveMathBench subsets (`<release>/<file stem>`), and the references excluded."""
    unknown = set(subsets) - SUBSETS
    if not subsets or unknown:
        raise ValueError(f"unknown LiveMathBench subsets: {sorted(unknown)}")
    token = None
    if data_dir is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SourceAccessError(_GATED)
    samples, excluded, files = [], [], []
    for subset in sorted(subsets):
        filename = f"{subset}.jsonl"
        try:
            path = hf_file(
                DATASET,
                filename,
                revision=REVISION,
                max_bytes=_MAX_BYTES,
                data_dir=data_dir,
                token=token,
            )
        except SourceAccessError:
            raise SourceAccessError(_GATED) from None
        files.append(file_record(path, source=DATASET, revision=REVISION, filename=filename))
        name = "livemathbench-" + re.sub(r"[^A-Za-z0-9]+", "-", subset).lower()
        with path.open(encoding="utf-8", newline="") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        rows = []
        for index, row in enumerate(records, 1):
            if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
                raise ValueError(f"LiveMathBench {subset} row {index} lacks a question or answer")
            metadata = {"benchmark": name, "subset": subset, "revision": REVISION}
            rows.append((f"{name}-{index}", row["question"], row["answer"], metadata))
        kept, dropped = math_samples(rows)
        samples += kept
        excluded += dropped
    return LoadedSource(samples, excluded, files)
