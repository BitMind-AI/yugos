"""Pinned CRUXEval output-prediction examples, without executing their programs."""

import json
from pathlib import Path

from inspect_ai.dataset import Sample

from ._common import LoadedSource
from ._fetch import file_record, hf_file

REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"


def load_cruxeval(*, data_dir: str | Path | None = None) -> LoadedSource:
    """Retain the function and input; the expected result stays in the gold target."""
    path = hf_file(
        "cruxeval-org/cruxeval",
        "test.jsonl",
        revision=REVISION,
        max_bytes=186873,
        data_dir=data_dir,
    )
    samples = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        samples.append(
            Sample(
                id=row["id"],
                input=(
                    "Predict the output of this Python function for the given input. "
                    "Return only a Python expression representing the output, without "
                    "calling f on the given input.\n\n"
                    f"```python\n{row['code']}\n```\n\nInput: f({row['input']})"
                ),
                target=row["output"],
                metadata={"code": row["code"], "input": row["input"]},
            )
        )
    return LoadedSource(
        samples,
        files=[
            file_record(
                path,
                source="cruxeval-org/cruxeval",
                revision=REVISION,
                filename="test.jsonl",
            )
        ],
    )
