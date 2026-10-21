"""What a source loader returns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from inspect_ai.dataset import Dataset, MemoryDataset, Sample


def with_sample_ids(dataset: Dataset) -> Dataset:
    """Identify unnamed public samples by input and choices, without exposing targets.

    Keep upstream IDs intact. Identical inputs still need distinct upstream IDs;
    the freezer rejects collisions rather than using unstable row numbers.
    """
    samples = list(dataset)
    if all(sample.id is not None for sample in samples):
        return dataset
    identified = []
    for sample in samples:
        if sample.id is None:
            fields = sample.model_dump(mode="json", include={"input", "choices"})
            raw = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            identity = "sample-" + hashlib.sha256(raw.encode()).hexdigest()
            sample = sample.model_copy(update={"id": identity})
        identified.append(sample)
    return MemoryDataset(
        identified, name=dataset.name, location=dataset.location, shuffled=dataset.shuffled
    )


@dataclass(frozen=True)
class LoadedSource:
    """Samples from pinned public files, the records left out and why, and file provenance.

    `excluded` rows are `{"id": ..., "reason": ...}`; `files` rows identify each
    file read (`url` or staged `path`, pinned `revision` and actual `sha256`).
    """

    samples: list[Sample]
    excluded: list[dict[str, str]] = field(default_factory=list)
    files: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        ids = [str(sample.id) for sample in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("source samples require unique IDs")

    def metadata(self) -> dict:
        """Task metadata recording this source's provenance and exclusions."""
        return {"sources": list(self.files), "excluded": list(self.excluded)}
