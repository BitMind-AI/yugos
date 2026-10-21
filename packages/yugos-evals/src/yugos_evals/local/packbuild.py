"""Build a development pack from one catalog benchmark.

Samples are chosen deterministically from the public task and frozen with the
catalog's `freeze_entry` (core's `freeze_task` plus the entry's role pins), so a
local pack runs exactly like a hosted one. It is a
development pack: public questions, no private holdouts, no difficulty
calibration, so its scores are not comparable to competition packs.
"""

from __future__ import annotations

import hashlib
import random
from importlib.metadata import version
from pathlib import Path

from yugos.evaluation.packs import PackManifest, records_bytes, write_pack

from ..catalog import entry, freeze_entry, load_task


def select_sample_ids(ids: list[str], *, limit: int | None, seed: int) -> list[str]:
    """`limit` IDs drawn with `seed`, kept in dataset order; every ID without a limit."""
    if limit is None or limit >= len(ids):
        return list(ids)
    if limit < 1:
        raise ValueError("limit must be positive")
    chosen = set(random.Random(seed).sample(ids, limit))
    return [i for i in ids if i in chosen]


def build_local_pack(
    benchmark: str,
    out: Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    task_args: dict | None = None,
) -> tuple[str, PackManifest]:
    """Freeze a selection of the benchmark's samples into a development pack at `out`."""
    selected = entry(benchmark)
    task = load_task(selected, **(task_args or {}))
    ids = [str(sample.id) for sample in task.dataset]
    try:
        questions = freeze_entry(
            selected, task=task, sample_ids=select_sample_ids(ids, limit=limit, seed=seed)
        )
    except ValueError as exc:
        if "sandbox" in str(exc):
            raise ValueError(
                f"{benchmark} needs prepared, digest-pinned sandbox images; "
                f"sandboxed benchmarks cannot be packed locally yet ({exc})"
            ) from exc
        raise
    records = records_bytes(questions)
    digest = hashlib.sha256(records).hexdigest()
    package = selected.source.replace("_", "-")
    manifest = PackManifest(
        pack_id=f"local-{selected.id}-{digest[:12]}",
        records_sha256=digest,
        source_releases=(f"{selected.task}@{package}=={version(package)}",),
        data_visibility="public",
        purpose="development",
    )
    return write_pack(Path(out), manifest, records), manifest
