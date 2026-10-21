"""Merge verified packs into one pack.

Every input is read and checked before anything is written. Merged records are
sorted by question id. Calibration bytes are kept when every input carries the
same bytes; a disagreement fails instead of dropping a member. Passing
`calibration` replaces that agreed member.

Inputs are pack files, or one directory. A directory contributes the `.tar`
files directly inside it (including `.pack.tar`), sorted by filename.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from yugos.evaluation import Question
from yugos.evaluation.packs import Pack, PackManifest, read_pack, records_bytes, write_pack


@dataclass(frozen=True)
class AssembledPack:
    """Digest, manifest, and question counts by source benchmark."""

    sha256: str
    manifest: PackManifest
    question_counts: tuple[tuple[str, int], ...]

    @property
    def question_count(self) -> int:
        return sum(count for _, count in self.question_counts)


def resolve_pack_inputs(inputs: Sequence[Path]) -> list[Path]:
    """Pack files in the order given, or the `.tar` files in one directory.

    Directory matches are regular files directly in that directory whose names
    end with `.tar`, including `.pack.tar`. They are sorted by filename. Other
    entries are ignored. Files and directories cannot be mixed.
    """
    paths = [Path(path) for path in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        found = sorted(
            (path for path in paths[0].iterdir() if path.is_file() and path.name.endswith(".tar")),
            key=lambda path: path.name,
        )
        if not found:
            raise ValueError(f"no .tar packs in {paths[0]}")
        if len(found) < 2:
            raise ValueError(f"{paths[0]} has 1 .tar pack; assemble requires at least two")
        return found
    if any(path.is_dir() for path in paths):
        raise ValueError("pass pack tar files, or a single directory of .tar packs")
    if len(paths) < 2:
        raise ValueError("assemble requires at least two packs")
    return paths


def assemble_packs(
    inputs: Sequence[Path],
    out: Path,
    *,
    calibration: Path | None = None,
) -> AssembledPack:
    """Merge `inputs` into a new pack at `out`."""
    paths = resolve_pack_inputs(inputs)
    packs = [read_pack(path) for path in paths]
    _require_uniform(packs, "purpose")
    visibility = _agreed_visibility(packs)
    questions = _merged_questions(packs, paths)
    records = records_bytes(questions)
    evidence = _calibration(packs, calibration)
    digest = hashlib.sha256(records).hexdigest()
    manifest = PackManifest(
        pack_id=f"local-assembled-{digest[:12]}",
        records_sha256=digest,
        source_releases=_source_releases(packs),
        data_visibility=visibility,
        purpose=packs[0].manifest.purpose,
        difficulty_calibration_sha256=(
            None if evidence is None else hashlib.sha256(evidence).hexdigest()
        ),
        world_card_sha256=_shared_hash(pack.manifest.world_card_sha256 for pack in packs),
    )
    counts = Counter(question.source_benchmark for question in questions)
    sha = write_pack(Path(out), manifest, records, difficulty_calibration=evidence)
    return AssembledPack(
        sha256=sha,
        manifest=manifest,
        question_counts=tuple(sorted(counts.items())),
    )


def _agreed_visibility(packs: Sequence[Pack]) -> Literal["public", "private"] | None:
    """The visibility every set input shares. Null means unset, not a value.

    All-null stays null. One distinct non-null value is that value. Two
    different non-null values fail.
    """
    values = {
        pack.manifest.data_visibility for pack in packs if pack.manifest.data_visibility is not None
    }
    if len(values) > 1:
        shown = ", ".join(repr(value) for value in sorted(values))
        raise ValueError(f"mixed data_visibility: {shown}")
    if len(values) == 1:
        return next(iter(values))
    return None


def _require_uniform(packs: Sequence[Pack], field: str) -> None:
    values = {getattr(pack.manifest, field) for pack in packs}
    if len(values) == 1:
        return
    shown = ", ".join(repr(value) for value in sorted(values, key=_sort_key))
    raise ValueError(f"mixed {field}: {shown}")


def _sort_key(value: str | None) -> tuple[bool, str]:
    return (value is None, "" if value is None else value)


def _merged_questions(packs: Sequence[Pack], paths: Sequence[Path]) -> list[Question]:
    seen: dict[str, Path] = {}
    questions = []
    for pack, path in zip(packs, paths, strict=True):
        for question in pack.questions:
            prior = seen.get(question.question_id)
            if prior is not None:
                raise ValueError(
                    f"duplicate question_id {question.question_id!r} in {path} and {prior}"
                )
            seen[question.question_id] = path
            questions.append(question)
    questions.sort(key=lambda question: question.question_id)
    return questions


def _source_releases(packs: Sequence[Pack]) -> tuple[str, ...]:
    releases = {release for pack in packs for release in pack.manifest.source_releases}
    return tuple(sorted(releases))


def _shared_hash(values: Iterable[str | None]) -> str | None:
    """The hash when every input has that same hash, otherwise none."""
    unique = set(values)
    if len(unique) == 1:
        return next(iter(unique))
    return None


def _calibration(packs: Sequence[Pack], calibration: Path | None) -> bytes | None:
    """Agreed calibration bytes, or the file when one is passed.

    Inputs must all carry the same bytes, or all omit them. A disagreement
    fails instead of dropping a member the assembled pack cannot represent.
    """
    blobs = [pack.difficulty_calibration for pack in packs]
    first = blobs[0]
    if any(blob != first for blob in blobs):
        raise ValueError("input packs disagree on difficulty calibration")
    if calibration is not None:
        return Path(calibration).read_bytes()
    return first
