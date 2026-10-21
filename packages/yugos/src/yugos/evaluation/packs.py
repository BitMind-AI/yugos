"""Read and write benchmark packs (`pack.tar`, schema `ycore.pack.v2`).

A pack is an uncompressed tar of `records.jsonl` (one selected Sample record per line),
`manifest.json` (a `PackManifest`) and, for competition packs, the
`difficulty-calibration.json` evidence the manifest pins. Ycore publishes packs;
hosted executors, local runs and validators read them. Published calibration
summaries reproduce labels from their outcomes locally. Full native evidence
replay stays with the producer; historical artifacts retain their original contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .questions import Question

PACK_SCHEMA = "ycore.pack.v2"
RECORDS = "records.jsonl"
MANIFEST = "manifest.json"
CALIBRATION = "difficulty-calibration.json"
MAX_PACK_BYTES = 4 * 1024**3
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PackManifest(BaseModel):
    """What a pack contains and what it may be used for."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
    schema_name: Literal["ycore.pack.v2"] = Field(default=PACK_SCHEMA, alias="schema")
    pack_id: str = Field(min_length=1)
    records_sha256: Sha256
    source_releases: tuple[str, ...] = Field(min_length=1)
    data_visibility: Literal["public", "private"] | None = None
    purpose: Literal["development", "competition"]
    difficulty_calibration_sha256: Sha256 | None = None
    # Prices and admitted models the pack was prepared for, when specified.
    world_card_sha256: Sha256 | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_retired_split(cls, value):
        """`evaluation_split` is not part of this manifest. Older packs may still carry it."""
        if isinstance(value, dict) and "evaluation_split" in value:
            return {key: item for key, item in value.items() if key != "evaluation_split"}
        return value


@dataclass(frozen=True)
class Pack:
    """A verified pack: its manifest, questions, exact record bytes and archive digest."""

    manifest: PackManifest
    questions: tuple[Question, ...]
    records: bytes
    sha256: str
    difficulty_calibration: bytes | None = None


def canonical_json(value: Any, *, ensure_ascii: bool = True) -> bytes:
    """Sorted, compact JSON: the byte form packs hash and publish."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    ).encode("utf-8")


def records_bytes(questions: Iterable[Question]) -> bytes:
    """Serialize questions as `records.jsonl`, one canonical JSON object per line."""
    return b"".join(
        canonical_json(q.model_dump(mode="json"), ensure_ascii=False) + b"\n" for q in questions
    )


def read_questions(raw: bytes) -> tuple[Question, ...]:
    """Parse `records.jsonl`; a pack needs at least one question and unique IDs."""
    questions = tuple(
        Question.model_validate_json(line) for line in raw.splitlines() if line.strip()
    )
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("pack requires questions with unique IDs")
    return questions


def _check(manifest: PackManifest, records: bytes, calibration: bytes | None) -> None:
    if hashlib.sha256(records).hexdigest() != manifest.records_sha256:
        raise ValueError("pack records do not match their manifest")
    if manifest.purpose == "competition" and manifest.difficulty_calibration_sha256 is None:
        raise ValueError("competition pack requires difficulty calibration evidence")
    pinned = manifest.difficulty_calibration_sha256
    if (pinned is None) != (calibration is None) or (
        calibration is not None and hashlib.sha256(calibration).hexdigest() != pinned
    ):
        raise ValueError("pack calibration evidence digest mismatch")
    if calibration is not None and "schema_version" in json.loads(calibration):
        from ..competition.difficulty import CalibrationSummary, verify_summary

        verify_summary(CalibrationSummary.model_validate_json(calibration), read_questions(records))


def pack_bytes(
    manifest: PackManifest, records: bytes, *, difficulty_calibration: bytes | None = None
) -> bytes:
    """The exact `pack.tar` bytes for verified inputs; never rewrites the questions."""
    read_questions(records)
    _check(manifest, records, difficulty_calibration)
    files = [
        (RECORDS, records),
        (MANIFEST, canonical_json(manifest.model_dump(mode="json", by_alias=True))),
    ]
    if difficulty_calibration is not None:
        files.append((CALIBRATION, difficulty_calibration))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in files:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def write_pack(
    path: Path,
    manifest: PackManifest,
    records: bytes,
    *,
    difficulty_calibration: bytes | None = None,
) -> str:
    """Write a new pack file (never overwriting one) and return its SHA-256."""
    raw = pack_bytes(manifest, records, difficulty_calibration=difficulty_calibration)
    with Path(path).open("xb") as file:
        file.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_pack(path: Path) -> Pack:
    """Read and verify a pack: known members only, matching digests, valid questions."""
    path = Path(path)
    if path.stat().st_size > MAX_PACK_BYTES:
        raise ValueError("pack exceeds the size limit")
    raw = path.read_bytes()
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if member.name not in (RECORDS, MANIFEST, CALIBRATION) or not member.isfile():
                raise ValueError(f"unexpected pack member: {member.name!r}")
            if member.name in files:
                raise ValueError(f"duplicate pack member: {member.name}")
            files[member.name] = archive.extractfile(member).read()
    if RECORDS not in files or MANIFEST not in files:
        raise ValueError(f"pack must contain {RECORDS} and {MANIFEST}")
    manifest = PackManifest.model_validate_json(files[MANIFEST])
    calibration = files.get(CALIBRATION)
    _check(manifest, files[RECORDS], calibration)
    return Pack(
        manifest=manifest,
        questions=read_questions(files[RECORDS]),
        records=files[RECORDS],
        sha256=hashlib.sha256(raw).hexdigest(),
        difficulty_calibration=calibration,
    )
