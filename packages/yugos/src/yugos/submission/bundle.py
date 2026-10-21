"""Verify and prepare uploaded model files inside an operator-provided sandbox."""

from __future__ import annotations

import hashlib
import math
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from yugos.inference.actions import RequiredModels

from .protocol import Digest, Failure, PreparedMetadata, StrictModel

BUNDLE_SCHEMA = "yugos.bundle.v3"
MAX_BUNDLE_BYTES = 24 * 1024 * 1024 * 1024
MAX_EXPANDED_BYTES = MAX_BUNDLE_BYTES
MAX_MEMBERS = 4096
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PARAMETERS = 10_000_000_000


class BundleManifest(StrictModel):
    """All model files are committed; the worker loads model.py with safetensors weights."""

    schema_name: Literal[BUNDLE_SCHEMA] = Field(alias="schema", default=BUNDLE_SCHEMA)
    name: str = Field(min_length=1, max_length=256)
    version: str = Field(default="0.1.0", min_length=1, max_length=128)
    runtime: Literal["python-safetensors-v2"]
    files: dict[str, Digest] = Field(min_length=2)
    required_models: RequiredModels

    @model_validator(mode="after")
    def required_files(self):
        if not {"model.py", "model.safetensors"}.issubset(self.files):
            raise ValueError("model.py and model.safetensors must be listed with their hashes")
        return self


@dataclass(frozen=True)
class PreparedBundle:
    root: Path
    metadata: PreparedMetadata


class PreparationError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.failure = Failure(code=code, detail=detail[:1000])


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or "\\" in name
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or ":" in path.parts[0]
        or path.as_posix() != name
        or name == "."
    ):
        raise ValueError(f"illegal bundle path: {name}")
    return name


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Reject ambiguous paths and oversized archives before reading payloads."""
    members = {}
    expanded = 0
    for member in archive.infolist():
        name = _safe_member(member.filename.rstrip("/") if member.is_dir() else member.filename)
        if member.orig_filename != member.filename or name in members:
            raise ValueError("ambiguous or duplicate archive member")
        expected_type = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
        if stat.S_IFMT(member.external_attr >> 16) not in (0, expected_type):
            raise ValueError("only regular files and directories are allowed")
        if member.flag_bits & 1:
            raise ValueError("encrypted archive members are not supported")
        expanded += member.file_size
        if expanded > MAX_EXPANDED_BYTES or len(members) >= MAX_MEMBERS:
            raise ValueError("archive exceeds the expansion or member limit")
        members[name] = member
    for name in members:
        for parent in PurePosixPath(name).parents:
            ancestor = members.get(str(parent))
            if ancestor is not None and not ancestor.is_dir():
                raise ValueError("archive has a file/directory conflict")
    manifest = members.get("manifest.json")
    if manifest is None or manifest.is_dir() or manifest.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest.json must be a bounded file at the archive root")
    return members


def _read_manifest(archive: zipfile.ZipFile) -> tuple[bytes, BundleManifest]:
    members = _archive_members(archive)
    raw = archive.read("manifest.json")
    manifest = BundleManifest.model_validate_json(raw)
    files = {name for name, member in members.items() if not member.is_dir()}
    if set(manifest.files) != files - {"manifest.json"}:
        raise ValueError("manifest must list every payload file exactly once")
    return raw, manifest


def read_bundle_manifest(source: Path) -> BundleManifest:
    """Read bounded declarations without scanning payloads or attesting their hashes.

    Callers must still prepare the bundle in isolation before executing it.
    """
    if source.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("bundle exceeds the upload limit")
    with zipfile.ZipFile(source) as archive:
        return _read_manifest(archive)[1]


def inspect_bundle(source: Path) -> PreparedMetadata:
    """Check a miner's ZIP and file hashes without extracting or importing its code.

    Tensor validation and execution remain the isolated worker's responsibility.
    """
    if source.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("bundle exceeds the upload limit")
    with zipfile.ZipFile(source) as archive:
        raw, manifest = _read_manifest(archive)
        for name, expected in manifest.files.items():
            with archive.open(name) as payload:
                actual = hashlib.file_digest(payload, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"file digest mismatch: {name}")
    return PreparedMetadata(
        submission_sha256=sha256_file(source),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        name=manifest.name,
        version=manifest.version,
        required_models=manifest.required_models,
    )


def _extract(archive: zipfile.ZipFile, destination: Path) -> dict[str, str]:
    """Check every ZIP entry before writing into a new, private directory."""
    members = _archive_members(archive)
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    digests = {}
    for name, member in members.items():
        target = destination / name
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        digest = hashlib.sha256()
        with archive.open(member) as source, target.open("xb") as output:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                total += len(chunk)
                if size > member.file_size or total > MAX_EXPANDED_BYTES:
                    raise ValueError("archive exceeds the expanded size limit")
                output.write(chunk)
                digest.update(chunk)
        digests[name] = digest.hexdigest()
        if size != member.file_size:
            raise ValueError("archive member size mismatch")
    return digests


def prepare_bundle(
    source: Path, destination: Path, *, submission_sha256: str, manifest_sha256: str | None = None
) -> PreparedBundle:
    """Verify hashes and tensor structure without importing any uploaded code.

    The operator must provide isolation and private staging/scratch storage.
    Static file validation is not a sandbox.
    """
    try:
        with source.open("rb") as upload:
            if source.stat().st_size > MAX_BUNDLE_BYTES:
                raise ValueError("bundle exceeds the upload limit")
            actual_submission = hashlib.file_digest(upload, "sha256").hexdigest()
            if actual_submission != submission_sha256:
                raise PreparationError("preparation.digest_mismatch", "submission digest mismatch")
            upload.seek(0)
            with zipfile.ZipFile(upload) as archive:
                digests = _extract(archive, destination)
    except (ValueError, NotImplementedError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PreparationError("preparation.archive_invalid", str(exc)) from exc
    manifest_bytes = (destination / "manifest.json").read_bytes()
    actual_manifest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 is not None and actual_manifest != manifest_sha256:
        raise PreparationError("preparation.digest_mismatch", "manifest digest mismatch")
    try:
        manifest = BundleManifest.model_validate_json(manifest_bytes)
        if set(manifest.files) != digests.keys() - {"manifest.json"}:
            raise ValueError("manifest must list every payload file exactly once")
    except ValueError as exc:
        raise PreparationError("preparation.manifest_invalid", str(exc)) from exc
    for name, digest in manifest.files.items():
        if digests[name] != digest:
            raise PreparationError("preparation.digest_mismatch", f"file digest mismatch: {name}")

    # This reader inspects tensor shapes without materializing model weights.
    from safetensors import safe_open

    try:
        with safe_open(str(destination / "model.safetensors"), framework="np") as weights:
            if not weights.keys():
                raise ValueError("weights must contain at least one tensor")
            parameters = sum(
                math.prod(weights.get_slice(key).get_shape()) for key in weights.keys()
            )
            if parameters > MAX_PARAMETERS:
                raise ValueError("model exceeds the parameter limit")
    except Exception as exc:
        raise PreparationError(
            "preparation.weights_invalid", "invalid or oversized model weights"
        ) from exc

    for path in sorted(destination.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return PreparedBundle(
        root=destination,
        metadata=PreparedMetadata(
            submission_sha256=actual_submission,
            manifest_sha256=actual_manifest,
            name=manifest.name,
            version=manifest.version,
            required_models=manifest.required_models,
        ),
    )
