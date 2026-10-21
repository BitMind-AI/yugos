"""Prepare bundles on CPU and consume operator-attested read-only worker inputs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .bundle import (
    MAX_MANIFEST_BYTES,
    BundleManifest,
    PreparationError,
    PreparedBundle,
    prepare_bundle,
)
from .protocol import PreparedMetadata


def prepare_to_directory(
    source: Path, destination: Path, *, submission_sha256: str, manifest_sha256: str | None = None
) -> PreparedBundle:
    """Verify/extract without importing miner code; the operator isolates this step."""
    destination.mkdir(parents=True, exist_ok=True)
    prepared = prepare_bundle(
        source,
        destination / "model",
        submission_sha256=submission_sha256,
        manifest_sha256=manifest_sha256,
    )
    temporary = destination / "metadata.partial"
    with temporary.open("x") as output:
        output.write(prepared.metadata.model_dump_json())
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination / "metadata.json")
    return prepared


def read_prepared_directory(
    destination: Path, *, submission_sha256: str, manifest_sha256: str | None = None
) -> PreparedBundle:
    """Read a completed operator staging directory mounted read-only into the worker.

    This is an attestation handoff, not validation of an arbitrary miner directory.
    The operator must stop the CPU preparer before mounting its output read-only.
    """
    root, metadata_path = destination / "model", destination / "metadata.json"
    try:
        if root.is_symlink() or metadata_path.is_symlink():
            raise ValueError("prepared inputs cannot be symlinks")
        with metadata_path.open("rb") as source:
            raw = source.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("prepared metadata exceeds size limit")
        metadata = PreparedMetadata.model_validate_json(raw)
        if metadata.submission_sha256 != submission_sha256:
            raise ValueError("prepared submission digest mismatch")
        if manifest_sha256 is not None and metadata.manifest_sha256 != manifest_sha256:
            raise ValueError("prepared manifest digest mismatch")
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("prepared manifest cannot be a symlink")
        with manifest_path.open("rb") as source:
            manifest_raw = source.read(MAX_MANIFEST_BYTES + 1)
        if len(manifest_raw) > MAX_MANIFEST_BYTES:
            raise ValueError("prepared manifest exceeds size limit")
        if hashlib.sha256(manifest_raw).hexdigest() != metadata.manifest_sha256:
            raise ValueError("prepared manifest differs from attestation")
        manifest = BundleManifest.model_validate_json(manifest_raw)
        if (metadata.name, metadata.version, metadata.runtime, metadata.required_models) != (
            manifest.name,
            manifest.version,
            manifest.runtime,
            manifest.required_models,
        ):
            raise ValueError("prepared metadata differs from manifest")
        return PreparedBundle(root=root, metadata=metadata)
    except (OSError, ValueError) as exc:
        raise PreparationError("preparation.digest_mismatch", str(exc)) from exc
