"""Fetch pinned public source files and verify them before any row is read."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import httpx

_REVISION = re.compile(r"[0-9a-f]{40}")


class SourceAccessError(PermissionError):
    """A source refused access, e.g. a gated dataset without an accepted token."""


def cache_dir() -> Path:
    """Where fetched sources are kept: `$YUGOS_EVALS_CACHE`, else `~/.cache/yugos-evals`."""
    return Path(os.environ.get("YUGOS_EVALS_CACHE") or Path.home() / ".cache" / "yugos-evals")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, sha256: str | None) -> Path:
    """Return `path` if its content has the pinned digest (or no digest is pinned)."""
    if sha256 is not None:
        actual = sha256_file(path)
        if actual != sha256:
            raise ValueError(f"{path.name} has SHA-256 {actual}, not the pinned {sha256}")
    return path


def fetch(
    url: str,
    *,
    name: str,
    max_bytes: int,
    sha256: str | None = None,
    headers: dict[str, str] | None = None,
) -> Path:
    """Download `url` once into the cache and verify it; a mismatch is never cached.

    Without a content digest the cache is keyed by URL, which must then pin its
    content itself (e.g. a commit revision).
    """
    key = sha256 or "url-" + hashlib.sha256(url.encode()).hexdigest()
    target = cache_dir() / "sources" / key / name
    if target.exists():
        return verified(target, sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    size = 0
    try:
        with httpx.stream(
            "GET", url, headers=headers, follow_redirects=True, timeout=60
        ) as response:
            if response.status_code in (401, 403):
                raise SourceAccessError(f"{name}: access refused ({response.status_code})")
            response.raise_for_status()
            with partial.open("wb") as stream:
                for block in response.iter_bytes():
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError(f"{name} exceeds {max_bytes} bytes")
                    stream.write(block)
        verified(partial, sha256)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return partial.rename(target)


def hf_file(
    repo_id: str,
    filename: str,
    *,
    revision: str,
    max_bytes: int,
    sha256: str | None = None,
    data_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """A dataset file at a pinned Hugging Face commit, or its staged copy under `data_dir`."""
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"{repo_id} must be pinned to a commit, not {revision!r}")
    if data_dir is not None:
        return verified(Path(data_dir) / filename, sha256)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return fetch(url, name=Path(filename).name, max_bytes=max_bytes, sha256=sha256, headers=headers)


def file_record(path: Path, *, source: str, revision: str, filename: str) -> dict[str, str]:
    """Provenance for one file read: its source, pinned revision, name and actual digest."""
    return {
        "source": source,
        "revision": revision,
        "file": filename,
        "sha256": sha256_file(path),
    }
