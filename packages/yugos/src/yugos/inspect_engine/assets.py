"""Retain sandbox inputs and validate prepared, immutable Inspect environments."""

import base64
import re
from pathlib import Path

from inspect_ai.util import SandboxEnvironmentSpec


def frozen_sandbox(spec):
    """Accept a prepared Compose configuration; never build an upstream Dockerfile at eval time."""
    if spec is None:
        return None
    value = spec.model_dump(mode="json")
    config = value.get("config")
    if not isinstance(config, dict) or not config.get("services"):
        raise ValueError("sandbox needs a prepared ComposeConfig with immutable images")
    for service in config["services"].values():
        if service.get("build") or not re.fullmatch(
            r"[^\s@]+@sha256:[0-9a-f]{64}", service.get("image") or ""
        ):
            raise ValueError(
                "sandbox images must be built and digest-pinned before pack preparation"
            )
        if service.get("volumes") or service.get("env_file"):
            raise ValueError(
                "sandbox host mounts and env files must be materialized before packing"
            )
    if value["type"] == "local":
        raise ValueError("untrusted task execution requires isolation")
    return value


def sandbox_spec(value):
    return SandboxEnvironmentSpec.model_validate(value) if value else None


def inline_file(value: str) -> str:
    """Freeze local sample files into Inspect's supported inline-data representation."""
    if value.startswith("data:"):
        return value
    if "\n" not in value and "://" in value:
        raise ValueError("sample files must be acquired before pack preparation")
    contents = value.encode()
    try:
        path = Path(value)
        if "\n" not in value and path.is_file():
            contents = path.read_bytes()
        elif "\n" not in value and value.startswith(("/", "./", "../")):
            raise ValueError("sample file does not exist")
    except OSError:
        pass
    return "data:application/octet-stream;base64," + base64.b64encode(contents).decode()
