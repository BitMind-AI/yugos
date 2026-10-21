"""Worker messages over bounded, newline-delimited JSON."""

from __future__ import annotations

import json
from typing import Annotated, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from yugos.inference.actions import CallAction, CallOutput, RequiredModels
from yugos.inference.chat import ModelInput

SCHEMA = "yugos.worker.v4"
MAX_FRAME_BYTES = 4 * 1024 * 1024
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Failure(StrictModel):
    code: Literal[
        "preparation.archive_invalid",
        "preparation.manifest_invalid",
        "preparation.dependencies_unsupported",
        "preparation.weights_invalid",
        "preparation.digest_mismatch",
        "preparation.internal",
        "handshake.protocol",
        "load.failed",
        "request.failed",
        "request.invalid_action",
        "request.protocol",
    ]
    detail: str = Field(default="", max_length=1000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def request_matches_phase(self):
        if self.code.startswith("request.") != (self.request_id is not None):
            raise ValueError(
                "request failures require a request ID; other failures cannot have one"
            )
        return self

    @property
    def owner(self) -> Literal["submission", "infrastructure"]:
        if self.code in {"preparation.internal", "handshake.protocol"}:
            return "infrastructure"
        return "submission"


class PreparedMetadata(StrictModel):
    """Verified identity and declarations, returned before any submission code loads."""

    submission_sha256: Digest
    manifest_sha256: Digest
    name: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    runtime: Literal["python-safetensors-v2"] = "python-safetensors-v2"
    required_models: RequiredModels


class Envelope(StrictModel):
    schema_name: Literal[SCHEMA] = Field(alias="schema", default=SCHEMA)
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=0)


class Start(Envelope):
    kind: Literal["start"] = "start"
    submission_sha256: Digest
    expected_manifest_sha256: Digest | None = None
    max_active_requests: int = Field(default=1, ge=1, le=8, strict=True)


class Prepared(Envelope):
    kind: Literal["prepared"] = "prepared"
    metadata: PreparedMetadata


class Load(Envelope):
    kind: Literal["load"] = "load"


class Loaded(Envelope):
    kind: Literal["loaded"] = "loaded"


class Solve(Envelope):
    kind: Literal["solve"] = "solve"
    request_id: str = Field(min_length=1, max_length=128)
    input: ModelInput


class Resume(Envelope):
    kind: Literal["resume"] = "resume"
    request_id: str = Field(min_length=1, max_length=128)
    output: CallOutput


class Cancel(Envelope):
    kind: Literal["cancel"] = "cancel"
    request_id: str = Field(min_length=1, max_length=128)


class CallRequested(Envelope):
    kind: Literal["call_requested"] = "call_requested"
    request_id: str = Field(min_length=1, max_length=128)
    action: CallAction


class Finished(Envelope):
    kind: Literal["finished"] = "finished"
    request_id: str = Field(min_length=1, max_length=128)


class Failed(Envelope):
    kind: Literal["failed"] = "failed"
    failure: Failure


class Shutdown(Envelope):
    kind: Literal["shutdown"] = "shutdown"


class Stopped(Envelope):
    kind: Literal["stopped"] = "stopped"


Message = Annotated[
    Start
    | Prepared
    | Load
    | Loaded
    | Solve
    | Resume
    | Cancel
    | CallRequested
    | Finished
    | Failed
    | Shutdown
    | Stopped,
    Field(discriminator="kind"),
]
_ADAPTER = TypeAdapter(Message)


def encode_line(message: Message) -> str:
    payload = json.dumps(
        message.model_dump(mode="json", by_alias=True, exclude_none=True),
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > MAX_FRAME_BYTES:
        raise ValueError("worker message exceeds the size limit")
    return payload + "\n"


def decode_line(line: str) -> Message:
    if not line.endswith("\n") or len(line[:-1].encode("utf-8")) > MAX_FRAME_BYTES:
        raise ValueError("worker message is incomplete or oversized")
    return _ADAPTER.validate_python(json.loads(line))


def read_message(stream: TextIO) -> Message:
    line = stream.readline(MAX_FRAME_BYTES + 2)
    if not line:
        raise EOFError("worker stream closed")
    return decode_line(line)


def write_message(stream: TextIO, message: Message) -> None:
    encoded = encode_line(message)
    if stream.write(encoded) != len(encoded):
        raise OSError("worker message write was incomplete")
    stream.flush()
