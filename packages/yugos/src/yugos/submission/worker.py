"""Load and query submitted models inside an operator-provided sandbox.

Run: python -m yugos.submission.worker SUBMISSION_ZIP RUNTIME_ROOT
This entry point does not establish isolation or make provider calls.
"""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TextIO

from yugos.inference.actions import ModelChoice

from .bundle import PreparationError, prepare_bundle
from .protocol import (
    Failed,
    Failure,
    Load,
    Loaded,
    Prepared,
    Route,
    Routed,
    Shutdown,
    Start,
    Stopped,
    read_message,
    write_message,
)


def _load_model(root: Path):
    """Load the submitted weights and require route(input)."""
    spec = spec_from_file_location("yugos_submission", root / "model.py")
    if spec is None or spec.loader is None:
        raise ValueError("model.py cannot be imported")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.load_model(str(root / "model.safetensors"))
    if not callable(getattr(model, "route", None)):
        raise ValueError("load_model must return a model with route(input)")
    return model


def serve(
    reader: TextIO,
    writer: TextIO,
    source: Path,
    runtime_root: Path,
    *,
    prepared_input: bool = False,
) -> None:
    """Keep one model loaded and return one routing choice per request."""
    start = read_message(reader)

    def failed(sequence, code, detail, request_id=None):
        write_message(
            writer,
            Failed(
                session_id=start.session_id,
                sequence=sequence,
                failure=Failure(code=code, detail=detail, request_id=request_id),
            ),
        )

    if not isinstance(start, Start) or start.sequence != 0:
        failed(start.sequence + 1, "handshake.protocol", "expected Start sequence 0")
        return
    try:
        if prepared_input:
            from .prepared import read_prepared_directory

            prepared = read_prepared_directory(
                source,
                submission_sha256=start.submission_sha256,
                manifest_sha256=start.expected_manifest_sha256,
            )
        else:
            prepared = prepare_bundle(
                source,
                runtime_root / "model",
                submission_sha256=start.submission_sha256,
                manifest_sha256=start.expected_manifest_sha256,
            )
    except PreparationError as exc:
        write_message(writer, Failed(session_id=start.session_id, sequence=1, failure=exc.failure))
        return
    except Exception:
        failed(1, "preparation.internal", "submission could not be prepared")
        return
    write_message(
        writer, Prepared(session_id=start.session_id, sequence=1, metadata=prepared.metadata)
    )
    model = None
    sequence = 2
    while True:
        message = read_message(reader)
        if message.session_id != start.session_id or message.sequence != sequence:
            failed(sequence + 1, "handshake.protocol", "unexpected session or sequence")
            return
        response_fields = {"session_id": start.session_id, "sequence": sequence + 1}
        if isinstance(message, Shutdown):
            write_message(writer, Stopped(**response_fields))
            return
        if model is None:
            if not isinstance(message, Load):
                failed(sequence + 1, "handshake.protocol", "expected Load after Prepared")
                return
            try:
                # Keep ordinary model logging off the message stream. The whole
                # worker remains untrusted after Load; the caller validates replies.
                with redirect_stdout(sys.stderr):
                    model = _load_model(prepared.root)
            except Exception:
                failed(sequence + 1, "load.failed", "submission model could not be loaded")
                return
            write_message(writer, Loaded(**response_fields))
        elif isinstance(message, Route):
            try:
                with redirect_stdout(sys.stderr):
                    candidate = model.route(message.input)
                choice = ModelChoice.model_validate(candidate)
            except (TypeError, ValueError):
                failed(
                    sequence + 1,
                    "request.invalid_action",
                    "route must return one ModelChoice",
                    message.request_id,
                )
            except Exception:
                failed(
                    sequence + 1, "request.failed", "submission route failed", message.request_id
                )
            else:
                write_message(
                    writer, Routed(**response_fields, request_id=message.request_id, model=choice)
                )
        else:
            failed(sequence + 1, "handshake.protocol", "expected Route or Shutdown")
            return
        sequence += 2


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--prepared", action="store_true")
    parser.add_argument("--submission-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("source", type=Path)
    parser.add_argument("runtime_root", type=Path)
    args = parser.parse_args()
    if args.prepare_only:
        if not args.submission_sha256:
            parser.error("--prepare-only requires --submission-sha256")
        from .prepared import prepare_to_directory

        try:
            prepare_to_directory(
                args.source,
                args.runtime_root,
                submission_sha256=args.submission_sha256,
                manifest_sha256=args.manifest_sha256,
            )
        except PreparationError as exc:
            args.runtime_root.mkdir(parents=True, exist_ok=True)
            (args.runtime_root / "failure.json").write_text(exc.failure.model_dump_json())
            return 2
        except Exception as exc:
            print(
                json.dumps({"error_type": type(exc).__name__, "detail": str(exc)[:1000]}),
                file=sys.stderr,
            )
            return 1
        return 0
    try:
        serve(sys.stdin, sys.stdout, args.source, args.runtime_root, prepared_input=args.prepared)
    except (EOFError, ValueError, OSError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
