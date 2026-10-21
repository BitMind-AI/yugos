"""Load and query submitted models inside an operator-provided sandbox.

Run: python -m yugos.submission.worker SUBMISSION_ZIP RUNTIME_ROOT
This entry point does not establish isolation or make provider calls.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import redirect_stdout, suppress
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TextIO

from yugos.inference.actions import CallAction

from .bundle import PreparationError, prepare_bundle
from .protocol import (
    CallRequested,
    Cancel,
    Failed,
    Failure,
    Finished,
    Load,
    Loaded,
    Prepared,
    Resume,
    Shutdown,
    Solve,
    Start,
    Stopped,
    read_message,
    write_message,
)


def _load_model(root: Path):
    """Load the submitted weights and check that the model exposes solve(input)."""
    spec = spec_from_file_location("yugos_submission", root / "model.py")
    if spec is None or spec.loader is None:
        raise ValueError("model.py cannot be imported")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.load_model(str(root / "model.safetensors"))
    if not callable(getattr(model, "solve", None)):
        raise ValueError("load_model must return a model with solve(input)")
    return model


def serve(
    reader: TextIO,
    writer: TextIO,
    source: Path,
    runtime_root: Path,
    *,
    prepared_input: bool = False,
) -> None:
    """Keep one model loaded while advancing question generators through controller messages."""
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
    requests = {}

    def discard(request_id):
        # Remove before closing: faulty generator cleanup must not occupy a slot.
        generator = requests.pop(request_id, None)
        if generator is not None:
            with redirect_stdout(sys.stderr), suppress(Exception):
                generator.close()

    sequence = 2
    while True:
        message = read_message(reader)
        if message.session_id != start.session_id or message.sequence != sequence:
            failed(sequence + 1, "handshake.protocol", "unexpected session or sequence")
            return
        response_fields = {"session_id": start.session_id, "sequence": sequence + 1}
        if isinstance(message, Shutdown):
            for request_id in list(requests):
                discard(request_id)
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
        elif isinstance(message, Cancel):
            discard(message.request_id)
            write_message(writer, Finished(**response_fields, request_id=message.request_id))
        elif isinstance(message, (Solve, Resume)):
            request_id = message.request_id
            if isinstance(message, Resume) and request_id not in requests:
                failed(sequence + 1, "handshake.protocol", "no matching pending call")
                return
            advancing = False
            try:
                with redirect_stdout(sys.stderr):
                    if isinstance(message, Solve):
                        if start.max_active_requests == 1:
                            # Preserve serial callers' replacement semantics.
                            for previous_id in list(requests):
                                discard(previous_id)
                        if request_id in requests or len(requests) >= start.max_active_requests:
                            raise ValueError("duplicate request or request capacity exceeded")
                        generator = model.solve(message.input)
                        if not isinstance(generator, Generator):
                            raise ValueError("solve(input) must return a generator")
                        requests[request_id] = generator
                        advancing = True
                        candidate = next(generator)
                    else:
                        advancing = True
                        candidate = requests[request_id].send(message.output)
            except StopIteration as finished:
                requests.pop(request_id, None)
                if finished.value is not None:
                    failed(
                        sequence + 1,
                        "request.invalid_action",
                        "solve must finish without returning an answer",
                        request_id,
                    )
                else:
                    write_message(writer, Finished(**response_fields, request_id=request_id))
                sequence += 2
                continue
            except Exception:
                if advancing:
                    discard(request_id)
                failed(sequence + 1, "request.failed", "submission solve failed", request_id)
                sequence += 2
                continue
            try:
                action = CallAction.model_validate(candidate)
            except (TypeError, ValueError):
                discard(request_id)
                failed(
                    sequence + 1,
                    "request.invalid_action",
                    "model yielded an invalid call",
                    request_id,
                )
                sequence += 2
                continue
            write_message(
                writer, CallRequested(**response_fields, request_id=request_id, action=action)
            )
        else:
            failed(sequence + 1, "handshake.protocol", "expected Solve, Resume, or Shutdown")
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
