import io
import stat
import subprocess
import sys
import zipfile

import pytest
from bundles import MODEL, MODEL_SOURCE, make_bundle

from yugos.inference.actions import CallAction, CallOutput
from yugos.inference.chat import ModelInput
from yugos.submission import bundle, protocol
from yugos.submission.protocol import (
    CallRequested,
    Finished,
    Load,
    Loaded,
    Prepared,
    Resume,
    Shutdown,
    Solve,
    Start,
    Stopped,
    decode_line,
    encode_line,
)
from yugos.submission.worker import serve

SESSION = "a" * 32


def conversation(tmp_path, **changes):
    archive, digest, manifest_digest = make_bundle(tmp_path / "source", **changes)
    messages = [
        Start(
            session_id=SESSION,
            sequence=0,
            submission_sha256=digest,
            expected_manifest_sha256=manifest_digest,
        ),
        Load(session_id=SESSION, sequence=2),
        Solve(
            session_id=SESSION,
            sequence=4,
            request_id="opaque",
            input=ModelInput.from_text("Please solve the equation x + 2 = 4."),
        ),
        Resume(
            session_id=SESSION,
            sequence=6,
            request_id="opaque",
            output=CallOutput(text="2", finish_reason="stop"),
        ),
        Solve(
            session_id=SESSION,
            sequence=8,
            request_id="other",
            input=ModelInput.from_text("Write a Python function to add two numbers."),
        ),
        Resume(
            session_id=SESSION,
            sequence=10,
            request_id="other",
            output=CallOutput(text="def add(a, b): return a + b", finish_reason="stop"),
        ),
        Shutdown(session_id=SESSION, sequence=12),
    ]
    return archive, messages


def run_messages(tmp_path, archive, messages):
    output = io.StringIO()
    serve(
        io.StringIO("".join(encode_line(m) for m in messages)),
        output,
        archive,
        tmp_path / "runtime",
    )
    return [decode_line(line + "\n") for line in output.getvalue().splitlines()]


@pytest.mark.parametrize("reverse_weights", [False, True])
def test_model_selection_uses_uploaded_weights_and_actual_prompt(tmp_path, reverse_weights):
    archive, messages = conversation(tmp_path, reverse_weights=reverse_weights)
    responses = run_messages(tmp_path, archive, messages)
    assert [type(m) for m in responses] == [
        Prepared,
        Loaded,
        CallRequested,
        Finished,
        CallRequested,
        Finished,
        Stopped,
    ]
    expected = ["math-model", "code-model"]
    assert [m.action.model.model_id for m in responses if isinstance(m, CallRequested)] == (
        expected[::-1] if reverse_weights else expected
    )
    assert responses[0].metadata.manifest_sha256 == messages[0].expected_manifest_sha256
    assert responses[2].request_id == "opaque"
    assert [m.sequence for m in responses] == [m.sequence + 1 for m in messages]


def model_yielding(candidate):
    return f"def load_model(weights_path):\n    class Model:\n        def solve(self, input):\n            yield {candidate!r}\n    return Model()\n"


def test_worker_preserves_the_miners_exact_prompt(tmp_path):
    action = {
        "kind": "call",
        "model": MODEL,
        "input": ModelInput.from_text("My exact prompt\n<draft>miner-owned</draft>").model_dump(),
    }
    archive, messages = conversation(tmp_path, model_source=model_yielding(action))
    responses = run_messages(tmp_path, archive, messages)
    assert responses[2].action == CallAction.model_validate(action)
    assert isinstance(responses[3], Finished)


@pytest.mark.parametrize(
    "candidate",
    [
        {"provider_id": "fixture", "model_id": "bare"},
        {"kind": "call", "model": MODEL, "prompt": "x", "claimed_score": 1},
        {"kind": "plan", "steps": [{"model": MODEL}, {"model": MODEL}]},
    ],
)
def test_invalid_calls_fail_the_request(tmp_path, candidate):
    archive, messages = conversation(tmp_path, model_source=model_yielding(candidate))
    responses = run_messages(tmp_path, archive, messages)
    assert responses[2].failure.code == "request.invalid_action"
    assert responses[2].failure.request_id == "opaque"
    assert responses[2].failure.owner == "submission"


def test_generator_return_value_cannot_replace_the_provider_answer(tmp_path):
    source = f"""
def load_model(weights_path):
    class Model:
        def solve(self, input):
            yield {{"model": {MODEL!r}, "input": input}}
            return "a replacement answer"
    return Model()
"""
    archive, messages = conversation(tmp_path, model_source=source)
    responses = run_messages(tmp_path, archive, messages)
    assert isinstance(responses[2], CallRequested)
    assert responses[3].failure.code == "request.invalid_action"
    assert isinstance(responses[-1], Stopped)
    assert not any(isinstance(reply, Finished) for reply in responses)


@pytest.mark.parametrize("phase", ["before_call", "wrong_question", "after_finish"])
def test_resume_requires_a_pending_call_for_that_question(tmp_path, phase):
    archive, messages = conversation(tmp_path)
    if phase == "before_call":
        messages = messages[:2] + [messages[3].model_copy(update={"sequence": 4})]
    elif phase == "wrong_question":
        messages[3] = messages[3].model_copy(update={"request_id": "wrong-question"})
    else:
        messages = messages[:4] + [messages[3].model_copy(update={"sequence": 8})]
    responses = run_messages(tmp_path, archive, messages)
    assert responses[-1].failure.code == "handshake.protocol"


def test_new_question_closes_an_unfinished_generator_and_shutdown_closes_the_next(tmp_path):
    marker = tmp_path / "closed.txt"
    source = f"""
def load_model(weights_path):
    class Model:
        active = False
        def solve(self, input):
            if self.active:
                raise RuntimeError("previous question was not closed")
            self.active = True
            try:
                while True:
                    yield {{"model": {MODEL!r}, "input": input}}
            finally:
                self.active = False
                with open({str(marker)!r}, "a") as out:
                    out.write(input.messages[0].content + "\\n")
    return Model()
"""
    archive, messages = conversation(tmp_path, model_source=source)
    first, second = messages[2], messages[4].model_copy(update={"sequence": 6})
    messages = messages[:2] + [first, second, Shutdown(session_id=SESSION, sequence=8)]
    responses = run_messages(tmp_path, archive, messages)
    assert [
        r.action.input.messages[0].content for r in responses if isinstance(r, CallRequested)
    ] == [
        first.input.messages[0].content,
        second.input.messages[0].content,
    ]
    assert marker.read_text().splitlines() == [
        first.input.messages[0].content,
        second.input.messages[0].content,
    ]
    assert isinstance(responses[-1], Stopped)


def test_preparation_never_imports_model_code(tmp_path):
    archive, messages = conversation(tmp_path, model_source="raise RuntimeError('imported')")
    responses = run_messages(
        tmp_path, archive, [messages[0], Shutdown(session_id=SESSION, sequence=2)]
    )
    assert [type(m) for m in responses] == [Prepared, Stopped]
    responses = run_messages(tmp_path / "load", archive, messages)
    assert responses[-1].failure.code == "load.failed"


def test_model_exception_is_reported_for_the_requested_prompt(tmp_path):
    source = """
def load_model(weights_path):
    class Model:
        def solve(self, input):
            raise RuntimeError("failed prediction")
    return Model()
"""
    archive, messages = conversation(tmp_path, model_source=source)
    responses = run_messages(tmp_path, archive, messages)
    assert responses[2].failure.code == "request.failed"
    assert responses[2].failure.request_id == "opaque"
    assert not any(isinstance(message, CallRequested) for message in responses)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"weights_data": b"invalid tensor header"}, "preparation.weights_invalid"),
        ({"manifest_changes": {"runtime": "unknown-runtime"}}, "preparation.manifest_invalid"),
        (
            {"manifest_changes": {"schema": "yugos.bundle.v1", "runtime": "python-safetensors-v1"}},
            "preparation.manifest_invalid",
        ),
    ],
)
def test_invalid_model_artifacts_are_rejected_before_load(tmp_path, changes, expected):
    archive, messages = conversation(tmp_path, **changes)
    responses = run_messages(tmp_path, archive, messages)
    assert len(responses) == 1
    assert responses[0].failure.code == expected


@pytest.mark.parametrize("field,value", [("session_id", "b" * 32), ("sequence", 12)])
def test_worker_rejects_replay_or_cross_session_requests(tmp_path, field, value):
    archive, messages = conversation(tmp_path)
    messages[2] = messages[2].model_copy(update={field: value})
    responses = run_messages(tmp_path, archive, messages)
    assert responses[-1].failure.code == "handshake.protocol"
    assert not any(isinstance(m, CallRequested) for m in responses)


def test_worker_never_serves_before_load(tmp_path):
    archive, messages = conversation(tmp_path)
    messages = [messages[0], messages[2].model_copy(update={"sequence": 2})]
    assert run_messages(tmp_path, archive, messages)[-1].failure.code == "handshake.protocol"


@pytest.mark.parametrize("which", ["submission_sha256", "expected_manifest_sha256"])
def test_commitment_mismatch_is_terminal_before_load(tmp_path, which):
    archive, messages = conversation(tmp_path)
    messages[0] = messages[0].model_copy(update={which: "0" * 64})
    responses = run_messages(tmp_path, archive, messages)
    assert len(responses) == 1
    assert responses[0].failure.code == "preparation.digest_mismatch"


def test_worker_cli_loads_the_trusted_model_fixture(tmp_path):
    archive, messages = conversation(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "yugos.submission.worker",
            str(archive),
            str(tmp_path / "runtime"),
        ],
        input="".join(encode_line(m) for m in messages),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    responses = [decode_line(line) for line in completed.stdout.splitlines(keepends=True)]
    assert responses[2].action.model.model_id == "math-model"
    assert isinstance(responses[-1], Stopped)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../escape", "a\\escape", "./alias", "a//alias"]
)
def test_unsafe_archive_is_rejected_before_any_member_is_written(tmp_path, name):
    archive, digest, _ = make_bundle(tmp_path / "source", extra={name: b"bad"})
    destination = tmp_path / "expanded"
    with pytest.raises(bundle.PreparationError) as caught:
        bundle.prepare_bundle(archive, destination, submission_sha256=digest)
    assert caught.value.failure.code == "preparation.archive_invalid"
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["duplicate", "symlink", "conflict"])
def test_ambiguous_or_special_archive_members_are_rejected(tmp_path, kind):
    archive, _, _ = make_bundle(tmp_path / "source")
    with zipfile.ZipFile(archive, "a") as out:
        if kind == "duplicate":
            with pytest.warns(UserWarning):
                out.writestr("model.py", "different")
        elif kind == "symlink":
            link = zipfile.ZipInfo("link")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            out.writestr(link, "model.py")
        else:
            out.writestr("model.py/child", "content")
    with pytest.raises(bundle.PreparationError) as caught:
        bundle.prepare_bundle(
            archive, tmp_path / "expanded", submission_sha256=bundle.sha256_file(archive)
        )
    assert caught.value.failure.code == "preparation.archive_invalid"
    assert not (tmp_path / "expanded").exists()


def test_expansion_limit_is_enforced_before_writing(tmp_path, monkeypatch):
    original, _, _ = make_bundle(tmp_path / "source", extra={"large": b"x" * 1_000_000})
    archive = tmp_path / "compressed.zip"
    with (
        zipfile.ZipFile(original) as source,
        zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for name in source.namelist():
            target.writestr(name, source.read(name))
    limit = 8192
    assert archive.stat().st_size < limit
    monkeypatch.setattr(bundle, "MAX_EXPANDED_BYTES", limit)
    with pytest.raises(bundle.PreparationError, match="limit"):
        bundle.prepare_bundle(
            archive, tmp_path / "expanded", submission_sha256=bundle.sha256_file(archive)
        )
    assert not (tmp_path / "expanded").exists()


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"unlisted.py": b"pass"}, "preparation.manifest_invalid"),
        ({"model.py": b"changed"}, "preparation.digest_mismatch"),
    ],
)
def test_all_payloads_must_match_the_manifest(tmp_path, extra, expected):
    archive, digest, _ = make_bundle(tmp_path / "source", extra=extra)
    with pytest.raises(bundle.PreparationError) as caught:
        bundle.prepare_bundle(archive, tmp_path / "expanded", submission_sha256=digest)
    assert caught.value.failure.code == expected


@pytest.mark.parametrize("payload", ["{}", "{}\n", '{"kind":"future"}\n'])
def test_invalid_or_incomplete_messages_are_rejected(payload):
    with pytest.raises(ValueError):
        decode_line(payload)


def test_message_size_is_bounded_in_both_directions(monkeypatch):
    message = Solve(
        session_id=SESSION, sequence=4, request_id="opaque", input=ModelInput.from_text("é" * 100)
    )
    encoded = encode_line(message)
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", len(encoded.encode()) - 2)
    with pytest.raises(ValueError):
        encode_line(message)
    with pytest.raises(ValueError):
        decode_line(encoded)


def test_old_bundle_schema_requires_repackaging(tmp_path):
    archive, _, _ = make_bundle(tmp_path, manifest_changes={"schema": "yugos.bundle.v2"})
    with pytest.raises(ValueError, match="yugos.bundle.v3"):
        bundle.inspect_bundle(archive)


@pytest.mark.parametrize("failure", ["exception", "invalid", "answer", "not_generator"])
def test_worker_serves_next_question_after_miner_error(tmp_path, failure):
    source = f"""
def load_model(weights_path):
    class Model:
        def solve(self, input):
            if "equation" in input.messages[0].content:
                if {failure!r} == "exception":
                    raise RuntimeError("bad prompt")
                if {failure!r} == "answer":
                    return "forged answer"
                yield {{"invalid": "action"}}
            else:
                yield {{"model": {MODEL!r}, "input": input}}
    return Model()
"""
    if failure == "not_generator":
        source = f"""
def load_model(weights_path):
    class Model:
        def solve(self, input):
            return [] if "equation" in input.messages[0].content else self.route(input)
        def route(self, input):
            yield {{"model": {MODEL!r}, "input": input}}
    return Model()
"""
    archive, messages = conversation(tmp_path, model_source=source)
    # A failed solve gets no Resume; the controller starts the next question.
    messages = messages[:3] + [
        m.model_copy(update={"sequence": m.sequence - 2}) for m in messages[4:]
    ]
    responses = run_messages(tmp_path, archive, messages)
    assert responses[2].failure.request_id == "opaque"
    assert isinstance(responses[3], CallRequested) and responses[3].request_id == "other"
    assert isinstance(responses[4], Finished)
    assert isinstance(responses[5], Stopped)
    assert [m.sequence for m in responses] == [m.sequence + 1 for m in messages]


def test_interleaved_requests_resume_their_own_generator_and_cancel_only_one(tmp_path):
    # A singleton generator would replace the first question with the second.
    from yugos.submission.protocol import Cancel

    source = f"""
def load_model(weights_path):
    class Model:
        def solve(self, input):
            response = yield {{"model": {MODEL!r}, "input": input}}
            from yugos.inference.chat import ModelInput
            yield {{"model": {MODEL!r}, "input": ModelInput.from_text(input.messages[0].content + response.text)}}
    return Model()
"""
    archive, original = conversation(tmp_path, model_source=source)
    start = original[0].model_copy(update={"max_active_requests": 4})
    messages = [
        start,
        original[1],
        original[2],
        original[4],
        original[5],
        Cancel(session_id=SESSION, sequence=0, request_id="other"),
        original[3],
        Shutdown(session_id=SESSION, sequence=0),
    ]
    messages = [m.model_copy(update={"sequence": i * 2}) for i, m in enumerate(messages)]
    responses = run_messages(tmp_path, archive, messages)
    assert responses[4].request_id == "other"
    assert responses[4].action.input.messages[0].content.endswith("def add(a, b): return a + b")
    assert responses[6].request_id == "opaque"
    assert responses[6].action.input.messages[0].content.endswith("x + 2 = 4.2")
    assert isinstance(responses[-1], Stopped)


def test_concurrent_worker_rejects_capacity_overflow_without_replacing_requests(tmp_path):
    archive, original = conversation(tmp_path)
    messages = [
        original[0].model_copy(update={"max_active_requests": 2}),
        original[1],
        original[2],
        original[4],
        original[2].model_copy(update={"request_id": "third"}),
        original[3],
        original[-1],
    ]
    messages = [m.model_copy(update={"sequence": i * 2}) for i, m in enumerate(messages)]
    replies = run_messages(tmp_path, archive, messages)
    assert replies[4].failure.code == "request.failed"
    assert isinstance(replies[5], Finished) and replies[5].request_id == "opaque"


@pytest.mark.parametrize("phase", ["solve", "resume"])
@pytest.mark.parametrize("failure", ["exception", "invalid", "invalid_cleanup"])
def test_failed_concurrent_request_releases_its_slot_and_id(tmp_path, phase, failure):
    source = f"""
def load_model(weights_path):
    class Model:
        def solve(self, input):
            if "equation" in input.messages[0].content:
                if {phase!r} == "resume":
                    yield {{"model": {MODEL!r}, "input": input}}
                if {failure!r} == "exception":
                    raise RuntimeError("failed request")
                try:
                    yield {{"invalid": "action"}}
                finally:
                    if {failure!r} == "invalid_cleanup":
                        raise RuntimeError("failed cleanup")
            else:
                yield {{"model": {MODEL!r}, "input": input}}
    return Model()
"""
    archive, original = conversation(tmp_path, model_source=source)
    messages = [
        original[0].model_copy(update={"max_active_requests": 2}),
        original[1],
        original[4],  # Keep another generator alive throughout the failure.
        original[2],
    ]
    if phase == "resume":
        messages.append(original[3])
    messages += [
        original[4].model_copy(update={"request_id": "opaque"}),
        original[5],
        original[3],
        original[-1],
    ]
    messages = [m.model_copy(update={"sequence": i * 2}) for i, m in enumerate(messages)]
    replies = run_messages(tmp_path, archive, messages)
    failed = replies[-5]
    assert failed.failure.request_id == "opaque"
    assert failed.failure.code == (
        "request.failed" if failure == "exception" else "request.invalid_action"
    )
    assert isinstance(replies[-4], CallRequested) and replies[-4].request_id == "opaque"
    assert isinstance(replies[-3], Finished) and replies[-3].request_id == "other"
    assert isinstance(replies[-2], Finished) and replies[-2].request_id == "opaque"
    assert isinstance(replies[-1], Stopped)


def test_duplicate_solve_does_not_discard_the_original_request(tmp_path):
    archive, original = conversation(tmp_path)
    messages = [
        original[0].model_copy(update={"max_active_requests": 2}),
        original[1],
        original[2],
        original[2],
        original[3],
        original[-1],
    ]
    messages = [m.model_copy(update={"sequence": i * 2}) for i, m in enumerate(messages)]
    replies = run_messages(tmp_path, archive, messages)
    assert replies[3].failure.code == "request.failed"
    assert isinstance(replies[4], Finished) and replies[4].request_id == "opaque"
    assert isinstance(replies[-1], Stopped)


@pytest.mark.parametrize("operation", ["cancel", "shutdown", "replace"])
def test_generator_cleanup_failure_does_not_abort_other_requests(tmp_path, operation, capsys):
    from yugos.submission.protocol import Cancel

    source = f"""
def load_model(weights_path):
    class Model:
        def solve(self, input):
            try:
                yield {{"model": {MODEL!r}, "input": input}}
            finally:
                if "equation" in input.messages[0].content:
                    raise RuntimeError("cleanup failed")
                print("healthy generator closed")
    return Model()
"""
    archive, original = conversation(tmp_path, model_source=source)
    messages = [
        original[0].model_copy(update={"max_active_requests": 1 if operation == "replace" else 2}),
        original[1],
        original[2],
        original[4],
    ]
    if operation == "cancel":
        messages += [Cancel(session_id=SESSION, sequence=0, request_id="opaque"), original[5]]
    elif operation == "replace":
        messages.append(original[5])
    messages.append(original[-1])
    messages = [m.model_copy(update={"sequence": i * 2}) for i, m in enumerate(messages)]
    replies = run_messages(tmp_path, archive, messages)
    assert len(replies) == len(messages)
    assert isinstance(replies[3], CallRequested) and replies[3].request_id == "other"
    if operation != "shutdown":
        assert isinstance(replies[-2], Finished) and replies[-2].request_id == "other"
    assert isinstance(replies[-1], Stopped)
    assert "healthy generator closed" in capsys.readouterr().err


def test_cpu_preparation_defers_import_and_worker_reuses_verified_files(tmp_path, monkeypatch):
    import yugos.submission.worker as worker
    from yugos.submission.prepared import prepare_to_directory

    # Importing the miner during preparation would execute this side effect.
    marker = tmp_path / "imported"
    archive, messages = conversation(
        tmp_path,
        model_source=f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + MODEL_SOURCE,
    )
    destination = tmp_path / "prepared"
    monkeypatch.setattr(
        bundle, "sha256_file", lambda *a: pytest.fail("rereading extracted payloads")
    )
    prepare_to_directory(archive, destination, submission_sha256=messages[0].submission_sha256)
    assert not marker.exists()
    archive.unlink()
    monkeypatch.setattr(
        worker, "prepare_bundle", lambda *a, **k: pytest.fail("repeated extraction")
    )
    output = io.StringIO()
    serve(
        io.StringIO("".join(encode_line(m) for m in messages)),
        output,
        destination,
        tmp_path / "scratch",
        prepared_input=True,
    )
    responses = [decode_line(line + "\n") for line in output.getvalue().splitlines()]
    assert isinstance(responses[0], Prepared)
    assert isinstance(responses[1], Loaded)
    assert isinstance(responses[-1], Stopped)
    assert marker.exists()


@pytest.mark.parametrize("changed", ["submission", "manifest", "metadata"])
def test_prepared_handoff_rejects_wrong_identity(tmp_path, changed):
    from yugos.submission.bundle import PreparationError
    from yugos.submission.prepared import prepare_to_directory, read_prepared_directory

    archive, messages = conversation(tmp_path)
    destination = tmp_path / "prepared"
    start = messages[0]
    prepare_to_directory(archive, destination, submission_sha256=start.submission_sha256)
    digest = "f" * 64 if changed == "submission" else start.submission_sha256
    manifest = "f" * 64 if changed == "manifest" else start.expected_manifest_sha256
    if changed == "metadata":
        path = destination / "metadata.json"
        path.write_text(path.read_text().replace('"name":', '"name":"forged","unused":'))
    with pytest.raises(PreparationError):
        read_prepared_directory(destination, submission_sha256=digest, manifest_sha256=manifest)


def test_prepare_only_entrypoint_and_prepared_worker_round_trip(tmp_path):
    archive, messages = conversation(tmp_path)
    destination = tmp_path / "prepared"
    prepare = subprocess.run(
        [
            sys.executable,
            "-m",
            "yugos.submission.worker",
            "--prepare-only",
            "--submission-sha256",
            messages[0].submission_sha256,
            "--manifest-sha256",
            messages[0].expected_manifest_sha256,
            str(archive),
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert prepare.returncode == 0, prepare.stderr
    archive.unlink()
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "yugos.submission.worker",
            "--prepared",
            str(destination),
            str(tmp_path / "scratch"),
        ],
        input="".join(encode_line(m) for m in messages),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    replies = [decode_line(line + "\n") for line in process.stdout.splitlines()]
    assert isinstance(replies[0], Prepared) and isinstance(replies[-1], Stopped)


def test_declaration_read_does_not_scan_payloads_or_attest_them(tmp_path, monkeypatch):
    archive, _, _ = make_bundle(tmp_path, extra={"model.py": b"tampered"})
    opened = []
    original = zipfile.ZipFile.open

    def open_member(self, name, *args, **kwargs):
        opened.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", open_member)
    assert bundle.read_bundle_manifest(archive).required_models
    assert opened == ["manifest.json"]
    with pytest.raises(ValueError, match="digest mismatch"):
        bundle.inspect_bundle(archive)
