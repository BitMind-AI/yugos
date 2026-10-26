import io
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from bundles import MODEL_SOURCE, make_bundle

from yugos.inference.chat import ModelInput
from yugos.submission import bundle, protocol
from yugos.submission.protocol import (
    Load,
    Loaded,
    Prepared,
    Route,
    Routed,
    Shutdown,
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
        Route(
            session_id=SESSION,
            sequence=4,
            request_id="opaque",
            input=ModelInput.from_text("Please solve the equation x + 2 = 4."),
        ),
        Route(
            session_id=SESSION,
            sequence=6,
            request_id="other",
            input=ModelInput.from_text("Write a Python function to add two numbers."),
        ),
        Shutdown(session_id=SESSION, sequence=8),
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
    assert [type(m) for m in responses] == [Prepared, Loaded, Routed, Routed, Stopped]
    assert [m.model.model_id for m in responses if isinstance(m, Routed)] == (
        ["code-model", "math-model"] if reverse_weights else ["math-model", "code-model"]
    )


@pytest.mark.parametrize(
    "result",
    [
        "'forged answer'",
        "[]",
        "iter(())",
        "{'provider_id': 'fixture', 'model_id': 'general', 'input': input}",
        "{'provider_id': 'fixture', 'model_id': 'general', 'effort': None, 'max_tokens': 999}",
    ],
)
def test_route_rejects_answers_actions_and_generation_overrides(tmp_path, result):
    source = f"""
class Model:
    def route(self, input):
        return {result}
def load_model(weights_path):
    return Model()
"""
    archive, messages = conversation(tmp_path, model_source=source)
    responses = run_messages(tmp_path, archive, messages)
    assert all(m.failure.code == "request.invalid_action" for m in responses[2:4])
    assert isinstance(responses[-1], Stopped)


def test_legacy_solve_entrypoint_does_not_load(tmp_path):
    archive, messages = conversation(
        tmp_path,
        model_source="""
class Model:
    def solve(self, input):
        yield {}
def load_model(weights_path):
    return Model()
""",
    )
    assert run_messages(tmp_path, archive, messages)[-1].failure.code == "load.failed"


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
        def route(self, input):
            raise RuntimeError("failed prediction")
    return Model()
"""
    archive, messages = conversation(tmp_path, model_source=source)
    responses = run_messages(tmp_path, archive, messages)
    assert responses[2].failure.code == "request.failed"
    assert responses[2].failure.request_id == "opaque"
    assert not any(isinstance(message, Routed) for message in responses)


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
    assert not any(isinstance(m, Routed) for m in responses)


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


@pytest.mark.parametrize("starter", [False, True], ids=["fixture", "starter"])
def test_worker_cli_loads_packaged_router(tmp_path, starter):
    archive, messages = conversation(tmp_path)
    if starter:
        # Exercise the published example unchanged: stale solve() code must fail.
        source = Path(__file__).resolve().parents[3] / "examples/starter-router"
        archive = tmp_path / "starter.zip"
        packed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "yugos.miner",
                "pack",
                str(source),
                "--output",
                str(archive),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert packed.returncode == 0, packed.stderr
        metadata = bundle.inspect_bundle(archive)
        messages[0] = messages[0].model_copy(
            update={
                "submission_sha256": metadata.submission_sha256,
                "expected_manifest_sha256": metadata.manifest_sha256,
            }
        )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "yugos.submission.worker",
            str(archive),
            str(tmp_path / "runtime"),
        ],
        cwd=tmp_path,
        input="".join(encode_line(m) for m in messages),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    responses = [decode_line(line) for line in completed.stdout.splitlines(keepends=True)]
    assert [type(m) for m in responses] == [Prepared, Loaded, Routed, Routed, Stopped]
    if starter:
        choices = [m.model for m in responses if isinstance(m, Routed)]
        assert all(choice in metadata.required_models for choice in choices)
        assert len(set(choices)) == 1
    else:
        assert responses[2].model.model_id == "math-model"


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
    message = Route(
        session_id=SESSION, sequence=4, request_id="opaque", input=ModelInput.from_text("é" * 100)
    )
    encoded = encode_line(message)
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", len(encoded.encode()) - 2)
    with pytest.raises(ValueError):
        encode_line(message)
    with pytest.raises(ValueError):
        decode_line(encoded)


@pytest.mark.parametrize("schema", ["yugos.bundle.v2", "yugos.bundle.v3"])
def test_old_bundle_schema_requires_repackaging(tmp_path, schema):
    archive, _, _ = make_bundle(tmp_path, manifest_changes={"schema": schema})
    with pytest.raises(ValueError, match="yugos.bundle.v4"):
        bundle.inspect_bundle(archive)


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
