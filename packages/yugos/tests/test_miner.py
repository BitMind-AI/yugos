"""Submission retries must preserve the original upload and paid evaluation job."""

import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bittensor.wallet import Keypair
from btauth import assert_unsigned, verify_signed
from bundles import make_bundle

from yugos.inference.actions import ModelChoice
from yugos.miner import job_progress, package, submit, upload
from yugos.service_client import ServiceClient
from yugos.state import save_json
from yugos.submission.bundle import inspect_bundle, prepare_bundle


def cached_submission(path):
    receipt = json.loads(path.read_text())
    funding = receipt.get("funding")
    if funding:
        funding = {
            **funding,
            "model_id": receipt["model_id"],
            "hotkey": receipt["hotkey"],
            "quote": {k: receipt[k] for k in ("network", "netuid")},
        }
    return httpx.Response(
        200,
        json={
            "data": {
                "model_id": receipt["model_id"],
                "r2_key": receipt["r2_key"],
                "uploaded": receipt.get("uploaded", False),
                "job_id": receipt.get("job_id"),
                **({"funding": funding} if funding else {}),
            }
        },
    )


class UploadService:
    def __init__(self, key, archive, *, fail=None):
        self.key = key
        self.archive = archive
        self.digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.r2_key = f"bundles/{key.ss58_address}/{self.digest}.zip"
        self.requests = []
        self.fail = fail
        self.jobs = set()
        self.uploaded = False

    def __call__(self, request):
        path = request.url.path.rsplit("/", 1)[-1]
        self.requests.append(path)
        if request.method == "PUT":
            body = request.read()
            assert_unsigned(request.headers)
            assert body == self.archive.read_bytes()
            self.uploaded = True
            return httpx.Response(200, headers={"etag": "fixture-part"})
        body = verify_signed(request, self.key)
        payload = json.loads(body)
        if path == "credits":
            return httpx.Response(200, json={"balance_cents": 0, "active_run_id": None})
        if path == "submissions":
            assert payload["expected_hash"] == self.digest
            return httpx.Response(
                200,
                json={
                    "data": {
                        "model_id": 3,
                        "r2_key": self.r2_key,
                        "uploaded": self.uploaded,
                        "job_id": next(iter(self.jobs), None),
                        "multipart": {
                            "upload_id": "fixture-session",
                            "part_bytes": self.archive.stat().st_size,
                            "parts": [],
                        },
                    }
                },
            )
        if path == "part":
            return httpx.Response(
                200,
                json={
                    "url": "https://objects.example/payload?signature=opaque",
                    "part_number": 1,
                    "size": self.archive.stat().st_size,
                },
            )
        if path == "complete":
            assert self.uploaded
            return httpx.Response(
                200, json={"uploaded": True, "model_id": 3, "r2_key": self.r2_key}
            )
        assert path == "confirm"
        assert payload["file_hash"] == self.digest and payload["model_id"] == 3
        self.jobs.add("job-1")
        if self.fail == "confirm":
            self.fail = None
            raise httpx.ReadTimeout("response lost after job creation")
        return httpx.Response(
            200, json={"ok": True, "model_id": 3, "r2_key": self.r2_key, "job_id": "job-1"}
        )


class MinerChain:
    def __init__(self, key, fail=False):
        self.wallet = SimpleNamespace(hotkey=key, coldkeypub=SimpleNamespace(ss58_address="cold"))
        self.netuid = 7
        self.commits = []
        self.fail = fail

    def metagraph(self):
        return object()

    def registered(self, graph):
        pass

    def commit(self, digest):
        self.commits.append(digest)
        if self.fail:
            self.fail = False
            raise RuntimeError("chain unavailable")
        return SimpleNamespace(block_hash="block-1")


@pytest.mark.parametrize("interruption", [None, "register", "upload"])
def test_funding_is_ready_before_upload_and_reused_for_final_authorization(
    tmp_path, monkeypatch, interruption
):
    archive, _, _ = make_bundle(tmp_path)
    metadata = inspect_bundle(archive)
    key = Keypair.create_from_uri("//Alice")
    chain = MinerChain(key)
    receipt_path = tmp_path / "receipt.json"
    r2_key = f"bundles/{key.ss58_address}/{metadata.submission_sha256}.zip"
    events, registrations = [], []
    failed = False
    uploaded = False
    monkeypatch.setattr("yugos.miner.funding.open_link", lambda _: events.append("browser"))

    def handle(request):
        nonlocal failed, uploaded
        route = request.url.path
        if request.method == "PUT":
            assert "browser" in events
            assert (
                json.loads(receipt_path.read_text())["funding"]["request_id"]
                == registrations[0]["request_id"]
            )
            assert request.content == archive.read_bytes()
            events.append("upload")
            if interruption == "upload" and not failed:
                failed = True
                raise httpx.ReadTimeout("upload interrupted")
            uploaded = True
            return httpx.Response(200, headers={"etag": "fixture-part"})
        payload = json.loads(request.content)
        if route.endswith("/submissions"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "model_id": 3,
                        "r2_key": r2_key,
                        "billing_enabled": True,
                        "multipart": {
                            "upload_id": "fixture-session",
                            "part_bytes": archive.stat().st_size,
                            "parts": [],
                        },
                    }
                },
            )
        if route.endswith("/multipart/part"):
            return httpx.Response(
                200,
                json={
                    "url": "https://objects.example/bundle",
                    "part_number": 1,
                    "size": archive.stat().st_size,
                },
            )
        if route.endswith("/multipart/complete"):
            assert uploaded
            return httpx.Response(200, json={"uploaded": True, "model_id": 3, "r2_key": r2_key})
        if route.endswith("/billing/register"):
            assert not uploaded
            registrations.append(payload)
            assert payload["file_hash"] == metadata.submission_sha256
            assert (
                payload["request_id"]
                == json.loads(receipt_path.read_text())["funding"]["request_id"]
            )
            if interruption == "register" and not failed:
                failed = True
                raise httpx.ReadTimeout("registration response lost")
            events.append("funding")
            return httpx.Response(
                200,
                json={
                    "state": "waiting",
                    "submission_ready": False,
                    "quote": {"reservation_cents": 10000},
                    "balance": {"balance_cents": 0},
                    "additional_required_cents": 10000,
                    "checkout_url": "https://checkout.stripe.com/c/pay/test",
                },
            )
        assert route.endswith("/billing/prepare") and uploaded
        assert payload["request_id"] == registrations[0]["request_id"]
        assert payload["portal_token"] == registrations[0]["portal_token"]
        events.append("authorize")
        return httpx.Response(200, json={"state": "started", "job_id": "funded-job"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        if interruption == "register":
            with pytest.raises(httpx.ReadTimeout):
                submit(archive, receipt_path, service, chain, network="test")
        result = submit(archive, receipt_path, service, chain, network="test")
    assert result["job_id"] == "funded-job" and result["committed"]
    assert events.index("browser") < events.index("upload") < events.index("authorize")
    assert events[-1] == "authorize"
    assert len({r["request_id"] for r in registrations}) == 1
    assert len(chain.commits) == 1


@pytest.fixture
def funding_submission(tmp_path):
    archive, _, _ = make_bundle(tmp_path)
    metadata = inspect_bundle(archive)
    key = Keypair.create_from_uri("//Alice")
    chain = MinerChain(key)
    receipt = {
        "schema": "yugos.submission.v1",
        "network": "test",
        "netuid": chain.netuid,
        "hotkey": key.ss58_address,
        "endpoint": "https://evaluation.example",
        "submission_sha256": metadata.submission_sha256,
        "manifest_sha256": metadata.manifest_sha256,
        "model_id": 3,
        "r2_key": f"bundles/{key.ss58_address}/{metadata.submission_sha256}.zip",
        "uploaded": True,
        "funding": {"request_id": "old-request", "portal_token": "old-private-token"},
    }
    path = tmp_path / "receipt.json"
    save_json(path, receipt)
    return archive, path, chain, receipt


@pytest.mark.parametrize("state", ["canceled", "expired", "attention"])
def test_renew_funding_replaces_terminal_intent_without_reupload(funding_submission, state):
    archive, path, chain, previous = funding_submission
    previous["status_url"] = previous["endpoint"] + "/billing/portal#old-request:old-private-token"
    save_json(path, previous)
    requests = []

    def respond(request):
        if request.url.path.endswith("/submissions"):
            return cached_submission(path)
        route = request.url.path
        requests.append(route)
        if route.endswith("/cancel"):
            return httpx.Response(200, json={"state": state})
        if route.endswith("/confirm"):
            return httpx.Response(409, json={"detail": {"code": "use_billing_prepare"}})
        assert route.endswith("/billing/prepare")
        payload = json.loads(request.read())
        saved = json.loads(path.read_text())
        assert payload["request_id"] == saved["funding"]["request_id"]
        assert payload["request_id"] != previous["funding"]["request_id"]
        assert payload["portal_token"] != previous["funding"]["portal_token"]
        assert saved["status_url"] == (
            f"{previous['endpoint']}/billing/portal#"
            f"{payload['request_id']}:{payload['portal_token']}"
        )
        assert saved["status_url"] != previous["status_url"]
        assert "status_url" not in payload
        assert (payload["model_id"], payload["network"], payload["netuid"]) == (3, "test", 7)
        return httpx.Response(200, json={"state": "started", "job_id": "renewed-job"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        service = ServiceClient(previous["endpoint"], client, chain.wallet.hotkey)
        result = submit(archive, path, service, chain, network="test", renew_funding=True)
        assert result["funding_history"] == [previous["funding"]]
        assert result["job_id"] == "renewed-job"
        assert result["confirmed"] and result["committed"]
        assert json.loads(path.read_text()) == result
        assert submit(archive, path, service, chain, network="test", renew_funding=True) == result
    assert requests == [
        "/billing/intents/old-request/cancel",
        "/confirm",
        "/billing/prepare",
    ]
    assert len(chain.commits) == 1


@pytest.mark.parametrize("job_id", [123, "job/123"])
@pytest.mark.parametrize("saved_alias", [False, True])
def test_renew_funding_preserves_started_job_as_status_compatible_string(
    funding_submission, job_id, saved_alias
):
    from urllib.parse import quote

    archive, path, chain, previous = funding_submission
    if saved_alias:
        previous["endpoint"] += "/upload"
        save_json(path, previous)
    requests = []

    def respond(request):
        if request.url.path.endswith("/submissions"):
            return cached_submission(path)
        requests.append(request.url.path)
        assert request.url.path.endswith("/old-request/cancel")
        return httpx.Response(200, json={"state": "started", "job_id": job_id})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        service = ServiceClient(previous["endpoint"], client, chain.wallet.hotkey)
        result = submit(archive, path, service, chain, network="test", renew_funding=True)
        saved = json.loads(path.read_text())
        assert saved["endpoint"] == service.endpoint
        assert saved["job_id"] == str(job_id)
        assert quote(saved["job_id"], safe="") == str(job_id).replace("/", "%2F")
        assert saved["funding"] == previous["funding"]
        assert "funding_history" not in saved
        assert result["confirmed"] and result["committed"]
        assert submit(archive, path, service, chain, network="test", renew_funding=True) == result
    assert len(requests) == len(chain.commits) == 1


@pytest.mark.parametrize("state", ["prepared", "waiting", "paid", "started", None])
def test_renew_funding_preserves_receipt_when_cancel_is_not_terminal(funding_submission, state):
    archive, path, chain, previous = funding_submission

    def respond(request):
        if request.url.path.endswith("/submissions"):
            return cached_submission(path)
        assert request.url.path.endswith("/old-request/cancel")
        return httpx.Response(200, json={"state": state})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        service = ServiceClient(previous["endpoint"], client, chain.wallet.hotkey)
        with pytest.raises(ValueError, match="could not be canceled"):
            submit(archive, path, service, chain, network="test", renew_funding=True)
    assert {k: v for k, v in json.loads(path.read_text()).items() if k != "status_url"} == previous
    assert chain.commits == []


def test_hosted_upload_recovers_confirmation_without_chain_or_reupload(tmp_path):
    archive, _, _ = make_bundle(tmp_path)
    key = Keypair.create_from_uri("//Alice")
    server = UploadService(key, archive, fail="confirm")
    path = tmp_path / "hosted.json"
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        with pytest.raises(httpx.ReadTimeout):
            upload(archive, path, service, coldkey="cold")
        receipt = upload(archive, path, service, coldkey="cold")
        requests = list(server.requests)
        assert upload(archive, path, service, coldkey="cold") == receipt
        assert server.requests == requests + ["submissions"]
        with pytest.raises(ValueError, match="another submission"):
            upload(archive, path, service, coldkey="different")
    assert server.requests.count("payload") == 1
    assert server.jobs == {receipt["job_id"]}
    assert receipt["confirmed"] and "committed" not in receipt


@pytest.mark.parametrize("failure", ["confirm", "commit"])
def test_resume_does_not_reupload_or_create_a_second_job(tmp_path, failure):
    archive, digest, _ = make_bundle(tmp_path)
    key = Keypair.create_from_uri("//Alice")
    server = UploadService(key, archive, fail=failure)
    chain = MinerChain(key, fail=failure == "commit")
    receipt_path = tmp_path / "receipt.json"
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        with pytest.raises((RuntimeError, httpx.ReadTimeout)):
            submit(archive, receipt_path, service, chain, network="test")
        receipt = submit(archive, receipt_path, service, chain, network="test")
        requests, commits = list(server.requests), list(chain.commits)
        assert submit(archive, receipt_path, service, chain, network="test") == receipt
        assert server.requests == requests + ["submissions"] and chain.commits == commits
    assert server.requests.count("payload") == 1
    assert server.requests.count("submissions") == 3
    assert server.requests.count("confirm") == 1
    assert server.jobs == {receipt["job_id"]}
    assert receipt["committed"] and receipt["submission_sha256"] == digest
    assert "presigned_url" not in receipt_path.read_text()


def test_submit_prints_each_stage_including_chain_finalization(tmp_path, capsys):
    archive, _, _ = make_bundle(tmp_path)
    key = Keypair.create_from_uri("//Alice")
    chain = MinerChain(key)
    with httpx.Client(transport=httpx.MockTransport(UploadService(key, archive))) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        submit(archive, tmp_path / "receipt.json", service, chain, network="test")
    output = capsys.readouterr().out
    assert "Checking hotkey registration." in output
    assert "Finalizing upload" in output
    assert "Upload stored." in output
    assert "Confirming the submission." in output
    assert "Evaluation job job-1 is confirmed." in output
    assert "Finalization can take a few minutes." in output
    assert "Committed in block block-1." in output


def test_status_summary_names_the_job_phase():
    summary = job_progress(
        {"id": "job-1", "phase": "scoring", "hint": "validator is scoring", "spend": [{}, {}]}
    )
    assert summary == "Job job-1: scoring. validator is scoring. 2 provider calls recorded."


def test_receipt_cannot_be_reused_for_another_identity(tmp_path):
    archive, _, _ = make_bundle(tmp_path)
    key = Keypair.create_from_uri("//Alice")
    chain = MinerChain(key)
    with httpx.Client(transport=httpx.MockTransport(UploadService(key, archive))) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        path = tmp_path / "receipt.json"
        submit(archive, path, service, chain, network="test")
        with pytest.raises(ValueError, match="another submission"):
            submit(archive, path, service, chain, network="finney")


def test_package_discovers_nested_assets_and_ignores_generated_and_local_files(tmp_path):
    archive, _, _ = make_bundle(
        tmp_path / "fixture", model_source="raise RuntimeError('do not import')"
    )
    source = tmp_path / "source"
    source.mkdir()
    with zipfile.ZipFile(archive) as contents:
        for name in ("model.py", "model.safetensors"):
            (source / name).write_bytes(contents.read(name))
    payloads = {
        "router.json": b'{"tokenizer": "tokenizer/vocab.json"}',
        "tokenizer/vocab.json": b'{"hello": 1}',
        "helpers/prompt.py": b"raise RuntimeError('do not import')",
        "tokenizer/merges.txt": b"hello world\n" * 2048,
        "models.json": b'[{"provider_id":"p","model_id":"m"}]',
    }
    for name, content in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (source / ".env").write_text("not a submission asset")
    (source / ".git").mkdir()
    (source / ".git/config").write_text("private local configuration")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/model.cpython-312.pyc").write_bytes(b"cache")
    (source / "manifest.json").write_text("outdated generated manifest")
    (source / "previous.zip").write_bytes(b"earlier archive")
    for filename in (
        "submission.receipt.json",
        "submission-receipt.json",
        "submission.receipt.json.lock",
        "submission-receipt.json.lock",
    ):
        (source / filename).write_text("private status and refund token")
    output = source / "packed.zip"
    kwargs = dict(
        name="router",
        required_models=(ModelChoice(provider_id="p", model_id="m"),),
    )
    first = package(source, output, **kwargs)
    os.utime(source / "model.py", (1234567890, 1234567890))
    second = package(source, source / "second.zip", **kwargs)
    assert first == second == inspect_bundle(output)
    with zipfile.ZipFile(output) as contents:
        assert set(contents.namelist()) == {
            "manifest.json",
            "model.py",
            "model.safetensors",
            *payloads,
        }
        manifest = json.loads(contents.read("manifest.json"))
        compressed = contents.getinfo("tokenizer/merges.txt")
        assert compressed.compress_type == zipfile.ZIP_DEFLATED
        assert compressed.compress_size < compressed.file_size
        for name, content in payloads.items():
            assert contents.read(name) == content
            assert manifest["files"][name] == hashlib.sha256(content).hexdigest()
    prepared = prepare_bundle(
        output, tmp_path / "expanded", submission_sha256=first.submission_sha256
    )
    for name, content in payloads.items():
        assert (prepared.root / name).read_bytes() == content
    (source / "tokenizer/vocab.json").write_text('{"changed": 2}')
    changed = package(source, source / "changed.zip", **kwargs)
    assert changed.submission_sha256 != first.submission_sha256


@pytest.mark.parametrize("directory", [False, True])
def test_discovered_symlinks_cannot_include_external_files(tmp_path, directory):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("# model")
    (source / "model.safetensors").write_bytes(b"weights")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("not a submission asset")
    (source / "linked").symlink_to(outside if directory else outside / "secret")
    output = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="symlinks"):
        package(
            source, output, name="router", required_models=[{"provider_id": "p", "model_id": "m"}]
        )
    assert not output.exists()


def test_tampered_payload_is_rejected_before_network_or_chain(tmp_path):
    archive, _, _ = make_bundle(tmp_path / "good")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"tampered" if name == "model.py" else source.read(name))
    with pytest.raises(ValueError, match="digest mismatch"):
        submit(bad, tmp_path / "receipt.json", None, None, network="test")


@pytest.mark.parametrize("models", [[], [{"provider_id": "p", "model_id": "m"}] * 2])
def test_packager_rejects_invalid_dependencies_without_replacing_output(tmp_path, models):
    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous submission")
    with pytest.raises(ValueError):
        package(tmp_path, output, name="router", required_models=models)
    assert output.read_bytes() == b"previous submission"


@pytest.mark.parametrize("entrypoint", ["public", "module"])
def test_installed_pack_cli_generates_manifest_without_research_or_hosting_dependencies(
    tmp_path, monkeypatch, entrypoint
):
    import subprocess
    import sys

    monkeypatch.setenv("YUGOS_ENDPOINT", "http://invalid.example")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("raise RuntimeError('must not import')")
    (source / "model.safetensors").write_bytes(b"weights")
    models = [{"provider_id": "p", "model_id": "m", "effort": "low"}]
    declaration = source / "models.json"
    declaration.write_text(json.dumps(models))
    (source / "router.json").write_text('{"vocab":"tokenizer/vocab.json"}')
    (source / "tokenizer").mkdir()
    (source / "tokenizer/vocab.json").write_text('{"hello":1}')
    script = """
import sys
for name in ('bittensor', 'brr', 'torch', 'modal'):
    sys.modules[name] = None
"""
    if entrypoint == "public":
        script += "from yugos.miner import main\nmain(['pack', sys.argv[1]])\n"
    else:
        script += (
            "import runpy\n"
            "sys.argv = ['yugos', 'pack', sys.argv[1]]\n"
            "runpy.run_module('yugos.miner', run_name='__main__')\n"
        )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    output = Path(json.loads(result.stdout)["output"])
    assert output.parent == source and output.name.startswith("submission-")
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read("router.json") == (source / "router.json").read_bytes()
        assert (
            archive.read("tokenizer/vocab.json") == (source / "tokenizer/vocab.json").read_bytes()
        )
    assert manifest["schema"] == "yugos.bundle.v4"
    assert manifest["name"] == source.name
    assert manifest["required_models"] == models and "fallback" not in manifest
    assert inspect_bundle(output).required_models == tuple(ModelChoice(**m) for m in models)


@pytest.mark.parametrize(
    "command,custom_config,exit_code",
    [("validator", False, 0), ("v", True, 0), ("v", True, 7)],
)
def test_installed_validator_start_wraps_pm2_and_preserves_exit_status(
    tmp_path, monkeypatch, command, custom_config, exit_code
):
    import subprocess
    import sys

    config = tmp_path / ("custom config.js" if custom_config else "validator.config.js")
    config.write_text("module.exports = {};")
    pm2 = tmp_path / "pm2"
    pm2.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"print(json.dumps(sys.argv[1:]))\nsys.exit({exit_code})\n"
    )
    pm2.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    script = "import sys\nfrom yugos.miner import main\nmain(sys.argv[1:])"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, command, "start"]
        + (["--config", str(config)] if custom_config else []),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == ["start", str(config)]


@pytest.mark.parametrize("missing", ["config", "pm2"])
def test_validator_start_reports_missing_prerequisites(tmp_path, monkeypatch, capsys, missing):
    from yugos.miner import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    if missing == "pm2":
        (tmp_path / "validator.config.js").write_text("module.exports = {};")
    with pytest.raises(SystemExit) as exc:
        main(["validator", "start"])
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert ("--config" if missing == "config" else "PATH") in error


@pytest.mark.parametrize("declaration", [None, "{", "[]", '[{"provider_id":"p"}]'])
def test_pack_cli_rejects_missing_or_invalid_models_without_replacing_output(
    tmp_path, declaration, capsys
):
    """Reject unusable miner declarations before importing code or replacing a prior ZIP."""
    from yugos.miner import main

    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    if declaration is not None:
        (tmp_path / "models.json").write_text(declaration)
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous submission")
    with pytest.raises(SystemExit) as exc:
        main(["pack", str(tmp_path), "--output", str(output)])
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "models.json" in error or "required_models" in error
    assert output.read_bytes() == b"previous submission"


def test_pack_cli_cannot_overwrite_its_model_declaration(tmp_path, capsys):
    """Keep the source model declarations intact if chosen as the ZIP destination."""
    from yugos.miner import main

    declaration = tmp_path / "models.json"
    content = '[{"provider_id":"p","model_id":"m"}]'
    declaration.write_text(content)
    with pytest.raises(SystemExit) as exc:
        main(["pack", str(tmp_path), "--output", str(declaration)])
    assert exc.value.code == 1
    assert "output must not overwrite models.json" in capsys.readouterr().err
    assert declaration.read_text() == content


@pytest.mark.parametrize("explicit_output", [False, True])
def test_pack_cli_preserves_previous_archive_on_name_collision(
    tmp_path, monkeypatch, capsys, explicit_output
):
    """Neither a repeated timestamp nor --output may replace an earlier miner pack."""
    from datetime import UTC, datetime

    from yugos.miner import cli as miner

    (tmp_path / "model.py").write_text("raise RuntimeError('must not import')")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "models.json").write_text('[{"provider_id":"p","model_id":"m"}]')
    fixed = datetime(2026, 9, 23, tzinfo=UTC)
    monkeypatch.setattr(miner, "datetime", SimpleNamespace(now=lambda _: fixed))
    args = ["pack", str(tmp_path)]
    if explicit_output:
        args += ["--output", str(tmp_path / "chosen.zip")]
    miner.main(args)
    output = Path(json.loads(capsys.readouterr().out)["output"])
    original = output.read_bytes()
    (tmp_path / "model.py").write_text("# changed model")
    with pytest.raises(SystemExit) as exc:
        miner.main(args)
    assert exc.value.code == 1
    assert "Output already exists" in capsys.readouterr().err
    assert output.read_bytes() == original


def test_package_does_not_overwrite_destination_created_during_validation(tmp_path, monkeypatch):
    """Publish without clobbering a competing pack that wins the destination during validation."""
    from yugos.miner import cli as miner

    (tmp_path / "model.py").write_text("# model")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    output = tmp_path / "submission.zip"

    def inspect_and_race(archive):
        metadata = inspect_bundle(archive)
        output.write_bytes(b"concurrent pack")
        return metadata

    monkeypatch.setattr(miner, "inspect_bundle", inspect_and_race)
    with pytest.raises(FileExistsError, match="Output already exists"):
        package(
            tmp_path,
            output,
            name="router",
            required_models=(ModelChoice(provider_id="p", model_id="m"),),
        )
    assert output.read_bytes() == b"concurrent pack"


@pytest.mark.parametrize("custom_receipt", [False, True])
@pytest.mark.parametrize("endpoint", [None, "https://evaluation.example"])
def test_submit_cli_reuses_receipt_after_lost_confirmation(
    tmp_path, monkeypatch, custom_receipt, endpoint
):
    """A plain submit retry resumes the same paid job; an explicit receipt still takes precedence."""
    from yugos.miner import cli as miner

    monkeypatch.delenv("YUGOS_ENDPOINT", raising=False)
    if endpoint is None:
        endpoint = "https://evaluation.example"
        monkeypatch.setenv("YUGOS_ENDPOINT", endpoint)
    archive, digest, _ = make_bundle(tmp_path)
    key = Keypair.create_from_uri("//Alice")
    server = UploadService(key, archive, fail="confirm")
    chain = MinerChain(key)
    monkeypatch.setattr(miner, "configured_chain", lambda _: chain)
    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(server), **kwargs),
    )
    args = [
        "submit",
        str(archive),
        "--network",
        "test",
        "--wallet",
        "miner",
        "--hotkey",
        "default",
    ]
    if endpoint:
        args += ["--endpoint", endpoint]
    expected = archive.with_suffix(".receipt.json")
    if custom_receipt:
        expected = tmp_path / "custom-receipt.json"
        args += ["--receipt", str(expected)]
    messages = []

    def observe_progress(message):
        if message.startswith(("Submission receipt saved:", "Resuming from receipt:")):
            assert expected.is_file()
            assert json.loads(expected.read_text())["model_id"] > 0
            messages.append(message)

    monkeypatch.setattr(miner, "_say", observe_progress)
    with pytest.raises(SystemExit) as exc:
        miner.main(args)
    assert exc.value.code == 1
    assert expected.is_file()
    assert json.loads(expected.read_text())["endpoint"] == endpoint
    miner.main(args)
    requests = list(server.requests)
    miner.main(args)
    assert server.requests == requests + ["submissions"]
    assert server.requests.count("submissions") == 3
    assert server.jobs == {"job-1"}
    assert len(messages) == 3
    assert messages[0].startswith("Submission receipt saved:")
    assert all(message.startswith("Resuming from receipt:") for message in messages[1:])
    assert chain.commits == [digest]
    assert json.loads(expected.read_text())["committed"] is True
    assert expected.stat().st_mode & 0o077 == 0
    if custom_receipt:
        assert not archive.with_suffix(".receipt.json").exists()


@pytest.mark.parametrize("interrupt_at", ["sleep", "poll"])
def test_ctrl_c_funding_wait_exits_cleanly_and_resumes_same_submission(
    funding_submission, monkeypatch, capsys, interrupt_at
):
    """Interrupting local polling must not cancel funding, reupload, or commit prematurely."""
    from yugos.miner import cli as miner

    archive, path, chain, receipt = funding_submission
    interrupted, prepares, routes = [], [], []

    def interrupt():
        interrupted.append(True)
        raise KeyboardInterrupt

    def respond(request):
        if request.url.path.endswith("/submissions"):
            return cached_submission(path)
        route = request.url.path
        routes.append(route)
        if route.endswith("/billing/register"):
            return httpx.Response(200, json={"submission_ready": False})
        if route.endswith("/billing/prepare"):
            prepares.append(json.loads(request.read()))
            return httpx.Response(
                200,
                json={"state": "started", "job_id": "paid-job"}
                if interrupted
                else {"state": "waiting"},
            )
        if route.endswith("/resume"):
            return httpx.Response(200, json={"state": "waiting"})
        assert request.method == "GET" and "/billing/intents/" in route
        interrupt()

    def sleep(_seconds):
        if interrupt_at == "sleep":
            interrupt()

    monkeypatch.setattr("yugos.miner.funding.time.sleep", sleep)
    monkeypatch.setattr(miner, "configured_chain", lambda _: chain)
    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: client_type(transport=httpx.MockTransport(respond), **kw),
    )
    args = [
        "submit",
        str(archive),
        "--receipt",
        str(path),
        "--network",
        "test",
        "--wallet",
        "miner",
        "--netuid",
        str(chain.netuid),
        "--endpoint",
        receipt["endpoint"],
    ]
    try:
        with pytest.raises(SystemExit) as stopped:
            miner.main(args)
    except KeyboardInterrupt:
        pytest.fail("Funding wait leaked KeyboardInterrupt instead of exiting cleanly")
    assert stopped.value.code == 130
    saved = json.loads(path.read_text())
    assert saved["funding"] == receipt["funding"] and not saved.get("confirmed")
    assert saved["uploaded"] and chain.commits == []
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Submission needs attention" not in captured.err
    miner.main(args)
    resumed = json.loads(path.read_text())
    assert prepares[0] == prepares[1]
    assert resumed["job_id"] == "paid-job" and resumed["committed"]
    assert chain.commits == [receipt["submission_sha256"]]
    assert not any(route.endswith("/cancel") for route in routes)


def test_missing_wallet_does_not_create_or_announce_a_receipt(tmp_path, monkeypatch, capsys):
    """Reject missing signing keys before claiming a submission receipt has been saved."""
    from yugos.miner import cli as miner

    archive, _, _ = make_bundle(tmp_path)

    class MissingWallet:
        @property
        def hotkey(self):
            raise FileNotFoundError("hotkey file does not exist")

    monkeypatch.setattr(
        miner, "configured_chain", lambda _: SimpleNamespace(wallet=MissingWallet())
    )
    with pytest.raises(SystemExit) as exc:
        miner.main(
            [
                "submit",
                str(archive),
                "--network",
                "test",
                "--endpoint",
                "https://evaluation.example",
                "--wallet",
                "missing",
                "--hotkey",
                "default",
            ]
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Submission receipt saved:" not in captured.out
    assert "hotkey file does not exist" in captured.err
    assert not archive.with_suffix(".receipt.json").exists()


@pytest.mark.parametrize("kind", ["files", "directories", "bytes", "ignored"])
def test_pack_stops_discovery_as_soon_as_a_limit_is_exceeded(tmp_path, monkeypatch, kind):
    """An oversized source must stop scanning before reading or creating payloads."""
    from contextlib import contextmanager

    from yugos.miner import cli as miner

    source = tmp_path / "source"
    source.mkdir()
    for index in range(10):
        path = source / (f".item-{index}" if kind == "ignored" else f"item-{index}")
        if kind == "directories":
            path.mkdir()
        else:
            path.write_bytes(b"x" * 10)
    monkeypatch.setattr(miner, "MAX_MEMBERS", 3)
    if kind == "bytes":
        monkeypatch.setattr(miner, "MAX_EXPANDED_BYTES", 5)
    scanned = []
    original = os.scandir

    @contextmanager
    def bounded_scan(directory):
        with original(directory) as entries:

            def counted():
                for entry in entries:
                    scanned.append(entry.path)
                    if len(scanned) >= 10:
                        pytest.fail("continued traversing an already oversized source")
                    yield entry

            yield counted()

    monkeypatch.setattr(miner.os, "scandir", bounded_scan)
    output = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="limit"):
        miner.package(
            source, output, name="router", required_models=[{"provider_id": "p", "model_id": "m"}]
        )
    assert not output.exists()


def test_pack_bounds_actual_bytes_when_a_file_grows_after_discovery(tmp_path, monkeypatch):
    from io import BytesIO

    from yugos.miner import cli as miner

    (tmp_path / "model.py").write_text("# model")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    original = Path.open

    def growing_file(path, *args, **kwargs):
        if path == weights:
            return BytesIO(b"x" * 2048)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(miner, "MAX_EXPANDED_BYTES", 1024)
    monkeypatch.setattr(Path, "open", growing_file)
    output = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="expanded bundle size limit"):
        miner.package(
            tmp_path, output, name="router", required_models=[{"provider_id": "p", "model_id": "m"}]
        )
    assert not output.exists()


def test_pack_rejects_a_compressed_zip_over_the_upload_limit(tmp_path, monkeypatch):
    import yugos.submission.bundle as bundle

    (tmp_path / "model.py").write_text("# model")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", 256)
    output = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="upload limit"):
        package(
            tmp_path, output, name="router", required_models=[{"provider_id": "p", "model_id": "m"}]
        )
    assert not output.exists()


@pytest.mark.parametrize("acknowledged", [True, False])
def test_cancel_upload_keeps_receipt_until_storage_acknowledges_abort(
    funding_submission, monkeypatch, acknowledged
):
    from yugos.miner import cli as miner

    _, path, chain, receipt = funding_submission
    receipt.pop("uploaded")
    receipt.pop("funding")
    receipt["multipart"] = {"upload_id": "session", "parts": {"1": "ack"}}
    save_json(path, receipt)
    monkeypatch.setattr(miner, "configured_chain", lambda _: chain)
    requests = []

    def respond(request):
        if request.url.path.endswith("/submissions"):
            return cached_submission(path)
        requests.append(request)
        assert request.url.path == "/multipart/abort"
        assert json.loads(request.read()) == {"model_id": 3, "upload_id": "session"}
        return httpx.Response(200, json={"ok": acknowledged, "aborted": acknowledged})

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client_type(transport=httpx.MockTransport(respond), **kw)
    )
    args = ["cancel-upload", str(path), "--network", "test", "--wallet", "miner"]
    if acknowledged:
        miner.main(args)
        receipt.pop("multipart")
    else:
        with pytest.raises(SystemExit):
            miner.main(args)
    assert json.loads(path.read_text()) == receipt
    assert len(requests) == 1 and not chain.commits
