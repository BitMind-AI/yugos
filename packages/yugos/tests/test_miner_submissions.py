"""Repeated submits and signed status use canonical server state without receipts."""

import json
from types import SimpleNamespace

import httpx
import pytest
from bittensor.wallet import Keypair
from btauth import verify_signed
from bundles import make_bundle
from test_miner import MinerChain

from yugos.miner import cli as miner
from yugos.miner.funding import funding_before_upload
from yugos.service_client import ServiceClient, ServiceError
from yugos.state import save_json
from yugos.submission.bundle import inspect_bundle


@pytest.fixture
def attempt(tmp_path):
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
    }
    submission = {
        "model_id": 3,
        "file_hash": metadata.submission_sha256,
        "r2_key": f"bundles/{key.ss58_address}/{metadata.submission_sha256}.zip",
    }
    funding = {
        "request_id": "canonical-intent",
        "model_id": 3,
        "hotkey": key.ss58_address,
        "state": "paid",
        "submission_ready": False,
        "quote": {
            "file_hash": metadata.submission_sha256,
            "network": "test",
            "netuid": chain.netuid,
            "reservation_cents": 10000,
        },
        "balance": {"available_cents": 10000, "balance_cents": 10000},
        "additional_required_cents": 0,
    }
    return SimpleNamespace(
        archive=archive,
        key=key,
        chain=chain,
        receipt=receipt,
        submission=submission,
        funding=funding,
        path=archive.with_suffix(".receipt.json"),
    )


@pytest.mark.parametrize("state", ["running", "done"])
def test_lost_receipt_recovers_existing_job_without_upload_or_new_funding(attempt, state):
    f = attempt
    calls = []

    def respond(request):
        calls.append(request.url.path)
        assert request.url.path == "/submissions" and request.method == "POST"
        verify_signed(request, f.key)
        payload = json.loads(request.content)
        assert payload["expected_hash"] == f.receipt["submission_sha256"]
        assert payload["network"] == "test" and payload["netuid"] == f.chain.netuid
        return httpx.Response(
            200,
            json={
                "data": {
                    **f.submission,
                    "uploaded": True,
                    "job_id": "existing-job",
                    "funding": {
                        **f.funding,
                        "state": "started",
                        "job_id": "existing-job",
                        "run_status": state,
                        "portal_token": "fresh-token",
                    },
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = miner.submit(
            f.archive,
            f.path,
            ServiceClient(f.receipt["endpoint"], client, f.key),
            f.chain,
            network="test",
        )
    assert (
        result["job_id"] == "existing-job" and result["funding"]["request_id"] == "canonical-intent"
    )
    assert json.loads(f.path.read_text())["confirmed"] is True
    assert calls == ["/submissions"]


def test_paid_receipt_loss_resumes_verification_and_authorizes_original_intent(attempt):
    f = attempt
    calls = []

    def respond(request):
        route = request.url.path
        calls.append(route)
        if route.endswith("/submissions"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        **f.submission,
                        "billing_enabled": True,
                        "uploaded": True,
                        "funding": {**f.funding, "portal_token": "fresh-token"},
                    }
                },
            )
        assert route.endswith("/billing/prepare")
        assert json.loads(request.content)["request_id"] == f.funding["request_id"]
        return httpx.Response(
            200, json={**f.funding, "state": "started", "job_id": "same-paid-job"}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = miner.submit(
            f.archive,
            f.path,
            ServiceClient(f.receipt["endpoint"], client, f.key),
            f.chain,
            network="test",
        )
    assert result["job_id"] == "same-paid-job"
    assert calls == ["/submissions", "/billing/prepare"]


@pytest.mark.parametrize("cache", [False, True, "stale_job", "old_endpoint"])
@pytest.mark.parametrize("funded", [False, True])
def test_status_queries_server_for_paid_unready_submission_without_starting_it(
    attempt, monkeypatch, capsys, cache, funded
):
    f = attempt
    if cache:
        cached = {**f.receipt, "model_id": 3, "r2_key": f.submission["r2_key"]}
        cached["funding"] = {"request_id": "canonical-intent", "portal_token": "cached-token"}
        if cache == "stale_job":
            cached.update(job_id="old-job", confirmed=True)
        if cache == "old_endpoint":
            cached["endpoint"] += "/upload"
        save_json(f.path, cached)
    calls = []

    def respond(request):
        calls.append(request.url.path)
        assert request.method == "GET" and request.url.path.startswith("/submissions/")
        return httpx.Response(
            200,
            json={
                "data": {
                    **f.submission,
                    "uploaded": False,
                    "funding": {**f.funding, "portal_token": "fresh-token"} if funded else None,
                }
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(respond), **kw)
    )
    monkeypatch.setattr(miner, "configured_chain", lambda _: f.chain)
    miner.main(
        [
            "status",
            str(f.path if cache else f.archive),
            "--network",
            "test",
            "--netuid",
            str(f.chain.netuid),
            "--wallet",
            "miner",
            "--endpoint",
            f.receipt["endpoint"],
        ]
    )
    output = capsys.readouterr().out
    assert '"status": "upload_incomplete"' in output
    assert ('"state": "paid"' in output) == funded
    assert "awaiting_confirmation" not in output
    assert output.count("Private benchmark status and refund link: ") == int(funded)
    assert "cached-token" not in output
    saved = json.loads(f.path.read_text())
    assert saved["endpoint"] == f.receipt["endpoint"]
    assert ("canonical-intent:" in saved.get("status_url", "")) == funded
    assert not f.chain.commits
    assert len(calls) == 1


def test_failed_status_recovery_prints_cached_link_once(attempt, monkeypatch, capsys):
    f = attempt
    save_json(
        f.path,
        {
            **f.receipt,
            "model_id": 3,
            "r2_key": f.submission["r2_key"],
            "funding": {"request_id": "canonical-intent", "portal_token": "cached-token"},
        },
    )

    def unavailable(request):
        raise httpx.ConnectError("service unavailable")

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(unavailable), **kw)
    )
    monkeypatch.setattr(miner, "configured_chain", lambda _: f.chain)
    with pytest.raises(SystemExit) as exc:
        miner.main(["status", str(f.path), "--network", "test", "--wallet", "miner"])
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert output.count("Private benchmark status and refund link: ") == 1
    assert "canonical-intent:cached-token" in output


def test_submission_rejects_unrelated_identity_before_persisting_it(attempt):
    f = attempt
    for change in [{"hotkey": "other"}, {"quote": {**f.funding["quote"], "netuid": 1}}]:
        with pytest.raises(ValueError, match="unrelated"):
            miner._submission_state(
                dict(f.receipt),
                {
                    **f.submission,
                    "funding": {**f.funding, "portal_token": "fresh-token", **change},
                },
                f.path,
            )
        assert not f.path.exists()


def test_recovery_failure_never_falls_through_to_a_new_submission(attempt):
    f = attempt
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args[1])
        raise ServiceError(args[1], 503, b"{}")

    service = SimpleNamespace(endpoint=f.receipt["endpoint"], request=unavailable)
    with pytest.raises(ServiceError):
        miner.submit(f.archive, f.path, service, f.chain, network="test")
    assert calls == ["/submissions"] and not f.path.exists()


def test_registration_adopts_canonical_intent_after_concurrent_create(attempt, monkeypatch):
    f = attempt
    receipt = {**f.receipt, "model_id": 3, "r2_key": f.submission["r2_key"]}
    links = []
    monkeypatch.setattr("yugos.miner.funding.status_link", links.append)

    def register(method, route, payload, **kwargs):
        assert payload["request_id"] != f.funding["request_id"]
        return f.funding

    funding_before_upload(SimpleNamespace(request=register), receipt, f.path)
    assert receipt["funding"]["request_id"] == f.funding["request_id"]
    assert links == [receipt["status_url"]]
    assert "canonical-intent:" in links[0]
