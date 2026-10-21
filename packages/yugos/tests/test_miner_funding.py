"""A lost payment response must resume the same submission and private portal."""

import json

import pytest

from yugos.miner.funding import continue_funding
from yugos.service_client import ServiceError


class Service:
    endpoint = "https://example.test/upload"

    def __init__(self):
        self.calls = []
        self.pending = True

    def request(self, method, path, payload, *, signed):
        assert signed
        self.calls.append((path, payload))
        if path == "/billing/prepare":
            self.identity = payload["request_id"]
            return {
                "state": "waiting",
                "checkout_url": "https://checkout.stripe.com/c/pay/test",
                "quote": {"reservation_cents": 10000},
                "additional_required_cents": 6500,
            }
        if method == "GET":
            return {"state": "waiting"}
        if self.pending:
            self.pending = False
            raise OSError("response lost")
        assert path == "/billing/intents/" + self.identity + "/resume"
        return {"state": "started", "job_id": "job-test"}


def test_funding_reuses_identity_after_interrupted_payment(tmp_path, monkeypatch):
    monkeypatch.setattr("yugos.miner.funding.webbrowser.open", lambda _: False)
    receipt = {"endpoint": Service.endpoint, "model_id": 12, "r2_key": "bundles/test"}
    path = tmp_path / "receipt.json"
    service = Service()
    with pytest.raises(OSError):
        continue_funding(service, receipt, path, poll_seconds=0)
    saved = json.loads(path.read_text())
    funding = saved["funding"]
    assert saved["status_url"] == (
        f"{Service.endpoint}/billing/portal#{funding['request_id']}:{funding['portal_token']}"
    )
    result = continue_funding(service, saved, path, poll_seconds=0)
    assert result["job_id"] == "job-test"
    prepares = [data for route, data in service.calls if route == "/billing/prepare"]
    assert prepares[0] == prepares[1]
    assert path.stat().st_mode & 0o077 == 0


def test_private_status_url_is_saved_before_prepare_can_fail(tmp_path):
    """A lost funding response must leave a directly usable link beside the ZIP."""
    path = tmp_path / "receipt.json"
    receipt = {"endpoint": "https://example.test/upload/", "model_id": 12, "r2_key": "x"}

    class Unavailable:
        def request(self, method, route, payload, *, signed):
            saved = json.loads(path.read_text())
            assert saved["status_url"] == (
                "https://example.test/upload/billing/portal#"
                f"{payload['request_id']}:{payload['portal_token']}"
            )
            assert "status_url" not in payload
            raise OSError("response lost")

    with pytest.raises(OSError, match="response lost"):
        continue_funding(Unavailable(), receipt, path)
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("outdated_service", [False, True])
def test_receipt_page_is_not_announced_until_registration_is_acknowledged(
    tmp_path, monkeypatch, outdated_service
):
    """A lost registration response must retry the same capability without advertising a broken URL."""
    from yugos.miner.funding import funding_before_upload
    from yugos.state import save_json

    receipt = {
        "endpoint": "https://example.test/upload",
        "model_id": 12,
        "submission_sha256": "a" * 64,
        "network": "test",
        "netuid": 169,
        "funding": {"request_id": "request", "portal_token": "private"},
    }
    path = tmp_path / "receipt.json"
    save_json(path, receipt)
    links, calls = [], []
    monkeypatch.setattr("yugos.miner.funding.status_link", links.append)

    class Registration:
        def request(self, method, route, payload, *, signed):
            assert method == "POST" and route == "/billing/register" and signed
            assert not links
            calls.append(payload)
            if len(calls) == 1:
                if outdated_service:
                    raise ServiceError(route, 404, b'{"detail":"Not Found"}')
                raise OSError("response lost")
            return {"submission_ready": False}

    service = Registration()
    with pytest.raises(
        RuntimeError if outdated_service else OSError,
        match="out of date" if outdated_service else "response lost",
    ):
        funding_before_upload(service, receipt, path)
    assert not links and json.loads(path.read_text()) == receipt
    funding_before_upload(service, receipt, path)
    assert calls[0] == calls[1]
    assert links == [json.loads(path.read_text())["status_url"]]


def test_payment_wait_reports_progress_until_the_job_starts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("yugos.miner.funding.webbrowser.open", lambda _: True)

    class Waiting:
        endpoint = "https://example.test/upload"

        def __init__(self):
            self.resumes = 0

        def request(self, method, path, payload, *, signed):
            if path == "/billing/prepare":
                return {
                    "state": "waiting",
                    "checkout_url": "https://checkout.stripe.com/c/pay/test",
                    "quote": {"reservation_cents": 10000},
                    "additional_required_cents": 50,
                }
            if method == "GET":
                return {"state": "waiting"}
            self.resumes += 1
            if self.resumes == 1:
                return {"state": "waiting"}
            return {"state": "started", "job_id": "job-test"}

    result = continue_funding(
        Waiting(),
        {"endpoint": Waiting.endpoint, "model_id": 1, "r2_key": "bundles/test"},
        tmp_path / "receipt.json",
        poll_seconds=0,
        wait_seconds=30,
    )
    output = capsys.readouterr().out
    assert "Waiting for payment (waiting)." in output
    assert "Starting evaluation job job-test." in output
    assert result["job_id"] == "job-test"


def test_browser_warnings_stay_out_of_the_submission_output(monkeypatch, capfd):
    import os

    from yugos.miner.funding import open_link

    def noisy(_url):
        os.write(2, b"DEP0169 noise\n")
        return True

    monkeypatch.setattr("yugos.miner.funding.webbrowser.open", noisy)
    open_link("https://checkout.stripe.com/c/pay/test")
    captured = capfd.readouterr()
    assert "DEP0169" not in captured.err
    assert "DEP0169" not in captured.out


def test_expired_quote_does_not_start_another_submission(tmp_path):
    class Expired:
        endpoint = "https://example.test"

        def request(self, *args, **kwargs):
            return {"state": "expired", "detail": "Quote expired"}

    with pytest.raises(RuntimeError, match="expired"):
        continue_funding(
            Expired(),
            {"endpoint": Expired.endpoint, "model_id": 1, "r2_key": "x"},
            tmp_path / "receipt.json",
            poll_seconds=0,
        )


def test_refund_switches_to_batch_endpoint_with_same_identity(tmp_path):
    from types import SimpleNamespace

    from yugos.miner.credits import refund
    from yugos.service_client import ServiceError

    class RefundService:
        endpoint = "https://example.test"
        hotkey = SimpleNamespace(ss58_address="hot")
        calls = []

        def request(self, method, path, payload, *, signed):
            self.calls.append((path, payload))
            if path == "/credits/refund":
                raise ServiceError(path, 409, b'{"detail":{"code":"use_billing_refund"}}')
            return {"ok": True, "request_id": payload["request_id"], "refunds": []}

    s = RefundService()
    refund(s, 1000, tmp_path / "refund.json")
    assert s.calls[0][1] == s.calls[1][1]


def test_pending_refund_is_rechecked_with_same_identity(tmp_path):
    from types import SimpleNamespace

    from yugos.miner.credits import refund

    class Pending:
        endpoint = "https://example.test"
        hotkey = SimpleNamespace(ss58_address="hot")
        calls = []

        def request(self, method, path, payload, *, signed):
            self.calls.append(payload.copy())
            return {
                "ok": True,
                "request_id": payload["request_id"],
                "refunds": [{"status": "pending" if len(self.calls) == 1 else "succeeded"}],
            }

    service = Pending()
    path = tmp_path / "refund.json"
    refund(service, 500, path)
    result = refund(service, 500, path)
    assert service.calls[0] == service.calls[1]
    assert result["response"]["refunds"][0]["status"] == "succeeded"
    refund(service, 500, path)
    assert len(service.calls) == 2


def test_paid_status_and_job_are_observable_while_resume_request_is_blocked(tmp_path, monkeypatch):
    """A slow verification response must not hide confirmed payment or a queued job."""
    import threading

    started, release = threading.Event(), threading.Event()

    class Preparing:
        endpoint = "https://example.test/upload"

        def request(self, method, path, payload, *, signed):
            if path == "/billing/prepare":
                return {"state": "waiting", "quote": {"reservation_cents": 10000}}
            if method == "POST":
                started.set()
                assert release.wait(3), "status polling blocked behind resume"
                return {"state": "started", "job_id": "funded-run"}
            assert started.wait(1)
            # The durable job can become visible before the resume response returns.
            return {"state": "started", "job_id": "funded-run"}

    try:
        result = continue_funding(
            Preparing(),
            {"endpoint": Preparing.endpoint, "model_id": 1, "r2_key": "bundle"},
            tmp_path / "receipt.json",
            poll_seconds=0,
            wait_seconds=5,
        )
        assert result["job_id"] == "funded-run"
        assert not release.is_set()
    finally:
        release.set()
