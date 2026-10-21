"""Resume a server-authorized submission after Stripe Checkout without reporting payment."""

import os
import secrets
import threading
import time
import webbrowser
from uuid import uuid4

from yugos.service_client import ServiceError, service_url
from yugos.state import save_json

from .display import (
    Activity,
    funding_next_step,
    funding_summary,
    funding_wait_hint,
    phase,
    progress_message,
    status_link,
)


class FundingWaitInterrupted(KeyboardInterrupt):
    """Stop local polling while retaining the server-authorized funding intent."""

    def __init__(self, receipt_path, status_url):
        super().__init__("Stopped waiting for funding.")
        self.receipt_path = receipt_path
        self.status_url = status_url


def report_balance(service):
    """Report account status when evaluation funding cannot be authorized."""
    import httpx

    try:
        with Activity("Checking available funds."):
            balance = service.request("GET", "/credits", {}, signed=True)
        funding_summary(balance)
        return balance
    except (OSError, RuntimeError, ValueError, httpx.HTTPError):
        phase(
            "Funding", "Balance unavailable; it will be checked before evaluation.", style="yellow"
        )
        return None


def receipt_status_url(receipt):
    """Build the private funding link from the receipt's saved capability."""
    funding = receipt.get("funding")
    if not funding:
        return None
    return (
        service_url(receipt["endpoint"])
        + "/billing/portal#"
        + funding["request_id"]
        + ":"
        + funding["portal_token"]
    )


def save_status_url(receipt, receipt_path):
    """Keep the private status URL explicit, including in receipts from older clients."""
    url = receipt_status_url(receipt)
    if url is None:
        return None
    if receipt.get("status_url") != url:
        receipt["endpoint"] = service_url(receipt["endpoint"])
        receipt["status_url"] = url
        save_json(receipt_path, receipt)
    return url


def _prepare_funding(service, receipt, receipt_path, *, before_upload=False):
    if not receipt.get("funding"):
        receipt["funding"] = {"request_id": str(uuid4()), "portal_token": secrets.token_urlsafe(32)}
        save_json(receipt_path, receipt)
    funding = receipt["funding"]
    scope = {key: receipt[key] for key in ("network", "netuid") if key in receipt}
    payload = {**funding, **scope, "model_id": receipt["model_id"]}
    route = "/billing/register" if before_upload else "/billing/prepare"
    if before_upload:
        payload["file_hash"] = receipt["submission_sha256"]
    else:
        save_status_url(receipt, receipt_path)
    with Activity(
        "Preparing funding before upload."
        if before_upload
        else "Checking funding and authorizing evaluation."
    ):
        try:
            status = service.request("POST", route, payload, signed=True)
        except ServiceError as exc:
            if before_upload and exc.status == 404:
                raise RuntimeError(
                    "The submission service is out of date. Your bundle and receipt "
                    "are safe; retry after the service is updated."
                ) from None
            raise
    canonical = status.get("request_id", funding["request_id"])
    if not isinstance(canonical, str) or not canonical:
        raise ValueError("service returned no funding identity")
    if canonical != funding["request_id"]:
        receipt["funding"]["request_id"] = canonical
        save_json(receipt_path, receipt)
    return status, save_status_url(receipt, receipt_path)


def _present_funding(status, link, *, previous=None, before_upload=False):
    def summary(value):
        return (
            value.get("balance"),
            value.get("quote", {}).get("reservation_cents"),
            value.get("additional_required_cents"),
        )

    if previous is None or (not status.get("job_id") and summary(status) != summary(previous)):
        funding_summary(
            status.get("balance"),
            required=status.get("quote", {}).get("reservation_cents"),
            shortfall=status.get("additional_required_cents"),
        )
    if previous is None:
        status_link(link)
    if status.get("checkout_url") and status.get("state") == "waiting":
        from urllib.parse import urlsplit

        url = status["checkout_url"]
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "checkout.stripe.com"
            or parsed.username
            or parsed.password
        ):
            raise ValueError("invalid Stripe checkout URL")
        if previous is None or previous.get("checkout_url") != url:
            funding_next_step(
                reserved=status["quote"]["reservation_cents"] / 100,
                shortfall=status["additional_required_cents"] / 100,
                before_upload=before_upload,
            )
            open_link(link)
    elif before_upload and not status.get("additional_required_cents"):
        phase("Funding", "Balance covers the evaluation allowance", style="green")


def funding_before_upload(service, receipt, receipt_path, *, status=None):
    """Offer funding while bytes transfer without authorizing evaluation."""
    if status is None:
        status, link = _prepare_funding(service, receipt, receipt_path, before_upload=True)
    else:
        link = save_status_url(receipt, receipt_path)
    _present_funding(status, link, before_upload=True)
    if status.get("state") in {"expired", "attention", "canceled"}:
        raise RuntimeError(
            "Funding needs a renewed quote. Rerun with --renew-funding; "
            "received funds remain available in your account."
        )
    return status


def open_link(url: str) -> None:
    """Open a payment page without letting the browser's own warnings look like a failure.

    Child processes inherit file descriptor 2, which is not always ``sys.stderr``.
    """
    opened = False
    try:
        saved = os.dup(2)
    except OSError:
        saved = None
    try:
        if saved is None:
            opened = _try_open(url)
        else:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), 2)
                opened = _try_open(url)
    finally:
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)
    if not opened:
        print("Browser did not open. Use the funding & submission link above.", flush=True)


def _try_open(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


class _ResumeRequest:
    """Keep status polling responsive while the server prepares a funded run."""

    def __init__(self, service, path):
        self.done = threading.Event()
        self.status = self.error = None

        def run():
            try:
                self.status = service.request("POST", path + "/resume", {}, signed=True)
            except (Exception, KeyboardInterrupt) as exc:
                self.error = exc
            finally:
                self.done.set()

        threading.Thread(target=run, daemon=True).start()

    def result(self):
        if self.error:
            raise self.error
        return self.status


def continue_funding(
    service, receipt, receipt_path, *, wait_seconds=600, poll_seconds=5, previous_status=None
):
    try:
        status, link = _prepare_funding(service, receipt, receipt_path)
    except ServiceError as exc:
        if exc.code == "credits_locked" and previous_status is None:
            report_balance(service)
        raise
    path = "/billing/intents/" + receipt["funding"]["request_id"]
    _present_funding(status, link, previous=previous_status)
    if status.get("blocking_run_id") and not status.get("job_id"):
        raise RuntimeError(
            f"Evaluation {status['blocking_run_id']} is still active. "
            "Your ZIP is retained; use the funding status link to follow progress."
        )
    deadline = time.monotonic() + wait_seconds
    noted_at = time.monotonic() - 15
    resume_request = None
    payment_reported = False
    if not status.get("job_id"):
        funding_wait_hint()
    try:
        with Activity("Waiting for funding confirmation") as waiting:
            while not status.get("job_id"):
                if resume_request is not None and resume_request.done.is_set():
                    status = resume_request.result()
                    resume_request = None
                    if status.get("job_id"):
                        break
                if status.get("state") in {"expired", "attention", "canceled"}:
                    raise RuntimeError(
                        "Funding needs attention: "
                        + (status.get("detail") or status["state"]).rstrip(".")
                        + ". Keep this receipt; received funds remain in your account."
                    )
                if time.monotonic() >= deadline:
                    if status.get("state") == "paid":
                        raise RuntimeError(
                            "Payment is confirmed; evaluation preparation is still running. "
                            "Resume with the same receipt. No additional payment is needed."
                        )
                    raise RuntimeError(
                        "Payment is still pending. Re-run with the same receipt; "
                        "the server will continue after payment even if this CLI closes."
                    )
                paid = status.get("state") == "paid"
                if paid and not payment_reported:
                    phase("Payment", "Confirmed · preparing evaluation", style="green")
                    funding_summary(status.get("balance"))
                    payment_reported = True
                    noted_at = time.monotonic() - 15
                now = time.monotonic()
                if now - noted_at >= 15:
                    remaining = max(0, int(deadline - now))
                    state = status.get("state") or "pending"
                    if waiting.interactive:
                        waiting.update(
                            (
                                "Payment confirmed · preparing evaluation"
                                if paid
                                else "Waiting for funds"
                                if state == "waiting"
                                else "Preparing evaluation"
                            )
                            + f" · {remaining // 60}:{remaining % 60:02d} remaining"
                        )
                    else:
                        print(
                            (
                                "Payment confirmed; preparing evaluation. "
                                if paid
                                else f"Waiting for payment ({state}). "
                            )
                            + f"{remaining}s left in this wait; "
                            "rerun the same command with this receipt to continue.",
                            flush=True,
                        )
                    noted_at = now
                if resume_request is None:
                    resume_request = _ResumeRequest(service, path)
                time.sleep(poll_seconds)
                status = service.request("GET", path, {}, signed=True)
    except KeyboardInterrupt:
        raise FundingWaitInterrupted(receipt_path, link) from None
    progress_message(f"Funding ready. Starting evaluation job {status['job_id']}.")
    return {
        "ok": True,
        "model_id": receipt["model_id"],
        "r2_key": receipt["r2_key"],
        "job_id": status["job_id"],
    }


def submission_status(view):
    """Describe the server's payment and upload state before a job exists."""
    if view.get("state") in {"expired", "canceled", "attention"}:
        return view["state"]
    if view.get("state") == "waiting":
        return "awaiting_funding"
    if view.get("submission_ready") is False:
        return "upload_incomplete"
    return "waiting_to_start" if view.get("state") == "paid" else "awaiting_authorization"
