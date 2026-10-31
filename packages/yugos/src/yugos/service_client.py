"""Wallet-authenticated requests and storage uploads for the evaluation service."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from .state import canonical_json

if TYPE_CHECKING:
    import httpx

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


DEFAULT_MINER_ENDPOINT = "https://upload.yugos.ai"


def resolve_miner_endpoint(explicit: str | None, network: str | None = None) -> str:
    """Use an explicit service, then YUGOS_ENDPOINT, then the public mainnet service.

    The test network has no built-in service URL.
    """
    if explicit:
        return service_url(explicit)
    configured = os.environ.get("YUGOS_ENDPOINT")
    if configured:
        return service_url(configured)
    if network == "test":
        raise ValueError("pass --endpoint or set YUGOS_ENDPOINT for the test network")
    return DEFAULT_MINER_ENDPOINT


def service_url(url: str) -> str:
    """Validate HTTPS and use root routes for URLs saved before the upload cutover."""
    parsed = urlsplit(url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and not (local and parsed.scheme == "http"))
    ):
        raise ValueError(
            "service URL must be HTTPS (or localhost HTTP), without credentials or query"
        )
    if parsed.path.rstrip("/") == "/upload":
        return parsed._replace(path="").geturl()
    return url.rstrip("/")


class ServiceError(RuntimeError):
    """An HTTP failure with a machine-readable service error code."""

    def __init__(self, path, status, raw):
        self.path = path
        self.status = status
        self.detail = {}
        try:
            data = json.loads(raw)
            detail = data.get("detail", {})
            self.code = detail.get("code") if isinstance(detail, dict) else None
            if isinstance(detail, dict):
                self.detail = detail
            # Recognize the upload service's response from before structured quota errors.
            elif (
                status == 429 and path in {"/submissions", "/presigned"} and isinstance(detail, str)
            ):
                limit = re.fullmatch(r"rate limit: ([1-9][0-9]*) new zips per day", detail)
                if limit:
                    self.code = "submission_limit_reached"
                    self.detail = {"submissions_max": int(limit[1])}
            # bt.http_auth.StaleRequest, returned as the 401 detail by the service.
            if isinstance(detail, str) and detail.startswith("nonce is older than the "):
                self.code = "request_signature_expired"
        except (ValueError, AttributeError):
            self.code = None
        super().__init__(
            f"hosted {path} returned HTTP {status}: {raw.decode(errors='replace')[:1000]}"
        )


class ServiceClient:
    """Speak the hosted upload and results API with bounded responses."""

    def __init__(self, endpoint: str, client: httpx.Client, hotkey=None):
        self.endpoint = service_url(endpoint)
        self.client = client
        self.hotkey = hotkey

    def _btauth(self, method: str, url: str, body: bytes) -> dict[str, str]:
        """Sign this call with the hotkey using btauth/1.

        The request target is the encoded path and query httpx will send.
        The service is not a neuron, so the signature is unbound and omits
        ``X-Bittensor-Receiver``.
        """
        import bittensor as bt

        parsed = urlsplit(url)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        return bt.http_auth.sign(self.hotkey, method=method, path=target, body=body)

    def request(self, method: str, path: str, payload=None, *, signed: bool = False) -> dict:
        """Sign hosted requests and wait for delayed results without replaying the operation."""
        body = b"" if payload is None else canonical_json(payload)
        headers = {"Content-Type": "application/json"}
        url = self.endpoint + path
        if signed:
            headers.update(self._btauth(method, url, body))
        # Submission admission and large object verification can outlast metadata reads.
        options = (
            {"timeout": 1800}
            if path
            in {"/submissions", "/presigned", "/confirm", "/billing/prepare", "/multipart/complete"}
            or path.endswith("/resume")
            else {}
        )
        original = urlsplit(url)
        redirects = authentication_retries = 0
        while True:
            with self.client.stream(
                method,
                url,
                content=body,
                headers=headers,
                follow_redirects=False,
                **options,
            ) as response:
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ValueError("hosted service response exceeds the size limit")
                if response.status_code == 303 and response.headers.get("Location"):
                    target = urlsplit(urljoin(url, response.headers["Location"]))
                    if (
                        (target.scheme, target.netloc, target.path)
                        != (original.scheme, original.netloc, original.path)
                        or target.username is not None
                        or target.password is not None
                        or target.fragment
                        or not target.query
                    ):
                        raise ValueError("hosted service returned an unsafe result redirect")
                    if redirects == 20:
                        raise RuntimeError(
                            "hosted service result is still pending; rerun with the same receipt"
                        )
                    # A long request may answer with a same-path result URL. Retrieve it
                    # without replaying the POST body or forwarding wallet authentication.
                    method, url, body, headers = "GET", target.geturl(), b"", {}
                    options = {"timeout": 180}
                    redirects += 1
                    continue
                if not response.is_success:
                    error = ServiceError(path, response.status_code, raw)
                    import bittensor as bt

                    if (
                        response.status_code == 401
                        and error.code == "request_signature_expired"
                        and bt.http_auth.HEADER_SIGNATURE in headers
                        and authentication_retries < 2
                    ):
                        # Authentication failed before dispatch, so a fresh nonce
                        # can safely retry after a cold start without replaying work.
                        authentication_retries += 1
                        headers.update(self._btauth(method, url, body))
                        continue
                    raise error
                break
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("hosted service response must be a JSON object")
        return result

    def upload_part(
        self, url: str, archive: Path, *, offset: int, size: int, on_progress=None
    ) -> str:
        """Send one bounded file segment using only the storage URL's authority."""
        import httpx

        parsed = urlsplit(url)
        service_url(parsed._replace(query="").geturl())

        def chunks():
            with archive.open("rb") as source:
                source.seek(offset)
                remaining = size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("submission ZIP changed during upload")
                    remaining -= len(chunk)
                    yield chunk
                    if on_progress is not None:
                        on_progress(len(chunk))

        try:
            with self.client.stream(
                "PUT",
                url,
                content=chunks(),
                headers={"Content-Length": str(size)},
                follow_redirects=False,
                timeout=300,
            ) as response:
                if not response.is_success:
                    raise RuntimeError(f"part upload returned HTTP {response.status_code}")
                tag = response.headers.get("etag", "")
                if not 1 <= len(tag) <= 256:
                    raise ValueError("storage did not return a valid part tag")
                return tag
        except httpx.TransportError as exc:
            raise OSError("part upload connection interrupted") from exc
