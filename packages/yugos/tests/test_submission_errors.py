"""Actionable submission errors without leaking raw service responses."""

import json

import httpx
import pytest

from yugos.miner.display import submission_error
from yugos.service_client import ServiceClient, ServiceError


@pytest.mark.parametrize("path", ["/submissions", "/presigned"])
@pytest.mark.parametrize(
    "detail",
    [
        "rate limit: 4 new zips per day",
        {"code": "submission_limit_reached", "submissions_max": 4, "submissions_used": 4},
    ],
)
def test_submission_quota_is_explained_without_retrying(path, detail, capsys):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, json={"detail": detail})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ServiceError) as caught:
            ServiceClient("https://evaluation.example", client).request("POST", path, {})
    submission_error(caught.value)
    message = " ".join(capsys.readouterr().err.split())
    assert "Submission limit reached" in message
    assert "4 new ZIPs per hotkey" in message
    assert "rolling 24-hour window" in message
    assert "resume the same ZIP" in message
    assert len(requests) == 1


@pytest.mark.parametrize("limit", [None, "4", True, -1])
def test_bad_quota_metadata_still_explains_the_limit_without_printing_raw_data(limit, capsys):
    detail = {"code": "submission_limit_reached", "submissions_max": limit, "private": "secret"}
    submission_error(ServiceError("/submissions", 429, json.dumps({"detail": detail}).encode()))
    message = " ".join(capsys.readouterr().err.split())
    assert "Submission limit reached" in message
    assert "rolling 24-hour window" in message
    assert "secret" not in message


@pytest.mark.parametrize(
    "path,status,body",
    [
        ("/submissions", 429, b'{"detail":"too many requests, private upstream data"}'),
        ("/billing/register", 429, b'{"detail":"rate limit: 4 new zips per day"}'),
        ("/submissions", 503, b'{"detail":"rate limit: 4 new zips per day"}'),
        ("/submissions", 429, b"private gateway response"),
    ],
)
def test_other_transport_errors_are_not_misreported_as_a_submission_cap(path, status, body, capsys):
    submission_error(ServiceError(path, status, body))
    message = capsys.readouterr().err
    assert "Submission limit reached" not in message
    assert f"HTTP {status}" in message
    assert "private" not in message
