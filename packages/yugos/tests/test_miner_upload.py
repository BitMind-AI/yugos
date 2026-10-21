"""Server-bound parts resume safely and transport stays bounded."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from btauth import assert_unsigned

from yugos.miner.uploads import continue_multipart
from yugos.service_client import ServiceClient


def test_part_transport_streams_only_the_requested_segment(tmp_path):
    archive = tmp_path / "model.zip"
    data = b"prefix" + b"x" * (2 * 1024 * 1024 + 7) + b"suffix"
    archive.write_bytes(data)
    updates = []

    def respond(request):
        assert_unsigned(request.headers)
        chunks = list(request.stream)
        assert max(map(len, chunks)) <= 1024 * 1024
        assert b"".join(chunks) == data[6:-6]
        assert int(request.headers["content-length"]) == len(data) - 12
        assert len(updates) > 1 and sum(updates) == len(data) - 12
        return httpx.Response(200, headers={"etag": '"part-tag"'})

    class Transport(httpx.BaseTransport):
        def handle_request(self, request):
            return respond(request)

    with httpx.Client(transport=Transport()) as client:
        service = ServiceClient("https://api.example", client)
        assert (
            service.upload_part(
                "https://objects.example/upload?signature=private",
                archive,
                offset=6,
                size=len(data) - 12,
                on_progress=updates.append,
            )
            == '"part-tag"'
        )


@pytest.mark.parametrize("retained", ["match", "missing-receipt", "changed-tag", "changed-size"])
def test_resume_checks_server_parts_against_available_acknowledgements(tmp_path, retained):
    archive = tmp_path / "model.zip"
    archive.write_bytes(b"abcdefgh")
    receipt = {
        "model_id": 1,
        "r2_key": "object",
        "submission_sha256": hashlib.sha256(b"abcdefgh").hexdigest(),
        "multipart": {"upload_id": "session", "part_bytes": 4, "parts": {"1": "saved"}},
    }
    remote = {"part_number": 1, "size": 4, "etag": "saved"}
    if retained == "changed-tag":
        remote["etag"] = "other"
    if retained == "changed-size":
        remote["size"] = 3
    if retained == "missing-receipt":
        receipt.pop("multipart")
    sent, attempts = [], []
    path = tmp_path / "receipt.json"

    def request(method, route, payload, **kwargs):
        assert kwargs["signed"]
        if route.endswith("/part"):
            attempts.append(payload["part_number"])
            return {
                "url": "https://objects.example/part",
                "size": 4,
                "part_number": payload["part_number"],
            }
        assert json.loads(path.read_text())["multipart"]["parts"]["2"] == "ack"
        return {"uploaded": True, "model_id": 1, "r2_key": "object"}

    def upload_part(url, source, *, offset, size, on_progress=None):
        sent.append(offset)
        if len(sent) == 1:
            raise OSError("connection lost")
        if on_progress is not None:
            on_progress(size)
        return "ack"

    continue_multipart(
        SimpleNamespace(request=request, upload_part=upload_part),
        archive,
        receipt,
        path,
        {"multipart": {"upload_id": "session", "part_bytes": 4, "parts": [remote]}},
    )
    expected = [4, 4] if retained in {"match", "missing-receipt"} else [0, 0, 4]
    assert sent == expected and attempts == [offset // 4 + 1 for offset in expected]
    assert "objects.example" not in path.read_text()


@pytest.mark.parametrize(
    "layout",
    [
        {"upload_id": "x", "part_bytes": 0, "parts": []},
        {"upload_id": "x", "part_bytes": 1, "parts": [{"part_number": 2, "etag": "x"}]},
        {"upload_id": "x", "part_bytes": 1, "parts": [{"part_number": 1, "etag": ""}]},
    ],
)
def test_invalid_layout_fails_before_any_upload(tmp_path, layout):
    archive = tmp_path / "model.zip"
    archive.write_bytes(b"x")
    with pytest.raises(ValueError, match="invalid multipart"):
        continue_multipart(object(), archive, {}, tmp_path / "receipt.json", {"multipart": layout})


def test_missing_multipart_layout_rejects_legacy_upload_url_without_sending_bytes(tmp_path):
    archive = tmp_path / "model.zip"
    archive.write_bytes(b"submission")
    with pytest.raises(ValueError, match="multipart upload"):
        continue_multipart(
            object(),
            archive,
            {},
            tmp_path / "receipt.json",
            {"presigned_url": "https://objects.example/legacy"},
        )
