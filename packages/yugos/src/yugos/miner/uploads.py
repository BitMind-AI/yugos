"""Resume server-bound storage parts before starting billable evaluation."""

from __future__ import annotations

import math

from yugos.state import save_json

from .display import Activity, UploadProgress

MAX_PART_BYTES = 128 * 1024 * 1024


def continue_multipart(service, archive, receipt, receipt_path, response):
    if response.get("uploaded") is True:
        return
    layout = response.get("multipart")
    if not isinstance(layout, dict):
        raise ValueError("service did not return a multipart upload")
    upload_id, part_bytes = layout.get("upload_id"), layout.get("part_bytes")
    if (
        not isinstance(upload_id, str)
        or not 1 <= len(upload_id) <= 1024
        or type(part_bytes) is not int
        or not 1 <= part_bytes <= MAX_PART_BYTES
    ):
        raise ValueError("service returned an invalid multipart layout")
    size = archive.stat().st_size
    count = math.ceil(size / part_bytes)
    if not 1 <= count <= 10000:
        raise ValueError("service returned too many multipart parts")
    remote = {}
    parts = layout.get("parts")
    if not isinstance(parts, list) or len(parts) > count:
        raise ValueError("service returned an invalid multipart part list")
    for part in parts:
        number = part.get("part_number") if isinstance(part, dict) else None
        if type(number) is not int or not 1 <= number <= count or number in remote:
            raise ValueError("service returned an invalid multipart part number")
        if not isinstance(part.get("etag"), str) or not 1 <= len(part["etag"]) <= 256:
            raise ValueError("service returned an invalid multipart part tag")
        remote[number] = part
    state = receipt.get("multipart") or {}
    if (state.get("upload_id"), state.get("part_bytes")) != (upload_id, part_bytes):
        state = {"upload_id": upload_id, "part_bytes": part_bytes, "parts": {}}
        receipt["multipart"] = state
    save_json(receipt_path, receipt)
    identity = {"model_id": receipt["model_id"], "upload_id": upload_id}
    retained = {
        number
        for number, part in remote.items()
        if part.get("size") == min(part_bytes, size - (number - 1) * part_bytes)
        and state["parts"].get(str(number), part["etag"]) == part["etag"]
    }
    for number in retained:
        state["parts"][str(number)] = remote[number]["etag"]
    save_json(receipt_path, receipt)
    stored = sum(remote[number]["size"] for number in retained)
    with UploadProgress(
        archive.name, size, count, stored=stored, retained=len(retained)
    ) as display:
        for number in range(1, count + 1):
            if number in retained:
                continue
            expected = min(part_bytes, size - (number - 1) * part_bytes)
            for attempt in range(3):
                display.begin(number, attempt)
                target = service.request(
                    "POST", "/multipart/part", {**identity, "part_number": number}, signed=True
                )
                if target.get("size") != expected or target.get("part_number") != number:
                    raise ValueError("service changed the multipart part boundaries")
                try:
                    tag = service.upload_part(
                        target["url"],
                        archive,
                        offset=(number - 1) * part_bytes,
                        size=expected,
                        on_progress=display.advance,
                    )
                    break
                except (OSError, RuntimeError) as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            f"Upload part {number} interrupted; rerun with the same receipt."
                        ) from exc
            state["parts"][str(number)] = tag
            save_json(receipt_path, receipt)
            display.confirm(expected)
    with Activity("Finalizing upload: checking stored bytes on the server."):
        result = service.request("POST", "/multipart/complete", identity, signed=True)
        if (
            result.get("uploaded") is not True
            or result.get("model_id") != receipt["model_id"]
            or result.get("r2_key") != receipt["r2_key"]
        ):
            raise ValueError("service did not verify this completed multipart upload")
