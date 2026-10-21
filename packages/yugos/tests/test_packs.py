"""Packs round-trip byte-for-byte in Ycore's `ycore.pack.v2` format and fail closed."""

import hashlib
import io
import json
import tarfile

import pytest

from yugos.evaluation import Question
from yugos.evaluation.packs import PackManifest, pack_bytes, read_pack, records_bytes, write_pack


def questions():
    return (
        Question(
            question_id="fixture:1",
            prompt="Name the capital of France — in English.",
            inspect_task={},
            inspect_sample={},
            expected_answer="Paris",
            workload_class="chat_qa",
            source_benchmark="fixture",
        ),
        Question(
            question_id="fixture:2",
            prompt="2 + 2?",
            inspect_task={},
            inspect_sample={},
            expected_answer="4",
            workload_class="math_code",
            source_benchmark="fixture",
        ),
    )


def manifest(records, **changes):
    fields = {
        "pack_id": "development-v1",
        "records_sha256": hashlib.sha256(records).hexdigest(),
        "source_releases": ("sha256:" + "a" * 64,),
        "data_visibility": "public",
        "purpose": "development",
    }
    return PackManifest(**{**fields, **changes})


def ycore_pack(manifest_dict, records, calibration=None):
    """Ycore's src/apps/shared/pack.py build_pack serialization, reproduced as the reference."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        files = [
            ("records.jsonl", records),
            (
                "manifest.json",
                json.dumps(
                    manifest_dict,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8"),
            ),
        ]
        if calibration is not None:
            files.append(("difficulty-calibration.json", calibration))
        for name, payload in files:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_pack_round_trips_and_matches_ycore_bytes(tmp_path):
    records = records_bytes(questions())
    # Ycore's release writer: canonical lines without ASCII escaping.
    assert records.decode().splitlines()[0] == json.dumps(
        questions()[0].model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    pinned = manifest(records)
    expected = {
        "schema": "ycore.pack.v2",
        "pack_id": "development-v1",
        "records_sha256": pinned.records_sha256,
        "source_releases": ["sha256:" + "a" * 64],
        "data_visibility": "public",
        "purpose": "development",
        "difficulty_calibration_sha256": None,
        "world_card_sha256": None,
    }
    assert pack_bytes(pinned, records) == ycore_pack(expected, records)

    digest = write_pack(tmp_path / "pack.tar", pinned, records)
    pack = read_pack(tmp_path / "pack.tar")
    assert pack.questions == questions()
    assert pack.manifest == pinned and pack.records == records
    assert pack.sha256 == digest == hashlib.sha256(ycore_pack(expected, records)).hexdigest()
    with pytest.raises(FileExistsError):
        write_pack(tmp_path / "pack.tar", pinned, records)


def test_reader_accepts_a_manifest_without_visibility(tmp_path):
    records = records_bytes(questions())
    first, newline, rest = records.partition(b"\n")
    record = json.loads(first)
    record["grading"] = None
    records = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + newline + rest
    fields = manifest(records).model_dump(mode="json", by_alias=True)
    del fields["data_visibility"]
    fields["evaluation_split"] = "public-development"
    path = tmp_path / "pack.tar"
    path.write_bytes(ycore_pack(fields, records))
    pack = read_pack(path)
    assert pack.manifest.data_visibility is None
    assert "evaluation_split" not in pack.manifest.model_dump()
    assert "grading" not in pack.questions[0].model_dump()


def test_competition_packs_carry_matching_calibration(tmp_path):
    records = records_bytes(questions())
    with pytest.raises(ValueError, match="calibration"):
        pack_bytes(manifest(records, purpose="competition"), records)
    evidence = b'{"calibration": true}'
    pinned = manifest(
        records,
        purpose="competition",
        difficulty_calibration_sha256=hashlib.sha256(evidence).hexdigest(),
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        pack_bytes(pinned, records, difficulty_calibration=b"{}")
    write_pack(tmp_path / "pack.tar", pinned, records, difficulty_calibration=evidence)
    assert read_pack(tmp_path / "pack.tar").difficulty_calibration == evidence


@pytest.mark.parametrize(
    "tamper",
    ["records", "duplicate_ids", "extra_member", "missing_manifest"],
)
def test_tampered_or_malformed_packs_are_rejected(tmp_path, tamper):
    records = records_bytes(questions())
    fields = manifest(records).model_dump(mode="json", by_alias=True)
    members = {"records.jsonl": records}
    if tamper == "records":
        members["records.jsonl"] = records.replace(b"Paris", b"Lyon")
    elif tamper == "duplicate_ids":
        members["records.jsonl"] = records.splitlines(keepends=True)[0] * 2
        fields["records_sha256"] = hashlib.sha256(members["records.jsonl"]).hexdigest()
    elif tamper == "extra_member":
        members["notes.txt"] = b"x"
    if tamper != "missing_manifest":
        members["manifest.json"] = json.dumps(fields).encode()
    path = tmp_path / "pack.tar"
    with tarfile.open(path, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError):
        read_pack(path)
