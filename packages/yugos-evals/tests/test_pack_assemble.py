"""`pack assemble` merges verified packs and writes nothing when a check fails."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from yugos.evaluation import Question
from yugos.evaluation.packs import PackManifest, read_pack, records_bytes, write_pack
from yugos_evals.cli import main
from yugos_evals.local.assemble import assemble_packs, resolve_pack_inputs

CARD = "ab" * 32
OTHER_CARD = "cd" * 32


def question(benchmark: str, sample: str, *, answer: str = "A") -> Question:
    return Question(
        question_id=f"{benchmark}:{sample}",
        prompt=f"Question {benchmark}:{sample}?",
        expected_answer=answer,
        inspect_task={},
        inspect_sample={},
        workload_class="chat_qa",
        source_benchmark=benchmark,
    )


def write_fixture(
    path: Path,
    questions: list[Question],
    *,
    purpose: str = "development",
    visibility: str | None = "public",
    releases: tuple[str, ...] = ("fixture@1",),
    calibration: bytes | None = None,
    world_card: str | None = None,
) -> Path:
    records = records_bytes(questions)
    write_pack(
        path,
        PackManifest(
            pack_id="fixture",
            records_sha256=hashlib.sha256(records).hexdigest(),
            source_releases=releases,
            data_visibility=visibility,
            purpose=purpose,
            difficulty_calibration_sha256=(
                None if calibration is None else hashlib.sha256(calibration).hexdigest()
            ),
            world_card_sha256=world_card,
        ),
        records,
        difficulty_calibration=calibration,
    )
    return path


def members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


def rewrite(path: Path, payloads: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_assembles_sorted_records_and_merged_manifest(tmp_path):
    write_fixture(
        tmp_path / "zeta.tar",
        [question("zeta", "1")],
        releases=("zeta@1", "shared@1"),
        world_card=CARD,
    )
    write_fixture(
        tmp_path / "alpha.tar",
        [question("alpha", "2"), question("alpha", "1")],
        releases=("alpha@1", "shared@1"),
        world_card=CARD,
    )
    out = tmp_path / "out.tar"
    # Zeta is passed first so input order is not question-id order.
    assembled = assemble_packs([tmp_path / "zeta.tar", tmp_path / "alpha.tar"], out)
    pack = read_pack(out)
    assert [item.question_id for item in pack.questions] == ["alpha:1", "alpha:2", "zeta:1"]
    assert pack.records == records_bytes(pack.questions)
    digest = hashlib.sha256(pack.records).hexdigest()
    assert pack.manifest.records_sha256 == digest
    assert pack.manifest.pack_id == f"local-assembled-{digest[:12]}"
    assert pack.manifest.source_releases == ("alpha@1", "shared@1", "zeta@1")
    assert pack.manifest.purpose == "development"
    assert pack.manifest.data_visibility == "public"
    assert pack.manifest.schema_name == "ycore.pack.v2"
    assert pack.manifest.world_card_sha256 == CARD
    assert pack.manifest.difficulty_calibration_sha256 is None
    assert pack.difficulty_calibration is None
    assert assembled.sha256 == pack.sha256
    assert assembled.question_counts == (("alpha", 2), ("zeta", 1))
    assert assembled.question_count == 3


def test_world_card_hash_is_dropped_when_inputs_disagree(tmp_path):
    write_fixture(tmp_path / "a.tar", [question("a", "1")], world_card=CARD)
    write_fixture(tmp_path / "b.tar", [question("b", "1")], world_card=OTHER_CARD)
    out = tmp_path / "out.tar"
    assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    assert read_pack(out).manifest.world_card_sha256 is None


def test_duplicate_question_id_writes_nothing(tmp_path):
    write_fixture(tmp_path / "a.tar", [question("a", "1")])
    write_fixture(tmp_path / "b.tar", [question("a", "1")])
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="duplicate question_id 'a:1'"):
        assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    assert not out.exists()


def test_records_sha256_mismatch_writes_nothing(tmp_path):
    source = write_fixture(tmp_path / "a.tar", [question("a", "1", answer="Paris")])
    payloads = members(source)
    payloads["records.jsonl"] = payloads["records.jsonl"].replace(b"Paris", b"LyonX")
    bad = tmp_path / "bad.tar"
    rewrite(bad, payloads)
    write_fixture(tmp_path / "b.tar", [question("b", "1")])
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="records"):
        assemble_packs([tmp_path / "b.tar", bad], out)
    assert not out.exists()


def test_manifest_that_does_not_parse_writes_nothing(tmp_path):
    source = write_fixture(tmp_path / "a.tar", [question("a", "1")])
    payloads = members(source)
    manifest = json.loads(payloads["manifest.json"])
    manifest["schema"] = "ycore.pack.v1"
    payloads["manifest.json"] = json.dumps(manifest).encode()
    bad = tmp_path / "bad.tar"
    rewrite(bad, payloads)
    write_fixture(tmp_path / "b.tar", [question("b", "1")])
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError):
        assemble_packs([bad, tmp_path / "b.tar"], out)
    assert not out.exists()


def omit_visibility(path: Path) -> Path:
    payloads = members(path)
    manifest = json.loads(payloads["manifest.json"])
    del manifest["data_visibility"]
    payloads["manifest.json"] = json.dumps(manifest).encode()
    rewrite(path, payloads)
    return path


def test_unset_visibility_adopts_the_set_value(tmp_path):
    # Omitted and JSON null both load as None. The legacy pack omits the field.
    legacy = omit_visibility(write_fixture(tmp_path / "legacy.tar", [question("a", "1")]))
    assert read_pack(legacy).manifest.data_visibility is None
    write_fixture(tmp_path / "null.tar", [question("c", "1")], visibility=None)
    write_fixture(tmp_path / "public.tar", [question("b", "1")], visibility="public")
    out = tmp_path / "out.tar"
    assemble_packs([legacy, tmp_path / "null.tar", tmp_path / "public.tar"], out)
    assert read_pack(out).manifest.data_visibility == "public"


def test_conflicting_visibility_writes_nothing(tmp_path):
    write_fixture(tmp_path / "public.tar", [question("a", "1")], visibility="public")
    write_fixture(tmp_path / "private.tar", [question("b", "1")], visibility="private")
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="mixed data_visibility"):
        assemble_packs([tmp_path / "public.tar", tmp_path / "private.tar"], out)
    assert not out.exists()


def test_all_unset_visibility_stays_unset(tmp_path):
    write_fixture(tmp_path / "a.tar", [question("a", "1")], visibility=None)
    write_fixture(tmp_path / "b.tar", [question("b", "1")], visibility=None)
    out = tmp_path / "out.tar"
    assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    assert read_pack(out).manifest.data_visibility is None


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"purpose": "development"}, {"purpose": "competition", "calibration": b'{"n":1}'}),
        ({"visibility": "public"}, {"visibility": "private"}),
    ],
)
def test_mixed_purpose_or_visibility_writes_nothing(tmp_path, left, right):
    write_fixture(tmp_path / "a.tar", [question("a", "1")], **left)
    write_fixture(tmp_path / "b.tar", [question("b", "1")], **right)
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="mixed"):
        assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    assert not out.exists()


def test_agreed_calibration_is_preserved(tmp_path):
    blob = b'{"note":"same"}'
    write_fixture(tmp_path / "a.tar", [question("a", "1")], calibration=blob)
    write_fixture(tmp_path / "b.tar", [question("b", "1")], calibration=blob)
    out = tmp_path / "out.tar"
    assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    pack = read_pack(out)
    assert pack.difficulty_calibration == blob
    assert pack.manifest.difficulty_calibration_sha256 == hashlib.sha256(blob).hexdigest()


def test_partial_calibration_writes_nothing(tmp_path):
    write_fixture(tmp_path / "a.tar", [question("a", "1")], calibration=b'{"n":1}')
    write_fixture(tmp_path / "b.tar", [question("b", "1")])
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="difficulty calibration"):
        assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    assert not out.exists()


def test_calibration_mismatch_writes_nothing(tmp_path):
    write_fixture(tmp_path / "a.tar", [question("a", "1")], calibration=b'{"n":1}')
    write_fixture(tmp_path / "b.tar", [question("b", "1")], calibration=b'{"n":2}')
    out = tmp_path / "out.tar"
    supplied = tmp_path / "difficulty-calibration.json"
    supplied.write_bytes(b'{"panel":1}')
    with pytest.raises(ValueError, match="difficulty calibration"):
        assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out, calibration=supplied)
    assert not out.exists()


def test_refuses_to_overwrite_and_needs_two_packs(tmp_path):
    write_fixture(tmp_path / "a.tar", [question("a", "1")])
    write_fixture(tmp_path / "b.tar", [question("b", "1")])
    out = tmp_path / "out.tar"
    out.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        assemble_packs([tmp_path / "a.tar", tmp_path / "b.tar"], out)
    assert out.read_bytes() == b"keep"
    with pytest.raises(ValueError, match="at least two"):
        assemble_packs([tmp_path / "a.tar"], tmp_path / "solo.tar")
    assert not (tmp_path / "solo.tar").exists()


def test_assemble_command_prints_counts_and_calibration(tmp_path, capsys):
    write_fixture(tmp_path / "zeta.tar", [question("zeta", "1")], releases=("zeta@1",))
    write_fixture(
        tmp_path / "alpha.tar",
        [question("alpha", "2"), question("alpha", "1")],
        releases=("alpha@1",),
    )
    calibration = tmp_path / "difficulty-calibration.json"
    calibration.write_bytes(b'{"panel":1}')
    out = tmp_path / "out.tar"
    assert (
        main(
            [
                "pack",
                "assemble",
                str(tmp_path / "zeta.tar"),
                str(tmp_path / "alpha.tar"),
                "--out",
                str(out),
                "--calibration",
                str(calibration),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    pack = read_pack(out)
    assert payload["path"] == str(out)
    assert payload["sha256"] == pack.sha256
    assert payload["schema"] == "ycore.pack.v2"
    assert payload["pack_id"] == pack.manifest.pack_id
    assert payload["records_sha256"] == pack.manifest.records_sha256
    assert payload["question_counts"] == {"alpha": 2, "zeta": 1}
    assert payload["question_count"] == 3
    assert pack.difficulty_calibration == b'{"panel":1}'
    assert payload["difficulty_calibration_sha256"] == hashlib.sha256(b'{"panel":1}').hexdigest()
    assert (
        main(
            [
                "pack",
                "assemble",
                str(tmp_path / "zeta.tar"),
                str(tmp_path / "zeta.tar"),
                "--out",
                str(tmp_path / "again.tar"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "yugos-evals pack assemble:" in captured.err
    assert "duplicate question_id" in captured.err
    assert not (tmp_path / "again.tar").exists()


def test_directory_assembles_three_packs(tmp_path, capsys):
    directory = tmp_path / "packs"
    directory.mkdir()
    # Created out of filename order. `.tar` and `.pack.tar` both count.
    write_fixture(directory / "c.pack.tar", [question("c", "1")])
    write_fixture(directory / "a.pack.tar", [question("a", "1"), question("a", "2")])
    write_fixture(directory / "b.tar", [question("b", "1")])
    (directory / "notes.txt").write_text("skip")
    nested = directory / "nested"
    nested.mkdir()
    write_fixture(nested / "hidden.pack.tar", [question("hidden", "1")])
    out = tmp_path / "out.tar"
    assert main(["pack", "assemble", str(directory), "--out", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["question_counts"] == {"a": 2, "b": 1, "c": 1}
    assert payload["question_count"] == 4
    assert [item.question_id for item in read_pack(out).questions] == ["a:1", "a:2", "b:1", "c:1"]


def test_directory_files_are_processed_in_filename_order(tmp_path):
    directory = tmp_path / "packs"
    directory.mkdir()
    write_fixture(directory / "c.pack.tar", [question("shared", "1")])
    write_fixture(directory / "a.pack.tar", [question("shared", "1")])
    write_fixture(directory / "b.pack.tar", [question("b", "1")])
    assert [path.name for path in resolve_pack_inputs([directory])] == [
        "a.pack.tar",
        "b.pack.tar",
        "c.pack.tar",
    ]
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match=r"c\.pack\.tar and .*a\.pack\.tar"):
        assemble_packs([directory], out)
    assert not out.exists()


def test_empty_directory_writes_nothing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a pack")
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="no .tar packs"):
        assemble_packs([empty], out)
    assert not out.exists()
    one = tmp_path / "one"
    one.mkdir()
    write_fixture(one / "only.pack.tar", [question("a", "1")])
    with pytest.raises(ValueError, match="at least two"):
        assemble_packs([one], out)
    assert not out.exists()


def test_files_and_a_directory_cannot_be_mixed(tmp_path):
    directory = tmp_path / "packs"
    directory.mkdir()
    write_fixture(directory / "a.pack.tar", [question("a", "1")])
    write_fixture(directory / "b.pack.tar", [question("b", "1")])
    write_fixture(tmp_path / "c.tar", [question("c", "1")])
    out = tmp_path / "out.tar"
    with pytest.raises(ValueError, match="single directory"):
        assemble_packs([directory, tmp_path / "c.tar"], out)
    assert not out.exists()


def test_assemble_help_documents_directory_inputs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["pack", "assemble", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "directory" in text and "*.tar" in text and "filename" in text
