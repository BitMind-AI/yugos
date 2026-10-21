"""`pack build` freezes a deterministic selection of a catalog task into a development pack."""

import json

import pytest
from inspect_ai._util import registry
from inspect_ai.dataset import Sample

from yugos.evaluation.packs import read_pack
from yugos.inspect_engine import restore_tasks
from yugos_evals.cli import main
from yugos_evals.local.packbuild import build_local_pack, select_sample_ids

CALLS = []


@pytest.fixture(autouse=True)
def tiny_upstream(monkeypatch):
    """Use the real native constructor with a small offline dataset."""
    import importlib

    from inspect_ai.dataset import MemoryDataset

    module = importlib.import_module("inspect_evals.mmlu_pro.mmlu_pro")
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kw: MemoryDataset(
            [
                Sample(id=f"q{i}", input=f"Question {i}?", choices=["yes", "no"], target="A")
                for i in range(10)
            ]
        ),
    )
    real = registry.registry_lookup

    def lookup(kind, name):
        factory = real(kind, name)
        if (kind, name) != ("task", "inspect_evals/mmlu_pro"):
            return factory

        def create(**kw):
            sandboxed = kw.pop("sandboxed", False)
            CALLS.append(kw)
            task = factory(**kw)
            if sandboxed:
                from inspect_ai.util import SandboxEnvironmentSpec

                task.sandbox = SandboxEnvironmentSpec("docker", "compose.yaml")
            return task

        return create

    monkeypatch.setattr(registry, "registry_lookup", lookup)
    CALLS.clear()


def test_selection_is_deterministic_and_keeps_dataset_order():
    ids = [f"q{i}" for i in range(10)]
    first = select_sample_ids(ids, limit=4, seed=7)
    assert first == select_sample_ids(ids, limit=4, seed=7)
    assert first == sorted(first, key=ids.index) and len(first) == 4
    assert select_sample_ids(ids, limit=None, seed=0) == ids


def test_built_pack_restores_with_catalog_settings(tmp_path):
    digest, manifest = build_local_pack("mmlu_pro", tmp_path / "pack.tar", limit=3, seed=1)
    # The catalog's settings reach the upstream factory.
    assert CALLS == [{"fewshot": 0, "shuffle": False}]
    pack = read_pack(tmp_path / "pack.tar")
    assert pack.sha256 == digest and pack.manifest == manifest
    assert manifest.purpose == "development" and manifest.data_visibility == "public"
    assert [q.question_id for q in pack.questions] == [
        f"mmlu_pro:{i}" for i in select_sample_ids([f"q{i}" for i in range(10)], limit=3, seed=1)
    ]
    assert {q.workload_class for q in pack.questions} == {"chat_qa"}
    [(restored, definition)] = restore_tasks(pack.questions)
    assert len(restored.dataset) == 3 and definition.scorer == "choice"


def test_unbuildable_benchmarks_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="no executable task"):
        build_local_pack("appworld", tmp_path / "a.tar")
    with pytest.raises(ValueError, match="sandbox"):
        build_local_pack("mmlu_pro", tmp_path / "b.tar", task_args={"sandboxed": True})
    assert not (tmp_path / "b.tar").exists()


def test_pack_build_command(tmp_path, capsys):
    out = tmp_path / "pack.tar"
    assert (
        main(["pack", "build", "--benchmark", "mmlu_pro", "--limit", "2", "--out", str(out)]) == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["purpose"] == "development"
    assert "not comparable to competition packs" in captured.err
    assert len(read_pack(out).questions) == 2
    assert main(["pack", "build", "--benchmark", "nope", "--out", str(tmp_path / "x")]) == 2
