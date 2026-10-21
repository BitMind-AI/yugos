"""Packed Samples run with installed task builders, without source reloads."""

import json
from importlib.metadata import version

import pytest
from inspect_fakes import freeze_fixture as freeze_task

pytest.importorskip("inspect_ai")
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import exact, match
from inspect_ai.solver import generate
from inspect_evals.mmlu_pro import mmlu_pro as factory
from inspect_fakes import ScriptedRouter, run

from yugos.evaluation import Question
from yugos.inspect_engine import measure_log
from yugos.inspect_engine.frozen import restore_tasks


def test_upstream_choice_task_config_roundtrips_and_scores_without_native_grading(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("inspect_evals.mmlu_pro.mmlu_pro")
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kw: MemoryDataset(
            [
                module.record_to_sample(
                    {
                        "question_id": 17,
                        "question": "Which is even?",
                        "options": ["three", "four"],
                        "answer": "B",
                        "cot_content": "",
                        "category": "math",
                    }
                )
            ]
        ),
    )
    task = factory(fewshot=0, shuffle=False)
    frozen = freeze_task(task, source="mmlu", workload_class="chat_qa")
    questions = [Question.model_validate_json(q.model_dump_json()) for q in frozen]
    monkeypatch.setattr(module, "hf_dataset", lambda **kw: pytest.fail("dataset reloaded"))
    [(restored, definition)] = restore_tasks(questions)

    class Router(ScriptedRouter):
        async def __call__(self, call):
            prompt = call.request.messages[-1].text
            assert "A) three" in prompt and "B) four" in prompt
            self.replies = {prompt: "ANSWER: B"}
            return await super().__call__(call)

    router = Router({})
    log = run(restored, router, tmp_path)
    [row] = measure_log(log, router.receipts.receipts())
    assert row.measurement.outcome.score == 1
    assert row.measurement.question_id == questions[0].question_id
    assert row.grade.grader == "choice"
    assert definition.scorer == "choice"
    assert "grading" not in json.dumps(router.calls[0].request.model_dump())


def test_selection_and_mixed_task_groups_preserve_identity():
    a = Task(
        dataset=[Sample(id="a", input="one", target="1"), Sample(id="b", input="two", target="2")],
        solver=generate(),
        scorer=exact(),
        name="fixture",
    )
    b = Task(
        dataset=[Sample(id="a", input="number", target="2")],
        solver=generate(),
        scorer=match(numeric=True),
        name="fixture",
    )
    questions = (
        *freeze_task(a, source="first", workload_class="chat_qa", sample_ids=["b"]),
        *freeze_task(a, source="second", workload_class="chat_qa", sample_ids=["a"]),
        *freeze_task(b, source="first", workload_class="math_code"),
    )
    groups = restore_tasks(questions)
    assert [[s.id for s in task.dataset] for task, _ in groups] == [
        ["first:b", "second:a"],
        ["first:a"],
    ]
    assert [s.target for task, _ in groups for s in task.dataset] == ["2", "1", "2"]
    per_task = restore_tasks(
        questions,
        model_roles=lambda definition: {"grader": f"mockllm/{definition.scorer}"},
    )
    assert all(
        str(task.model_roles["grader"]) == f"mockllm/{definition.scorer}"
        for task, definition in per_task
    )
    assert per_task[0][0].model_roles != per_task[1][0].model_roles
    with pytest.raises(ValueError, match="absent"):
        freeze_task(a, source="first", workload_class="chat_qa", sample_ids=["missing"])


def test_unportable_task_callbacks_and_missing_roles_are_not_silently_dropped():
    async def cleanup(state):
        pass

    task = Task(dataset=[Sample(id="q", input="q")], scorer=exact(), cleanup=cleanup)
    questions = freeze_task(task, source="s", workload_class="chat_qa")
    [(restored, _)] = restore_tasks(questions)
    assert restored.cleanup is cleanup
    task.cleanup = None
    questions = freeze_task(task, source="s", workload_class="chat_qa", required_roles=["grader"])
    with pytest.raises(ValueError, match="service model roles"):
        restore_tasks(questions)


def test_multiple_scores_require_explicit_selection_and_pins_are_enforced():
    task = Task(dataset=[Sample(id="q", input="q", target="a")], scorer=[exact(), match()])
    with pytest.raises(ValueError, match="authoritative scorer"):
        freeze_task(task, source="s", workload_class="chat_qa")
    [question] = freeze_task(task, source="s", workload_class="chat_qa", scorer="exact")
    question.inspect_task["packages"]["inspect-ai"] = "wrong"
    with pytest.raises(ValueError, match="prepared task requires"):
        restore_tasks([question])


def test_sample_assets_are_frozen_and_revalidated_before_execution(tmp_path):
    from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec

    asset = tmp_path / "input.txt"
    asset.write_bytes(b"selected data")
    sandbox = SandboxEnvironmentSpec(
        "modal",
        ComposeConfig.model_validate(
            {
                "services": {"default": {"image": "fixture/image@sha256:" + "a" * 64}},
                "x-modal": {"timeout": 60, "block_network": True},
            }
        ),
    )
    task = Task(
        dataset=[Sample(id="q", input="q", target="a", files={"input.txt": str(asset)})],
        scorer=exact(),
        sandbox=sandbox,
    )
    [q] = freeze_task(task, source="s", workload_class="math_code")
    asset.unlink()
    [(restored, _)] = restore_tasks([q])
    assert restored.dataset[0].files["input.txt"].startswith("data:")
    q.inspect_task["sandbox"]["config"]["services"]["default"]["image"] = "fixture:latest"
    with pytest.raises(ValueError, match="digest-pinned"):
        restore_tasks([q])


def test_display_fields_cannot_diverge_from_the_executable_sample():
    [q] = freeze_task(
        Task(dataset=[Sample(id="q", input="q", target="a")], scorer=exact()),
        source="s",
        workload_class="chat_qa",
    )
    with pytest.raises(ValueError, match="differs"):
        restore_tasks([q.model_copy(update={"expected_answer": "changed"})])


def test_ifeval_positional_arguments_survive_json_without_reimplementing_checks(
    tmp_path, monkeypatch
):
    import importlib

    checker = pytest.importorskip("instruction_following_eval.evaluation")
    # This keyword instruction uses no tokenizer; acquisition is outside this test.
    monkeypatch.setattr(checker, "ensure_nltk_resource", lambda: None)
    module = importlib.import_module("inspect_evals.ifeval.ifeval")
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kw: MemoryDataset(
            [
                module.record_to_sample(
                    {
                        "key": index,
                        "prompt": prompt,
                        "instruction_id_list": ["keywords:existence"],
                        "kwargs": [{"keywords": ["four"]}],
                    }
                )
                for index, prompt in enumerate(["say four", "omit four"])
            ]
        ),
    )
    frozen = freeze_task(
        module.ifeval(), source="ifeval", workload_class="chat_qa", score_key="prompt_level_strict"
    )
    questions = [Question.model_validate_json(q.model_dump_json()) for q in frozen]
    [(task, _)] = restore_tasks(questions)
    router = ScriptedRouter({"say four": "four", "omit four": "other"})
    rows = measure_log(run(task, router, tmp_path), router.receipts.receipts())
    assert [r.measurement.outcome.score for r in rows] == [1, 0]


@pytest.mark.parametrize(
    "asset", ["https://example.invalid/data", "s3://bucket/data", "./input.txt"]
)
def test_restoration_refuses_unprepared_assets_before_inspect_can_fetch_them(asset):
    [question] = freeze_task(
        Task(
            dataset=[Sample(id="q", input="q", target="a", files={"input": "inline text"})],
            scorer=exact(),
        ),
        source="s",
        workload_class="chat_qa",
    )
    question.inspect_sample["files"]["input"] = asset
    with pytest.raises(ValueError, match="frozen inline"):
        restore_tasks([question])


def test_preparation_freezes_literal_setup_and_refuses_remote_files():
    from yugos.inspect_engine.assets import inline_file

    assert inline_file("#!/bin/sh\necho ready").startswith("data:")
    with pytest.raises(ValueError, match="acquired before"):
        inline_file("https://example.invalid/data")


def _identity_task(target="4", *, scorer=match):
    return Task(
        dataset=[Sample(id=i, input=f"2+{i}?", target=target) for i in ("a", "b")],
        solver=generate(),
        scorer=scorer(),
    )


def test_evaluation_identity_commits_to_task_configs_and_pins_not_samples_or_sources(monkeypatch):
    from yugos.inspect_engine import evaluation_identity
    from yugos.inspect_engine.frozen import TaskConfig

    questions = freeze_task(_identity_task(), source="toy", workload_class="chat_qa")
    identity = evaluation_identity(questions)
    assert identity["packages"] == {
        "inspect-ai": version("inspect-ai"),
        "inspect-evals": version("inspect-evals"),
        "yugos": version("yugos"),
        "pytest": version("pytest"),
    }
    assert set(identity) == {"kind", "packages", "recipes_sha256"}
    # Questions share one task config: order, repetition and Sample selection do not matter.
    assert evaluation_identity([questions[1], questions[0], questions[1]]) == identity
    assert evaluation_identity(questions[:1]) == identity
    task_config = TaskConfig.model_validate(questions[0].inspect_task)
    assert evaluation_identity([task_config]) == identity
    assert evaluation_identity([questions[0].inspect_task]) == identity
    # Unrelated edits to installed source files do not change the task config identity.
    from pathlib import Path

    read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: read(path) + b"\n")
    try:
        assert evaluation_identity(questions) == identity
    finally:
        monkeypatch.undo()

    # A different scorer, setting or pin is a different evaluation.
    rescored = freeze_task(_identity_task(scorer=exact), source="toy", workload_class="chat_qa")
    assert evaluation_identity(rescored)["recipes_sha256"] != identity["recipes_sha256"]
    limited = task_config.model_copy(update={"message_limit": 3})
    assert evaluation_identity([limited])["recipes_sha256"] != identity["recipes_sha256"]
    repinned = task_config.model_copy(update={"packages": {**task_config.packages, "inspect-evals": "0.0.1"}})
    changed = evaluation_identity([repinned])
    assert changed["packages"]["inspect-evals"] == "0.0.1"
    assert changed["recipes_sha256"] != identity["recipes_sha256"]
    # A mixed pack covers every task config; conflicting pins cannot run together.
    both = evaluation_identity([*questions, *rescored])
    assert both["recipes_sha256"] not in {
        identity["recipes_sha256"],
        evaluation_identity(rescored)["recipes_sha256"],
    }
    with pytest.raises(ValueError, match="different versions"):
        evaluation_identity([task_config, repinned])
    with pytest.raises(ValueError, match="at least one"):
        evaluation_identity([])
    invalid = questions[0].model_copy(update={"inspect_task": None})
    with pytest.raises(ValueError, match="TaskConfig"):
        evaluation_identity([invalid])
