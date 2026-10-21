"""Native scoring, private grader roles and shareable evidence contracts."""

import importlib

import anyio
import pytest
from inspect_fakes import freeze_fixture as freeze_task

pytest.importorskip("inspect_ai")

from inspect_ai import Task  # noqa: E402
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402
from inspect_ai.scorer import Score, mean, scorer  # noqa: E402
from inspect_ai.solver import generate  # noqa: E402
from inspect_fakes import ScriptedRouter, exact_question, pack_task, run  # noqa: E402

from yugos.inspect_engine import (  # noqa: E402
    UnboundGraderRole,
    active_seconds,
    bind_grader_roles,
    measure_log,
    redacted_log,
    require_grader_roles,
    restore_tasks,
    router_model,
    run_competition_eval,
)


@scorer(metrics=[mean()])
def reference_scorer(rubric="private-rubric", delay=0, fail=False):
    async def score(state, target):
        await anyio.sleep(delay)
        if fail:
            raise ValueError(f"Grader failed processing reference {target.text}")
        return Score(
            value=int(state.output.completion == target.text),
            explanation=f"Compared to {target.text} with {rubric}",
        )

    return score


def test_timing_excludes_scoring(tmp_path):
    question = exact_question("slow", "Paris")
    task = pack_task([question], name="timing")
    task.scorer = [reference_scorer(delay=0.4)]
    task.dataset[0].metadata["yugos_scorer"] = "reference_scorer"
    router = ScriptedRouter({question.prompt: "Paris"}, delay_s=0.05)
    log = run(task, router, tmp_path)
    [sample] = log.samples
    [row] = measure_log(log, router.receipts.receipts())
    assert 0.05 <= active_seconds(sample) < 0.4 <= sample.working_time
    assert row.measurement.outcome.latency_s == active_seconds(sample)
    assert row.active_ns == round(active_seconds(sample) * 1e9)


def test_grader_roles_can_never_fall_back_to_the_router(tmp_path):
    for bound in ({}, {"grader": "yugos-router/sub"}, {"grader": ["yugos-router/sub"]}):
        with pytest.raises(UnboundGraderRole):
            require_grader_roles(["grader"], bound, model_under_test="yugos-router/sub")
    router = ScriptedRouter({})
    model = router_model(router, submission_id="sub")
    task = pack_task([exact_question("q", "Paris")], name="pack")
    for roles in (None, {"grader": model}):
        with pytest.raises(UnboundGraderRole):
            run_competition_eval(
                task, model, log_dir=str(tmp_path), required_roles=["grader"], model_roles=roles
            )
    assert router.calls == []


def test_frozen_upstream_simpleqa_guards_roles_even_in_direct_inspect_eval(tmp_path, monkeypatch):
    # A lost role declaration would leak the reference to the submission as its own judge.
    module = importlib.import_module("inspect_evals.simpleqa.simpleqa")
    prompt, key = "Who wrote Hamlet?", "hidden-reference-Shakespeare"
    monkeypatch.setattr(
        module,
        "simpleqa_dataset",
        lambda *a, **kw: MemoryDataset([Sample(id="sq", input=prompt, target=key)]),
    )
    prepared = bind_grader_roles(module.simpleqa(scorer="original"), ["grader"])
    questions = freeze_task(
        prepared, source="simpleqa", workload_class="chat_qa", score_key="correct"
    )
    grader = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content(model="mockllm/model", content="A")],
        memoize=False,
    )
    [(task, definition)] = restore_tasks(questions, model_roles={"grader": grader})
    assert definition.required_roles == ("grader",)
    assert len(task.setup) == 1, "freezing an already guarded task must not duplicate guards"
    router = ScriptedRouter({prompt: "a guess"})
    model = router_model(router, submission_id="sub")
    for roles in ({"grader": model}, {"grader": "yugos-router/sub"}):
        with pytest.raises(UnboundGraderRole):
            run_competition_eval(task, model, log_dir=str(tmp_path), model_roles=roles)
    task.model_roles = None
    with pytest.raises(UnboundGraderRole):
        run_competition_eval(task, model, log_dir=str(tmp_path))
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none")
    assert "UnboundGraderRole" in log.samples[0].error.traceback
    assert router.calls == []
    log = run(task, router, tmp_path, model_roles={"grader": grader})
    [call] = router.calls
    assert key not in call.model_dump_json()
    [row] = measure_log(log, router.receipts.receipts())
    assert row.measurement.outcome.score == 1
    assert row.measurement.source_benchmark == "simpleqa"


def test_redacted_log_contains_no_references_or_grading_arguments(tmp_path):
    questions = [
        exact_question("q1", "hidden-reference-one"),
        exact_question("q2", "hidden-reference-two"),
    ]
    task = pack_task(questions, name="redact")
    task.scorer = [reference_scorer(rubric="private-rubric")]
    task.metadata = {"answer_key": "hidden-reference-header"}
    router = ScriptedRouter({q.prompt: q.prompt + " hidden-reference-echo" for q in questions})
    log = run(task, router, tmp_path)
    assert "hidden-reference" in log.model_dump_json() and "private-rubric" in log.model_dump_json()
    public = redacted_log(log, data_visibility="private")
    path = tmp_path / "public.eval"
    write_eval_log(public, str(path))
    for view in (public, read_eval_log(str(path))):
        text = view.model_dump_json()
        assert "hidden-reference" not in text and "private-rubric" not in text
        assert questions[0].prompt not in text
        assert [s.scores["reference_scorer"].value for s in view.samples] == [0, 0]
        assert view.results.scores[0].metrics == log.results.scores[0].metrics
    assert questions[0].prompt in redacted_log(log, data_visibility="public").model_dump_json()
    assert "hidden-reference" in log.model_dump_json(), "redaction must not mutate the source"


def test_measurements_require_workload_labels_and_select_a_scalar_from_dict_scores(tmp_path):
    @scorer(metrics=[mean()])
    def dict_scorer():
        async def score(state, target):
            return Score(value={"strict": True, "partial": 0.5})

        return score

    task = Task(
        dataset=[Sample(id="q", input="Write a haiku.", target="")],
        solver=generate(),
        scorer=dict_scorer(),
    )
    router = ScriptedRouter({"Write a haiku.": "a haiku"})
    log = run(task, router, tmp_path)
    with pytest.raises(ValueError, match="workload class"):
        measure_log(log, router.receipts.receipts())
    log.samples[0].metadata = {"workload_class": "chat_qa", "source_benchmark": "fixture"}
    with pytest.raises(ValueError, match="score key"):
        measure_log(log, router.receipts.receipts())
    log.samples[0].metadata["yugos_score_key"] = "strict"
    [row] = measure_log(log, router.receipts.receipts())
    assert row.measurement.outcome.score == 1
    assert row.measurement.source_benchmark == "fixture"


def test_redacted_log_drops_references_from_grader_errors(tmp_path):
    question = exact_question("q", "hidden-reference-judged")
    task = pack_task([question], name="judge-error")
    task.scorer = [reference_scorer(fail=True)]
    router = ScriptedRouter({question.prompt: "a guess"})
    [log] = inspect_eval(
        task,
        model=router_model(router, submission_id="sub"),
        log_dir=str(tmp_path),
        display="none",
        fail_on_error=True,
    )
    assert log.status == "error"
    assert "hidden-reference" in log.samples[0].error.message
    assert "hidden-reference" in log.error.message
    public = redacted_log(log, data_visibility="private")
    path = tmp_path / "public-error.eval"
    write_eval_log(public, str(path))
    for view in (public, read_eval_log(str(path))):
        assert "hidden-reference" not in view.model_dump_json()
        assert view.samples[0].error.message == view.error.message == "ValueError"
        assert view.samples[0].error.traceback == view.error.traceback == ""


def test_pinned_roles_need_their_model_and_settings(tmp_path):
    # Catches a judge running at provider defaults, or another model, when the pack pins it.
    from inspect_ai.model import GenerateConfig

    from yugos.inspect_engine import (
        RolePin,
        RolePinMismatch,
        read_role_pins,
        require_role_pins,
        role_pins_metadata,
    )

    pin = RolePin(model="mockllm/model", params={"temperature": 0, "max_tokens": 64})
    metadata = role_pins_metadata({"grader": pin})
    assert read_role_pins(metadata) == {"grader": pin}
    for bad in ({"temprature": 0}, {"temperature": None}, {"temperature": "hot"}):
        with pytest.raises(ValueError):
            RolePin(model="a/b", params=bad)

    def grader(**config):
        return get_model("mockllm/model", config=GenerateConfig(**config), memoize=False)

    pinned = grader(temperature=0.0, max_tokens=64, seed=3)
    require_role_pins({"grader": pin}, {"grader": pinned})
    # A proxy is identified by the model it forwards to.
    proxy = router_model(ScriptedRouter({}), submission_id="mockllm/model", config=pin.config())
    require_role_pins({"grader": pin}, {"grader": proxy})
    for bound in (
        grader(),
        grader(temperature=0.5, max_tokens=64),
        "mockllm/model",
        get_model("mockllm/other", config=pin.config(), memoize=False),
    ):
        with pytest.raises(RolePinMismatch):
            require_role_pins({"grader": pin}, {"grader": bound})
    with pytest.raises(UnboundGraderRole):
        require_role_pins({"grader": pin}, {})

    question = exact_question("q", "Paris")
    router = ScriptedRouter({question.prompt: "Paris"})
    model = router_model(router, submission_id="sub")
    task = pack_task([question], name="pinned")
    task.metadata = {**(task.metadata or {}), **metadata}
    task = bind_grader_roles(task, [])
    with pytest.raises(RolePinMismatch):
        run_competition_eval(task, model, log_dir=str(tmp_path), model_roles={"grader": grader()})
    # Evaluated directly, the setup guard checks the settings in every sample.
    [log] = inspect_eval(
        task, model=model, model_roles={"grader": grader()}, log_dir=str(tmp_path), display="none"
    )
    assert "RolePinMismatch" in log.samples[0].error.traceback
    assert router.calls == []
    log = run(task, router, tmp_path, model_roles={"grader": pinned})
    assert log.status == "success" and len(router.calls) == 1


def test_frozen_pins_require_the_role_and_survive_restoring(tmp_path):
    from inspect_ai.scorer import exact

    from yugos.inspect_engine import RolePin, read_role_pins, role_pins_metadata

    task = Task(
        dataset=[Sample(id="q", input="Capital of France?", target="Paris")],
        solver=generate(),
        scorer=exact(),
        metadata=role_pins_metadata({"user": RolePin(model="mockllm/model", params={"seed": 1})}),
    )
    [question] = freeze_task(task, source="pinned", workload_class="chat_qa")
    assert question.inspect_task["required_roles"] == ["user"]
    with pytest.raises(ValueError, match="explicitly bound"):
        restore_tasks([question])
    [(restored, _)] = restore_tasks([question], model_roles={"user": "mockllm/model"})
    assert read_role_pins(restored.metadata)["user"].params == {"seed": 1}
