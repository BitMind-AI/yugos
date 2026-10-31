"""End-to-end local runs: a real bundle in a real worker subprocess, a scripted provider."""

import json
from decimal import Decimal

import pytest
from evals_fakes import freeze_fixture as freeze_task
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.log import read_eval_log
from local_fixtures import (
    CALL_COST,
    CARD,
    ScriptedProvider,
    graded_questions,
    make_bundle,
    make_pack,
)

from yugos.inspect_engine import JsonlReceipts
from yugos_evals.local import providers
from yugos_evals.local.runner import LocalRunError, RunOptions, parse_model_choice, run_local
from yugos_evals.local.world import WorldCard, load_world_card

ENV = {"OPENAI_API_KEY": "sk-local-test", "PATH": "/usr/bin:/bin"}


@pytest.fixture
def provider(monkeypatch):
    fake = ScriptedProvider()
    monkeypatch.setattr(providers, "get_model", fake)
    return fake


def options(tmp_path, **changes):
    fields = {
        "pack": tmp_path / "pack.tar",
        "bundle": tmp_path / "router.zip",
        "max_cost_usd": 1.0,
        "log_dir": tmp_path / "run",
        "world_card": str(CARD),
        "display": "none",
    }
    return RunOptions(**{**fields, **changes})


def run(tmp_path, *, questions=None, **changes):
    if not (tmp_path / "pack.tar").exists():
        make_pack(tmp_path / "pack.tar", questions)
    if not (tmp_path / "router.zip").exists():
        make_bundle(tmp_path / "router")
    return run_local(
        options(tmp_path, **changes), env=ENV, card_cache=tmp_path / "cache", docker_available=False
    )


def test_a_missing_provider_sdk_stops_the_run_before_anything_is_written(
    tmp_path, provider, monkeypatch
):
    def missing(feature):
        raise PrerequisiteError(f"{feature} requires optional dependencies")

    monkeypatch.setitem(providers._SDK_CHECKS, "openai", (missing, "openai"))
    with pytest.raises(LocalRunError, match="uv pip install -U openai"):
        run(tmp_path)
    assert not (tmp_path / "run").exists() and not provider.models


def test_run_measures_quality_spend_time_and_receipts(tmp_path, provider):
    result = run(tmp_path)
    assert result.failure is None, result.failure
    saved = json.loads((result.path / "measurements.json").read_text())
    assert saved["status"] == "complete" and saved["comparable_to_hosted"] is False
    summary = saved["summary"]
    # One wrong answer in each category; the crashed question is the router's failure.
    assert Decimal(summary["chat_qa"]["quality"]) == Decimal("0.5")
    assert Decimal(summary["math_code"]["quality"]) == Decimal("0.5")
    assert all(row["complete"] for row in summary.values())
    rows = {row["question_id"]: row for row in saved["questions"]}
    assert rows["trivia:t2"]["failure_owner"] == "submission"
    assert Decimal(rows["trivia:t2"]["cost_usd"]) == 0

    receipts = JsonlReceipts(result.path / "receipts.jsonl").receipts()
    assert len(receipts) == 3
    assert all(r.cost_usd == Decimal(str(CALL_COST)) for r in receipts)
    assert {r.sample_uuid for r in receipts} == {
        rows[q]["sample_uuid"] for q in ("trivia:t1", "arithmetic:m1", "arithmetic:m2")
    }
    assert Decimal(summary["math_code"]["cost_usd"]) == 2 * Decimal(str(CALL_COST))
    assert saved["spent_usd"]["router"] == pytest.approx(3 * CALL_COST)

    # Inspect logs are real, readable logs of the restored tasks.
    logs = [read_eval_log(path) for path in saved["logs"]]
    assert sorted(len(log.samples) for log in logs) == [2, 2]
    # The provider saw the miner's key and the world-card model through Inspect.
    assert {m["name"] for m in provider.models} == {"openai/gpt-test"}
    assert {m["api_key"] for m in provider.models} == {"sk-local-test"}
    assert (result.path / "worker.log").exists()


def test_cost_cap_stops_the_run_cleanly(tmp_path, provider):
    # A one-cent reservation fits; the known bill leaves less than another quote.
    result = run(tmp_path, max_cost_usd=0.012)
    assert result.failure and "budget" in result.failure
    saved = json.loads((result.path / "measurements.json").read_text())
    assert saved["status"] == "failed"
    assert saved["spent_usd"]["router"] <= 0.012
    assert not all(row["complete"] for row in saved["summary"].values())
    assert len(JsonlReceipts(result.path / "receipts.jsonl").receipts()) == 1


def test_grader_roles_must_be_bound_before_anything_runs(tmp_path, provider):
    with pytest.raises(LocalRunError, match="grader"):
        run(tmp_path, questions=graded_questions())
    assert not (tmp_path / "run").exists() and provider.models == []


def test_bound_grader_is_metered_apart_from_router_spend(tmp_path, provider):
    result = run(
        tmp_path,
        questions=graded_questions(),
        service_models={"grader": parse_model_choice("openai/gpt-judge")},
    )
    assert result.failure is None, result.failure
    saved = json.loads((result.path / "measurements.json").read_text())
    assert Decimal(saved["summary"]["chat_qa"]["quality"]) == 1
    [grader] = JsonlReceipts(result.path / "service-receipts.jsonl").receipts()
    assert grader.model == "gpt-judge"
    # 1000 input tokens at $0.5/M plus 500 output tokens at $2/M.
    assert grader.cost_usd == Decimal("0.0015")
    assert Decimal(saved["summary"]["chat_qa"]["cost_usd"]) == Decimal(str(CALL_COST))
    assert saved["spent_usd"]["services"] == pytest.approx(0.0015)


@pytest.mark.parametrize("with_grader", [False, True])
def test_distinct_sample_sandboxes_restore_before_local_worker_start(
    tmp_path, provider, monkeypatch, with_grader
):
    from local_fixtures import default_questions

    from yugos_evals.local import runner

    original = graded_questions()[0] if with_grader else default_questions()[0]
    questions = []
    for index in (1, 2):
        q = original.model_copy(deep=True, update={"question_id": f"sandbox:{index}"})
        spec = {
            "type": "modal",
            "config": {
                "services": {
                    "default": {"image": f"fixture/image-{index}@sha256:" + str(index) * 64}
                },
                "x-modal": {"block_network": True},
            },
        }
        q.inspect_task["sandbox"] = spec
        q.inspect_sample["sandbox"] = spec
        questions.append(q)

    def inspect_restored(tasks, *, docker_available):
        [(task, definition)] = tasks
        assert len(task.dataset) == 2 and task.sandbox is None
        assert len({s.sandbox.config.services["default"].image for s in task.dataset}) == 2
        if with_grader:
            assert definition.required_roles == ("grader",) and task.model_roles["grader"]
        raise LocalRunError("fixture reached sandbox preflight")

    monkeypatch.setattr(runner, "use_local_sandboxes", inspect_restored)
    roles = {"grader": parse_model_choice("openai/gpt-judge")} if with_grader else {}
    with pytest.raises(LocalRunError, match="fixture reached sandbox preflight"):
        run(tmp_path, questions=questions, service_models=roles)
    assert not (tmp_path / "run").exists() and provider.models == []


def test_unused_role_bindings_are_refused(tmp_path, provider):
    with pytest.raises(LocalRunError, match="does not use"):
        run(tmp_path, service_models={"judge": parse_model_choice("openai/gpt-judge")})
    assert provider.models == []


def test_a_pinned_world_card_is_required_and_read_from_the_cache(tmp_path, provider):
    card = WorldCard(json.loads(CARD.read_text()))
    make_pack(tmp_path / "pack.tar", world_card_sha256="0" * 64)
    with pytest.raises(ValueError, match="pins world card"):
        run(tmp_path)
    load_world_card(str(CARD), cache_dir=tmp_path / "cache")
    make_pack(tmp_path / "pinned.tar", world_card_sha256=card.sha256)
    result = run(tmp_path, pack=tmp_path / "pinned.tar", world_card=None)
    assert result.summary["world_card_sha256"] == card.sha256


def test_missing_keys_and_unpriced_models_fail_before_spending(tmp_path, provider):
    make_pack(tmp_path / "pack.tar")
    make_bundle(tmp_path / "router")
    with pytest.raises(LocalRunError, match="OPENAI_API_KEY"):
        run_local(options(tmp_path), env={}, card_cache=tmp_path / "cache")
    make_bundle(
        tmp_path / "unpriced", model={"provider_id": "openai", "model_id": "gpt-unreviewed"}
    )
    with pytest.raises(LocalRunError, match="outside the world card"):
        run_local(
            options(tmp_path, bundle=tmp_path / "unpriced.zip"),
            env=ENV,
            card_cache=tmp_path / "cache",
        )
    assert provider.models == []


def test_run_command_prints_the_category_summary(tmp_path, provider, monkeypatch, capsys):
    from yugos_evals.cli import main

    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-test")
    monkeypatch.setenv("YUGOS_EVALS_CACHE", str(tmp_path / "cache"))
    make_pack(tmp_path / "pack.tar")
    make_bundle(tmp_path / "router")
    capsys.readouterr()
    argv = ["run", "--pack", str(tmp_path / "pack.tar"), "--bundle", str(tmp_path / "router.zip")]
    argv += ["--world-card", str(CARD), "--max-cost", "1", "--log-dir", str(tmp_path / "run")]
    assert main([*argv, "--display", "none"]) == 0
    out = capsys.readouterr().out
    assert "sk-local-test" not in out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.strip()}
    assert rows["chat_qa"][:4] == ["chat_qa", "2", "0.500", "0.001500"]
    assert rows["math_code"][:4] == ["math_code", "2", "0.500", "0.003000"]
    assert "not comparable to hosted scores" in out


def pinned_questions(**params):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import model_graded_qa
    from inspect_ai.solver import generate

    from yugos.inspect_engine import RolePin, role_pins_metadata

    pin = RolePin(model="openai/gpt-judge", params=params)
    task = Task(
        dataset=[Sample(id="g1", input="Explain why the sky is blue.", target="Rayleigh")],
        solver=generate(),
        scorer=model_graded_qa(),
        name="graded",
        metadata=role_pins_metadata({"grader": pin}),
    )
    return freeze_task(task, source="graded", workload_class="chat_qa")


def test_pinned_role_settings_reach_every_service_call(tmp_path, provider):
    # Catches a pinned grader silently running at provider-default sampling locally.
    result = run(
        tmp_path,
        questions=pinned_questions(temperature=0, max_tokens=800),
        service_models={"grader": parse_model_choice("openai/gpt-judge")},
    )
    assert result.failure is None, result.failure
    [grader] = [m for m in provider.models if m["name"] == "openai/gpt-judge"]
    assert grader["config"].temperature == 0 and grader["config"].max_tokens == 800


def test_task_pin_takes_precedence_over_role_defaults(tmp_path, provider):
    result = run(
        tmp_path,
        questions=pinned_questions(temperature=0),
        service_models={"grader": parse_model_choice("openai/gpt-other")},
    )
    assert result.failure is None
    assert "openai/gpt-judge" in {m["name"] for m in provider.models}
    assert "openai/gpt-other" not in {m["name"] for m in provider.models}


def effort_only_card(tmp_path):
    card = json.loads(CARD.read_text())
    provider = next(p for p in card["providers"] if p["provider_id"] == "openai")
    for model in provider["models"]:
        if model["model_id"] == "gpt-judge":
            model["effort_levels"] = ["low"]
    path = tmp_path / "effort-only-card.json"
    path.write_text(json.dumps(card))
    return str(path)


@pytest.mark.parametrize("default", [None, "openai/gpt-judge@low"])
def test_pinned_effort_is_admitted_without_a_no_effort_entry(tmp_path, provider, default):
    result = run(
        tmp_path,
        questions=pinned_questions(reasoning_effort="low"),
        world_card=effort_only_card(tmp_path),
        service_models={} if default is None else {"grader": parse_model_choice(default)},
    )
    assert result.failure is None, result.failure
    [grader] = [m for m in provider.models if m["name"] == "openai/gpt-judge"]
    assert grader["config"].reasoning_effort == "low"
    [receipt] = JsonlReceipts(result.path / "service-receipts.jsonl").receipts()
    assert receipt.reasoning_effort == "low"


@pytest.mark.parametrize("effort", [None, "high"])
def test_unadmitted_effective_effort_fails_before_spending(tmp_path, provider, effort):
    with pytest.raises(LocalRunError, match="models without world-card prices and limits"):
        run(
            tmp_path,
            questions=pinned_questions(**({} if effort is None else {"reasoning_effort": effort})),
            world_card=effort_only_card(tmp_path),
        )
    assert not (tmp_path / "run").exists() and provider.models == []


def test_local_tasks_keep_distinct_settings_for_the_same_grader(tmp_path, provider):
    first = pinned_questions(temperature=0)[0]
    second = pinned_questions(temperature=0.7)[0].model_copy(update={"question_id": "graded:g2"})
    result = run(tmp_path, questions=[first, second])
    assert result.failure is None, result.failure
    graders = [m for m in provider.models if m["name"] == "openai/gpt-judge"]
    assert [m["config"].temperature for m in graders] == [0, 0.7]


def test_unappliable_role_settings_fail_before_spending(tmp_path, provider):
    with pytest.raises(LocalRunError, match="cannot override"):
        run(tmp_path, questions=pinned_questions(extra_headers={"Authorization": "forbidden"}))
    assert not (tmp_path / "run").exists() and provider.models == []
