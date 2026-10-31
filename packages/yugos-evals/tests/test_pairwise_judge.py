"""`pairwise_judge`: Arena-Hard v2 parity with the legacy policy, grader-role isolation, loaders."""

import hashlib
import itertools
import json
import math
import os
from functools import partial
from pathlib import Path

import pytest
from evals_fakes import ScriptedRouter, assert_portable_task_config, judge, run
from inspect_ai import eval as inspect_eval
from local_fixtures import CARD, ScriptedProvider, make_bundle, make_pack

from yugos.inspect_engine import (
    RolePinMismatch,
    UnboundGraderRole,
    category_measurements_from_log,
    restore_tasks,
    router_model,
    run_competition_eval,
)
from yugos_evals.local import providers
from yugos_evals.local.runner import LocalRunError, RunOptions, parse_model_choice, run_local
from yugos_evals.prepared import prepare_task as freeze_task
from yugos_evals.scorers import arenahard_upstream as upstream
from yugos_evals.scorers.pairwise import judge_messages, parse_verdict, preference
from yugos_evals.sources import _fetch, arenahard_v2
from yugos_evals.tasks import pairwise_judge

FIXTURES = Path(__file__).parent / "fixtures"
PARSER = [
    json.loads(line) for line in (FIXTURES / "arenahard_parser.jsonl").read_text().splitlines()
]
LABELS = ["A>B", "A>>B", "A=B", "A<<B", "A<B", "B>A", "B>>A", "B=A", "B<<A", "B<A"]


# The legacy Arena-Hard v2 policy's prompts, messages and aggregates, recorded from
# `yugos.grading.arenahard` (yugos cb20e3c) before it was deleted.
LEGACY = json.loads((FIXTURES / "arenahard_legacy_parity.json").read_text())


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def test_vendored_prompts_and_rules_equal_the_legacy_copy():
    assert {
        "SYSTEM": _sha256(upstream.SYSTEM),
        "PROMPT_TEMPLATE": _sha256(upstream.PROMPT_TEMPLATE),
        "PATTERNS": _sha256(upstream.PATTERNS),
        "JUDGE_SETTINGS": _sha256(upstream.JUDGE_SETTINGS),
        "BASELINES": _sha256(arenahard_v2.BASELINES),
    } == LEGACY["vendored_sha256"]


@pytest.mark.parametrize("category", arenahard_v2.CATEGORIES)
@pytest.mark.parametrize("round", [1, 2])
def test_judge_messages_match_the_legacy_policy(category, round):
    case = LEGACY["judge_messages"]
    ours = judge_messages(category, case["prompt"], case["baseline"], case["candidate"], round)
    assert ours == case["expected"][f"{category}/{round}"]


@pytest.mark.parametrize("case", PARSER, ids=lambda case: case["text"])
def test_verdict_parsing_matches_the_legacy_parser(case):
    ours = parse_verdict(case["text"], stop_reason="stop")
    assert (ours["label"], ours["valid"]) == (case["label"], case["valid"])


def test_preference_matches_the_legacy_aggregate():
    pairs = list(itertools.product(LABELS + ["[[AA]]", "none"], repeat=2))
    texts = [[f"[[{label}]]" if "[" not in label else label for label in pair] for pair in pairs]
    assert [case["texts"] for case in LEGACY["aggregates"]] == texts
    for case in LEGACY["aggregates"]:
        ours = preference([parse_verdict(t, stop_reason="stop") for t in case["texts"]])
        if case["verdict"] == "needs_review":
            assert not ours.settled and ours.code == "needs_review", case
        else:
            assert ours.settled and ours.score == case["score"], case
            assert ours.details["weighted_battles"] == case["weighted_battles"], case


def test_incomplete_judgments_never_count_as_votes():
    truncated = parse_verdict("[[B>A]]", stop_reason="max_tokens")
    assert not truncated["valid"]
    assert not preference([truncated, parse_verdict("[[A>B]]", stop_reason="stop")]).settled


# --- Loader, freezing and the grader role ---------------------------------------------------

QUESTIONS = [
    {"uid": "q1", "category": "hard_prompt", "subcategory": "code", "prompt": "Explain recursion."},
    {
        "uid": "q2",
        "category": "creative_writing",
        "subcategory": "poem",
        "prompt": "Write a haiku.",
    },
]
BASELINE_ANSWERS = {
    "hard_prompt": [{"uid": "q1", "messages": [{"content": {"answer": "BASELINE-RECURSION"}}]}],
    "creative_writing": [{"uid": "q2", "messages": [{"content": {"answer": "BASELINE-HAIKU"}}]}],
}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return _fetch.sha256_file(path)


def _arena_fixture(tmp_path, monkeypatch, questions=QUESTIONS):
    monkeypatch.setattr(
        arenahard_v2, "QUESTIONS_SHA256", _write(tmp_path / "question.jsonl", questions)
    )
    baselines = {}
    for category, rows in BASELINE_ANSWERS.items():
        model = arenahard_v2.BASELINES[category][0]
        baselines[category] = (model, _write(tmp_path / f"model_answer/{model}.jsonl", rows))
    monkeypatch.setattr(arenahard_v2, "BASELINES", baselines)
    return tmp_path


def test_loader_binds_each_question_to_its_category_baseline(tmp_path, monkeypatch):
    data = _arena_fixture(tmp_path, monkeypatch)
    loaded = arenahard_v2.load_arenahard_v2(data_dir=data)
    assert [(s.id, s.target, s.metadata["category"]) for s in loaded.samples] == [
        ("arenahard-v2-q1", "BASELINE-RECURSION", "hard_prompt"),
        ("arenahard-v2-q2", "BASELINE-HAIKU", "creative_writing"),
    ]
    [only] = arenahard_v2.load_arenahard_v2(["creative_writing"], data_dir=data).samples
    assert only.id == "arenahard-v2-q2"
    monkeypatch.setattr(arenahard_v2, "QUESTIONS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        arenahard_v2.load_arenahard_v2(data_dir=data)


def test_a_question_without_its_baseline_answer_fails_loading(tmp_path, monkeypatch):
    extra = QUESTIONS + [{"uid": "q3", "category": "hard_prompt", "prompt": "Unanswered."}]
    data = _arena_fixture(tmp_path, monkeypatch, extra)
    with pytest.raises(ValueError, match="no answer for question q3"):
        arenahard_v2.load_arenahard_v2(data_dir=data)


def _frozen(tmp_path, monkeypatch, **task_args):
    data = _arena_fixture(tmp_path, monkeypatch)
    task = pairwise_judge(data_dir=str(data), **task_args)
    questions = freeze_task(task, source="arenahard_v2", workload_class="chat_qa")
    questions = [q.model_validate_json(q.model_dump_json()) for q in questions]
    monkeypatch.setattr(arenahard_v2, "QUESTIONS_SHA256", "0" * 64)
    return questions


def test_frozen_task_requires_an_explicit_grader_role(tmp_path, monkeypatch):
    questions = _frozen(tmp_path, monkeypatch)
    for question in questions:
        assert_portable_task_config(question)
        assert question.inspect_task["required_roles"] == ["grader"]
    with pytest.raises(ValueError, match="explicitly bound"):
        restore_tasks(questions)
    router = ScriptedRouter("answer")
    model = router_model(router, submission_id="sub")
    [(task, _)] = restore_tasks(questions, model_roles={"grader": model})
    with pytest.raises(UnboundGraderRole):
        run_competition_eval(task, model, log_dir=str(tmp_path), model_roles={"grader": model})
    assert not router.calls


def test_judged_run_scores_preference_and_never_shows_the_router_a_baseline(tmp_path, monkeypatch):
    questions = _frozen(tmp_path, monkeypatch)

    def verdict(text):
        if "haiku" in text:
            return "no verdict"
        # Prefer the candidate in both positions: second in round 1, first in round 2.
        candidate_first = text.index("My answer.") < text.index("BASELINE")
        return "[[A>B]]" if candidate_first else "[[B>A]]"

    grader = judge(verdict)
    [(task, _)] = restore_tasks(questions, model_roles={"grader": grader})
    router = ScriptedRouter("My answer.")
    log = run(task, router, tmp_path, model_roles={"grader": grader})
    scores = {s.id: s.scores["pairwise_judge_scorer"] for s in log.samples}
    assert scores["arenahard_v2:arenahard-v2-q1"].value == 1.0
    unusable = scores["arenahard_v2:arenahard-v2-q2"]
    assert math.isnan(unusable.value) and unusable.reason == "grader_failed"
    assert unusable.metadata["code"] == "needs_review"
    assert "BASELINE" not in router.seen() and len(router.calls) == 2


@pytest.mark.parametrize("answer", ["", " \n\t"])
def test_empty_stopped_answers_are_incorrect_completed_attempts(tmp_path, monkeypatch, answer):
    questions = _frozen(tmp_path, monkeypatch)

    def unexpected_judge(text):
        pytest.fail("an empty answer must settle without a judge call")

    grader = judge(unexpected_judge)
    [(task, _)] = restore_tasks(questions, model_roles={"grader": grader})
    router = ScriptedRouter(answer)
    log = run(task, router, tmp_path, model_roles={"grader": grader})
    for sample in log.samples:
        assert sample.error is None
        score = sample.scores["pairwise_judge_scorer"]
        assert score.value == 0.0
        assert score.metadata["status"] == "graded"
        assert score.metadata["code"] == "incorrect"
    measurements = category_measurements_from_log(
        log,
        (),
        {q.question_id: q.workload_class for q in questions},
        run_id="run",
        miner_id="miner",
        submission_sha256="a" * 64,
        cohort_sha256="b" * 64,
        integrity_pass=True,
    )
    assert all(row.execution_complete for row in measurements)


def test_a_mismatched_judge_pin_fails_before_the_router_is_called(tmp_path, monkeypatch):
    # Catches router spend (and time) on samples whose judge can never be the pinned one.
    questions = _frozen(tmp_path, monkeypatch, judge_model="openai/gpt-5.4-mini")
    grader = judge(lambda text: "[[B>A]]")
    [(task, _)] = restore_tasks(questions, model_roles={"grader": grader})
    router = ScriptedRouter("My answer.")
    with pytest.raises(RolePinMismatch):
        run(task, router, tmp_path, model_roles={"grader": grader})
    # Evaluated directly, the per-sample guard still fails every sample first.
    model = router_model(router, submission_id="sub")
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path / "direct"), display="none")
    assert all(s.error and "RolePinMismatch" in s.error.message for s in log.samples)
    assert not router.calls


def test_the_pinned_judge_runs(tmp_path, monkeypatch):
    questions = _frozen(tmp_path, monkeypatch, judge_model="yugos_evals_judge/judge")
    grader = judge(lambda text: "[[A=B]]")
    [(task, _)] = restore_tasks(questions, model_roles={"grader": grader})
    router = ScriptedRouter("My answer.")
    log = run(task, router, tmp_path, model_roles={"grader": grader})
    assert [s.scores["pairwise_judge_scorer"].value for s in log.samples] == [0.5, 0.5]
    assert len(router.calls) == 2


class _ArenaAnswers(dict):
    """The router's scripted answers; any other prompt is a judge's, given its verdict."""

    def __missing__(self, prompt):
        return "no verdict" if "haiku" in prompt else "[[A=B]]"


def _local_run(tmp_path, monkeypatch, *, pin, bound):
    provider = ScriptedProvider(
        _ArenaAnswers({"Explain recursion.": "My answer.", "Write a haiku.": "My answer."})
    )
    monkeypatch.setattr(providers, "get_model", provider)
    make_pack(tmp_path / "pack.tar", _frozen(tmp_path / "data", monkeypatch, judge_model=pin))
    make_bundle(tmp_path / "router")
    options = RunOptions(
        pack=tmp_path / "pack.tar",
        bundle=tmp_path / "router.zip",
        max_cost_usd=1.0,
        log_dir=tmp_path / "run",
        world_card=str(CARD),
        display="none",
        service_models={"grader": parse_model_choice(bound)},
    )
    env = {"OPENAI_API_KEY": "sk-local-test", "PATH": "/usr/bin:/bin"}
    return provider, partial(
        run_local, options, env=env, card_cache=tmp_path / "cache", docker_available=False
    )


def test_the_local_runner_refuses_an_unadmitted_task_judge_before_spending(tmp_path, monkeypatch):
    provider, run_pack = _local_run(
        tmp_path, monkeypatch, pin="openai/gpt-5.4-mini", bound="openai/gpt-judge"
    )
    with pytest.raises(LocalRunError, match="models without world-card prices and limits"):
        run_pack()
    assert not (tmp_path / "run").exists() and provider.models == []


def test_a_local_run_with_the_pinned_judge_measures_unusable_judgments(tmp_path, monkeypatch):
    # Catches an unscored judgment aborting local measurement instead of voiding its category.
    _, run_pack = _local_run(
        tmp_path, monkeypatch, pin="openai/gpt-judge", bound="openai/gpt-judge"
    )
    result = run_pack()
    assert result.failure is None, result.failure
    rows = {row["question_id"]: row for row in result.summary["questions"]}
    assert rows["arenahard_v2:arenahard-v2-q1"]["grade"] == "0.5"
    unusable = rows["arenahard_v2:arenahard-v2-q2"]
    assert unusable["grade_rule"] == "needs_review" and unusable["failure_owner"] is None
    assert result.summary["summary"]["chat_qa"]["complete"] is False


@pytest.mark.skipif(
    not os.environ.get("YUGOS_EVALS_NETWORK_TESTS"), reason="downloads pinned sources (~11 MB)"
)
def test_pinned_upstream_files_load_with_every_baseline():
    loaded = arenahard_v2.load_arenahard_v2()
    categories = {s.metadata["category"] for s in loaded.samples}
    assert categories == set(arenahard_v2.CATEGORIES) and len(loaded.files) == 3
    print(f"arenahard_v2: {len(loaded.samples)} questions")
