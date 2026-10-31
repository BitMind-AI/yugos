"""Load, grade and locally publish the synthetic competition-v0.0.3 pack."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import anyio
import pytest
from evals_fakes import task_state
from inspect_ai.scorer import Target

from yugos.competition import (
    CategoryMeasurement,
    CategoryRule,
    CompetitionCohort,
    CompetitionHistory,
    CompetitionPolicy,
    CompetitionPublication,
    CompetitionResult,
    CompetitionSnapshot,
    SubmissionMeasurement,
    digest,
)
from yugos.competition.net_value import CategoryValue, NetValuePolicy
from yugos.evaluation.grades import GradeCode
from yugos.evaluation.packs import read_pack
from yugos.inspect_engine import restore_tasks
from yugos_evals.packs.competition_v003 import (
    AGENTIC_SUBJECT,
    EXACT_SUBJECTS,
    MATH_SUBJECT,
    PACK_ID,
    QUESTION_COUNT,
    ROLES,
    SUBJECT_COUNTS,
    WORKLOAD,
    WORLD_CARD_SHA256,
    build_pack,
    mean_score,
    question_spend_usd,
    role_scores,
    subject_counts,
)
from yugos_evals.scorers.math import grade_math
from yugos_evals.scorers.rubric_chat import grade_rubric_chat

# Same private keys the upload service strips from an unauthenticated snapshot.
_PRIVATE = frozenset(
    {
        "question_ids",
        "hard_question_ids",
        "private_question_ids",
        "grades",
        "question_spend_usd",
    }
)
_RANGES = {
    "weak": (Decimal("0.20"), Decimal("0.40")),
    "mid": (Decimal("0.50"), Decimal("0.70")),
    "strong": (Decimal("0.80"), Decimal("1")),
}


@pytest.fixture(scope="module")
def built():
    return build_pack()


def _strip(value):
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key not in _PRIVATE}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


_CATEGORIES = ("chat_qa", "math_code", "agentic")
_CATEGORY_VALUE = {
    "chat_qa": CategoryValue(
        weight=Decimal("0.40"),
        value_regular_usd=Decimal("1.00"),
        time_budget_seconds=Decimal("3600"),
    ),
    "math_code": CategoryValue(
        weight=Decimal("0.35"),
        value_regular_usd=Decimal("1.00"),
        time_budget_seconds=Decimal("14400"),
    ),
    "agentic": CategoryValue(
        weight=Decimal("0.25"),
        value_regular_usd=Decimal("10.00"),
        time_budget_seconds=Decimal("28800"),
    ),
}


def _policy(questions, *, cohort: str) -> CompetitionPolicy:
    rules = []
    for category in _CATEGORIES:
        selected = [question for question in questions if question.workload_class == category]
        if not selected:
            continue
        rules.append(
            CategoryRule(
                category=category,
                cohort_sha256=cohort,
                question_ids=tuple(question.question_id for question in selected),
                hard_question_ids=tuple(
                    sorted(
                        question.question_id
                        for question in selected
                        if question.panel_solve_share == 0
                    )
                ),
                initial_quality=Decimal(0),
                initial_hard_quality=Decimal(0),
                panel_leader_time_ns=60_000_000_000,
            )
        )
    return CompetitionPolicy(
        categories=tuple(rules),
        net_value=NetValuePolicy(categories=dict(_CATEGORY_VALUE)),
    )


def _measurement(questions, scores, spends, run_id, cohort):
    rows = []
    for category in _CATEGORIES:
        selected = [question for question in questions if question.workload_class == category]
        grades = {question.question_id: scores[question.question_id] for question in selected}
        spend = {question.question_id: spends[question.question_id] for question in selected}
        rows.append(
            CategoryMeasurement(
                run_id=run_id,
                miner_id=run_id,
                submission_sha256=digest({"submission": run_id}),
                cohort_sha256=cohort,
                category=category,
                grades=grades,
                question_spend_usd=spend,
                cost_usd=sum(spend.values(), Decimal(0)),
                execution_time_ns=30_000_000_000,
                cost_complete=True,
                execution_complete=True,
                integrity_pass=True,
            )
        )
    return SubmissionMeasurement(categories=tuple(rows))


def test_pack_loads_three_workload_categories(built, tmp_path):
    path = tmp_path / "pack.tar"
    path.write_bytes(built.archive)
    pack = read_pack(path)
    assert pack.manifest.pack_id == PACK_ID
    assert pack.manifest.purpose == "competition"
    assert pack.manifest.data_visibility == "private"
    assert pack.manifest.world_card_sha256 == WORLD_CARD_SHA256
    assert pack.sha256 == built.sha256
    assert len(pack.questions) == QUESTION_COUNT == 124
    assert {question.workload_class for question in pack.questions} == set(_CATEGORIES)
    assert subject_counts(built) == SUBJECT_COUNTS
    # Twelve-question benchmarks are valid. The config no longer demands 25 each.
    assert SUBJECT_COUNTS[MATH_SUBJECT] == 12
    assert SUBJECT_COUNTS[AGENTIC_SUBJECT] == 12
    assert all(question.difficulty_decile is not None for question in pack.questions)
    assert all(question.panel_solve_share is not None for question in pack.questions)
    builders = {question.inspect_task["builder"] for question in pack.questions}
    assert builders == {"yugos_evals.prepared"}
    by_origin = {
        question.workload_class: question.inspect_task["origin"]["task"]
        for question in pack.questions
    }
    assert by_origin == {
        "chat_qa": "yugos_evals/rubric_chat",
        "math_code": "yugos_evals/boxed_math",
        "agentic": "yugos_evals/rubric_chat",
    }
    assert built.summary.plan.execution["collection"] == "local-stand-in"
    assert built.summary.plan.execution["inference"] == "not-called"


def test_each_subject_mixes_exact_or_partial_difficulty(built):
    scores = role_scores(built)
    by_subject: dict[str, list] = {subject: [] for subject in SUBJECT_COUNTS}
    for question in built.questions:
        by_subject[question.source_benchmark].append(scores[question.question_id])
    for subject, rows in by_subject.items():
        assert len(rows) == SUBJECT_COUNTS[subject]
        weak = [row["weak"] for row in rows]
        strong = [row["strong"] for row in rows]
        assert min(weak) == 0
        assert max(strong) == 1
        if subject in EXACT_SUBJECTS or subject == MATH_SUBJECT:
            assert set(weak) <= {Decimal(0), Decimal(1)}
        else:
            assert any(Decimal(0) < score < Decimal(1) for row in rows for score in row.values())


def _means(scores, question_ids):
    return {role: mean_score({qid: scores[qid][role] for qid in question_ids}) for role in ROLES}


def test_stand_in_means_match_the_difficulty_targets(built):
    scores = role_scores(built)
    groups = {
        "overall": [question.question_id for question in built.questions],
        **{
            category: [
                question.question_id
                for question in built.questions
                if question.workload_class == category
            ]
            for category in _CATEGORIES
        },
    }
    for name, question_ids in groups.items():
        means = _means(scores, question_ids)
        for role, (low, high) in _RANGES.items():
            assert low <= means[role] <= high, (name, role, means[role])
        assert means["weak"] < means["mid"] < means["strong"]


def test_choice_prompts_require_an_answer_marker(built):
    choices = [item for item in built.items if item.subject == "choice_logic"]
    assert len(choices) == SUBJECT_COUNTS["choice_logic"] == 20
    for item in choices:
        assert "Answer: L" in item.prompt
        assert "Replace L with one letter: A, B, C, or D." in item.prompt
        assert "Reply with the letter only." not in item.prompt
        assert item.metadata["format"] == "choice"
        assert item.target in {"A", "B", "C", "D"}
        for completion in item.answers.values():
            assert completion.startswith("Answer: ")
        # Copied prompt text, including its option letters, is not an answer.
        copied = grade_rubric_chat(item.target, item.prompt, item.metadata)
        assert copied.code == GradeCode.INVALID_ANSWER


def test_choice_marker_ignores_incidental_letters():
    meta = {"format": "choice"}

    def grade(gold, completion):
        return grade_rubric_chat(gold, completion, meta)

    prose = grade("C", "a morning count found option b in the notes.\nAnswer: C")
    assert prose.code == GradeCode.CORRECT and prose.score == 1.0
    restated = grade("D", "A) 4 B) 9 C) 12\nThe letter a appears above.\nAnswer: D")
    assert restated.code == GradeCode.CORRECT and restated.score == 1.0
    repeated = grade("B", "Answer: A\nI revised that.\nAnswer: C\nAnswer: B")
    assert repeated.code == GradeCode.CORRECT and repeated.score == 1.0
    earlier = grade("C", "Answer: C\nAnswer: A.")
    assert earlier.code == GradeCode.INCORRECT and earlier.score == 0.0
    missing = grade("A", "a careful reading points at letter c")
    assert missing.code == GradeCode.INVALID_ANSWER
    bare = grade("C", "C")
    assert bare.code == GradeCode.INVALID_ANSWER
    lower = grade("C", "answer: c")
    assert lower.code == GradeCode.CORRECT and lower.score == 1.0
    punctuated = grade("D", "working shown above\nAnswer: D.")
    assert punctuated.code == GradeCode.CORRECT and punctuated.score == 1.0
    tight = grade("B", "Answer:B!")
    assert tight.code == GradeCode.CORRECT and tight.score == 1.0
    empty = grade("A", "")
    assert empty.code == GradeCode.INCORRECT and empty.details["cause"] == "empty_answer"
    blank = grade("A", " \n\t")
    assert blank.code == GradeCode.INCORRECT and blank.details["cause"] == "empty_answer"


def test_exact_and_rubric_graders_reject_near_miss_prose():
    exact = grade_rubric_chat("42", "42", {"format": "exact"})
    assert exact.code == GradeCode.CORRECT and exact.score == 1.0
    for near in ("the answer is 42", "a 42", "42.", "42 42", "x42"):
        outcome = grade_rubric_chat("42", near, {"format": "exact"})
        assert outcome.code == GradeCode.INCORRECT and outcome.score == 0.0, near
    facts = ["cartons=4", "kits-each=5", "withdrawn=1", "remaining=19"]
    meta = {"format": "rubric", "rubric": facts}
    full = grade_rubric_chat("unused", "\n".join(facts), meta)
    assert full.code == GradeCode.CORRECT and full.score == 1.0
    partial = grade_rubric_chat("unused", "a note says cartons=4\nkits-each=50", meta)
    assert partial.code == GradeCode.CORRECT and partial.score == 0.25
    for near in ("cartons=40", "cartons = 4", "notremaining=19", "see cartons"):
        outcome = grade_rubric_chat("unused", near, meta)
        assert outcome.code == GradeCode.INCORRECT and outcome.score == 0.0, near
    assert grade_rubric_chat("42", "", {"format": "exact"}).code == GradeCode.INCORRECT
    assert grade_rubric_chat("unused", "", meta).code == GradeCode.INCORRECT


def test_restored_grader_matches_the_stand_in_scores(built):
    scores = role_scores(built)
    restored = {
        sample.id: (task, sample)
        for task, _config in restore_tasks(built.questions)
        for sample in task.dataset
    }
    assert set(restored) == set(scores)

    def check(question_id, role, completion):
        task, sample = restored[question_id]
        state = task_state(completion, input=str(sample.input), metadata=dict(sample.metadata))
        scored = anyio.run(task.scorer[0], state, Target(sample.target))
        if question_id.startswith(f"{MATH_SUBJECT}:"):
            direct = grade_math(str(sample.target), completion)
        else:
            direct = grade_rubric_chat(str(sample.target), completion, dict(sample.metadata))
        assert Decimal(str(scored.value)) == scores[question_id][role] == Decimal(str(direct.score))

    for item in built.items:
        for role in ROLES:
            check(item.question_id, role, item.answers[role])


def test_question_spend_uses_panel_prices_and_sums(built):
    by_id = {item.question_id: item for item in built.items}
    spends = {
        role: {
            item.question_id: question_spend_usd(
                "deepseek-v4-flash-0731" if role == "weak" else "gemini-3.8-flash",
                item.prompt,
                item.answers[role],
            )
            for item in built.items
        }
        for role in ("weak", "strong")
    }
    for role, rows in spends.items():
        assert all(amount > 0 for amount in rows.values())
        assert sum(rows.values(), Decimal(0)) > 0
    # Gemini Flash is priced above DeepSeek Flash on the pinned world card.
    assert sum(spends["strong"].values(), Decimal(0)) > sum(spends["weak"].values(), Decimal(0))
    priced = {
        row.question_id: row.spend_usd for row in built.summary.outcomes if row.question_id in by_id
    }
    assert set(priced) == {item.question_id for item in built.items}
    assert all(amount > 0 for amount in priced.values())


def test_local_snapshot_publication_keeps_question_ids_private(built):
    scores = role_scores(built)
    cohort = digest({"pack": built.sha256})
    policy = _policy(built.questions, cohort=cohort)
    # The floor starts at zero so the weak stand-in can be the first champion.
    policy = policy.model_copy(
        update={
            "categories": tuple(
                rule.model_copy(
                    update={
                        "initial_quality": mean_score(
                            {qid: scores[qid]["weak"] for qid in rule.question_ids}
                        )
                    }
                )
                for rule in policy.categories
            )
        }
    )
    spends = {
        role: {
            item.question_id: question_spend_usd(
                "deepseek-v4-flash-0731" if role == "weak" else "gemini-3.8-flash",
                item.prompt,
                item.answers[role],
            )
            for item in built.items
        }
        for role in ("weak", "strong")
    }
    weak = _measurement(
        built.questions,
        {qid: row["weak"] for qid, row in scores.items()},
        spends["weak"],
        "stand-in-weak",
        cohort,
    )
    strong = _measurement(
        built.questions,
        {qid: row["strong"] for qid, row in scores.items()},
        spends["strong"],
        "stand-in-strong",
        cohort,
    )
    history = CompetitionHistory(policy=policy).challenge(weak).challenge(strong)
    assert history.champion().run_id == "stand-in-strong"
    decision = history.challenges[1].decision
    assert decision.promoted
    assert decision.lower_bound > decision.min_net_value

    snapshot = CompetitionSnapshot(
        history=history, proposed_champion_run_id=history.champion().run_id
    )
    published_snapshot = {"id": digest(snapshot), "payload": snapshot.model_dump(mode="json")}
    CompetitionSnapshot.read(published_snapshot, policy_sha256=digest(policy))

    now = datetime.now(timezone.utc)
    result = CompetitionResult(netuid=169, history=history)
    publication = {
        "id": digest(result),
        "confirmed_at": now.isoformat(),
        "payload": result.model_dump(mode="json"),
    }
    CompetitionPublication.read(
        publication,
        cohort=CompetitionCohort(netuid=169, policy_sha256=digest(policy)),
        now=now,
        max_age_s=60,
    )
    private = json.dumps(publication)
    public = json.dumps(_strip(publication))
    for question in built.questions:
        assert question.question_id in private
        assert question.question_id not in public
    # The validator payload keeps the ids; the stripped body is what an
    # unauthenticated scoring snapshot returns.
    assert "question_ids" in private and "question_ids" not in public


def test_pack_file_round_trip_matches_the_builder(built, tmp_path: Path):
    path = tmp_path / "pack.tar"
    path.write_bytes(built.archive)
    assert read_pack(path).questions == built.questions


def test_dropping_a_category_fails_the_three_category_policy(built):
    cohort = digest({"pack": built.sha256})
    policy = _policy(built.questions, cohort=cohort)
    assert {rule.category for rule in policy.categories} == set(_CATEGORIES)
    assert {question.workload_class for question in built.questions} == {
        WORKLOAD[question.source_benchmark] for question in built.questions
    }
    kept = tuple(question for question in built.questions if question.workload_class != "agentic")
    with pytest.raises(ValueError, match="net-value categories"):
        _policy(kept, cohort=cohort)
