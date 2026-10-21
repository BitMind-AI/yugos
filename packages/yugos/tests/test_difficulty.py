"""Difficulty preserves fractional grades, balanced treatments and frozen evidence identities."""

from decimal import Decimal

import pytest

from yugos.competition import digest
from yugos.competition.difficulty import (
    CalibrationOutcome,
    CalibrationSpec,
    CalibrationSummary,
    DifficultyCalibrationPlan,
    Observation,
    Treatment,
    calculate,
    question_identity,
    validate_labels,
    verify_summary,
)
from yugos.evaluation import Question
from yugos.inference.actions import ModelChoice
from yugos.inference.calls import CallLimits
from yugos.spend import KnownSpend, OpenReservation


@pytest.fixture
def evidence():
    plan = DifficultyCalibrationPlan(
        questions=tuple(
            Question(
                question_id=f"q{i}",
                prompt=f"Question {i}",
                expected_answer="yes",
                inspect_task={"builder": "fixture"},
                inspect_sample={"id": "fixture"},
                workload_class="chat_qa",
            )
            for i in range(12)
        ),
        panel=(
            Treatment(model=ModelChoice(provider_id="fixture", model_id="a"), weight=1, repeats=1),
            Treatment(model=ModelChoice(provider_id="fixture", model_id="b"), weight=3, repeats=2),
        ),
        grader={"source": "fixture"},
        execution={"runtime": "fixture"},
        seed="declared-before-outcomes",
        limits=CallLimits(max_cost_usd=1, max_question_cost_usd=1),
    )
    rows = []
    for q in plan.questions:
        for treatment in plan.panel:
            for repeat in range(treatment.repeats):
                score = Decimal("0.1") + Decimal(q.question_id[1:]) / 20
                if treatment.model.model_id == "b":
                    score += Decimal("0.1") * repeat
                rows.append(
                    Observation(
                        plan_sha256=plan.sha256,
                        question_id=q.question_id,
                        question_sha256=question_identity(q),
                        treatment=treatment.key,
                        repeat=repeat,
                        outcome_sha256=digest(
                            {"q": q.question_id, "model": treatment.key, "repeat": repeat}
                        ),
                        score=score,
                    )
                )
    return plan, rows


def test_native_partial_credit_and_within_treatment_repeats(evidence):
    plan, rows = evidence
    result = calculate(plan, rows)
    label = result.labels["q0"]
    # Equal treatment weighting or treating any positive grade as solved both fail this.
    assert label.easiness == (Decimal("0.1") + 3 * Decimal("0.15")) / 4
    assert label.difficulty == 1 - label.easiness
    assert label.decile == 9
    assert label.observations == 3
    assert sorted(label.treatment_ranges.values()) == [
        (Decimal("0.1"), Decimal("0.1")),
        (Decimal("0.1"), Decimal("0.2")),
    ]


def test_summary_replays_full_pool_and_rejects_substituted_outcomes(evidence):
    plan, rows = evidence
    spec = CalibrationSpec.from_plan(plan)
    assert calculate(spec, rows) == calculate(plan, rows)
    outcomes = tuple(
        CalibrationOutcome(
            **row.model_dump(),
            evidence_bytes=20,
            active_ns=100,
            spend_usd="0.001",
            service_spend_usd=0,
            cost_complete=True,
            call_ids=(row.outcome_sha256,),
        )
        for row in rows
    )
    summary = CalibrationSummary(plan=spec, outcomes=outcomes, result=calculate(plan, rows))
    selected = plan.questions[:2]
    selected = tuple(
        q.model_copy(
            update={
                "difficulty_decile": summary.result.labels[q.question_id].decile,
                "panel_solve_share": summary.result.labels[q.question_id].easiness,
            }
        )
        for q in selected
    )
    verify_summary(summary, selected)
    changed = outcomes[0].model_copy(update={"score": Decimal("0.99")})
    with pytest.raises(ValueError, match="differs"):
        verify_summary(summary.model_copy(update={"outcomes": (changed, *outcomes[1:])}))
    changed = outcomes[1].model_copy(update={"call_ids": outcomes[0].call_ids})
    with pytest.raises(ValueError, match="reuses a provider call"):
        verify_summary(
            summary.model_copy(update={"outcomes": (outcomes[0], changed, *outcomes[2:])})
        )
    changed = outcomes[0].model_copy(
        update={"spend": OpenReservation(reservation_usd=Decimal("0.001"))}
    )
    with pytest.raises(ValueError, match="unresolved"):
        verify_summary(summary.model_copy(update={"outcomes": (changed, *outcomes[1:])}))
    missing = outcomes[0].model_copy(
        update={
            "score": None,
            "missing_reason": "absent",
            "spend": OpenReservation(reservation_usd=Decimal("0.001")),
        }
    )
    with pytest.raises(ValueError, match="unresolved"):
        verify_summary(summary.model_copy(update={"outcomes": (missing, *outcomes[1:])}))
    zero = outcomes[0].model_copy(
        update={"spend_usd": Decimal("0"), "spend": KnownSpend(usd=Decimal("0"))}
    )
    verify_summary(summary.model_copy(update={"outcomes": (zero, *outcomes[1:])}))


def test_input_order_does_not_change_plan_or_ranked_ties(evidence):
    plan, rows = evidence
    rows = [
        r.model_copy(update={"score": Decimal("0.2") if r.question_id != "q11" else Decimal("0.9")})
        for r in rows
    ]
    first = calculate(plan, rows)
    shuffled = DifficultyCalibrationPlan.model_validate(
        plan.model_dump()
        | {
            "questions": tuple(reversed(plan.questions)),
            "panel": tuple(reversed(plan.panel)),
        }
    )
    assert shuffled.sha256 == plan.sha256
    assert calculate(shuffled, list(reversed(rows))) == first
    assert first.pools["chat_qa"]["tie_broken_hard_boundary"]
    assert first.labels["q0"].tied_questions == 11


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "failure", "question", "panel", "repeat"]
)
def test_invalid_or_incomplete_evidence_cannot_change_difficulty(evidence, change):
    plan, rows = evidence
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    else:
        updates = {
            "failure": {"score": None, "missing_reason": "transport timeout"},
            "question": {"question_sha256": "a" * 64},
            "panel": {"plan_sha256": "b" * 64},
            "repeat": {"repeat": 100},
        }
        rows[0] = rows[0].model_copy(update=updates[change])
    with pytest.raises(ValueError):
        calculate(plan, rows)


def test_all_zero_questions_need_no_review_and_missing_can_be_excluded(evidence):
    plan, rows = evidence
    rows = [
        r.model_copy(update={"score": Decimal(0)}) if r.question_id == "q0" else r for r in rows
    ]
    # Unsolved by the whole panel is a valid label: paired scoring makes a broken
    # question that nobody can solve contribute nothing to net value.
    result = calculate(plan, rows)
    assert result.labels["q0"].difficulty == 1 and result.labels["q0"].easiness == 0
    rows = [r for r in rows if r.question_id != "q0"]
    result = calculate(plan, rows, exclusions={"q0": "Unresolved environment failure"})
    assert "q0" not in result.labels


def test_small_and_all_tied_pools_cannot_claim_deciles(evidence):
    plan, rows = evidence
    with pytest.raises(ValueError, match="pool too small"):
        calculate(plan, rows, exclusions={f"q{i}": "outside scope" for i in range(3)})
    with pytest.raises(ValueError, match="all scores tied"):
        calculate(plan, [r.model_copy(update={"score": Decimal("0.5")}) for r in rows])


@pytest.mark.parametrize(
    "change",
    [
        "prompt",
        "expected_answer",
        "inspect_task",
        "difficulty_decile",
        "panel_solve_share",
        "unset",
    ],
)
def test_selection_rejects_stale_or_substituted_inputs(evidence, change):
    plan, rows = evidence
    result = calculate(plan, rows)
    label = result.labels[plan.questions[0].question_id]
    question = plan.questions[0].model_copy(
        update={"difficulty_decile": label.decile, "panel_solve_share": label.easiness}
    )
    validate_labels([question], result)
    updates = {
        "prompt": "different",
        "expected_answer": "different",
        "inspect_task": {"builder": "different"},
        "difficulty_decile": 0,
        "panel_solve_share": label.easiness + Decimal("0.01"),
    }
    if change == "unset":
        change, updates["unset"] = "panel_solve_share", None
    value = updates.pop("unset", updates[change])
    with pytest.raises(ValueError, match="stale or substituted"):
        validate_labels([question.model_copy(update={change: value})], result)


def test_solve_share_is_derived_not_part_of_question_identity(evidence):
    plan, _ = evidence
    question = plan.questions[0]
    labeled = question.model_copy(update={"panel_solve_share": Decimal("0.2")})
    assert question_identity(labeled) == question_identity(question)
    with pytest.raises(ValueError, match="derived labels"):
        DifficultyCalibrationPlan.model_validate(
            plan.model_dump() | {"questions": (labeled, *plan.questions[1:])}
        )
