"""Turn an Inspect log and provider receipts into Yugos measurements."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Literal

from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL, Score, value_to_float

from yugos.competition import (
    CategoryMeasurement,
    competition_grade,
    grade_settled,
)
from yugos.competition.net_value import PRECISION
from yugos.evaluation.grades import GradeCode, GradingResult
from yugos.evaluation.measurements import Outcome, QuestionMeasurement, WorkloadClass
from yugos.spend import KnownSpend, OpenReservation

from .receipts import ProviderReceipt, receipts_by_sample
from .router_model import QUESTION_FAILURE_KEY, ROUTER_FAILURE_KEY

# Limits that are Yugos deadlines rather than task rules; hitting one is a non-answer.
_DEADLINE_LIMITS = frozenset({"time", "working", "cost"})
# Text values Inspect's `value_to_float` converts; any other text is a malformed score.
_SENTINELS = frozenset({CORRECT, INCORRECT, PARTIAL, NOANSWER})
_BOOLEAN_TEXT = frozenset({"yes", "no", "true", "false"})
_to_float = value_to_float()


@dataclass(frozen=True)
class SampleMeasurement:
    """One logged sample as a Yugos measurement, with the evidence behind it."""

    measurement: QuestionMeasurement
    sample_uuid: str
    grade: GradingResult | None
    spend_usd: Decimal
    # Pre-scoring working time, the category execution time this sample contributes.
    active_ns: int | None
    failure_owner: Literal["submission", "infrastructure"] | None
    error: str | None = None


def active_seconds(sample: EvalSample) -> float:
    """Working time from the start of the solvers until scoring begins.

    Inspect's `working_time` includes scoring, so the start of the top-level
    `scorers` span is used instead. Working time already excludes waits for
    connection slots, sandboxes and retry backoff. A sample that errored before
    scoring has no such span and its whole working time counts.
    """
    if not sample.events:
        raise ValueError(f"sample {sample.id} has no events to time")
    starts: dict[str, float] = {}
    for event in sample.events:
        if event.event == "span_begin" and event.parent_id is None:
            starts.setdefault(event.name, event.working_start)
    if "scorers" not in starts:
        return sample.working_time or 0.0
    return max(0.0, starts["scorers"] - starts.get("solvers", 0.0))


def _selected_score(sample: EvalSample, scorer: str | None) -> Score | None:
    """The sample's score; only an errored sample may lack one, and it then scores zero."""
    scores = sample.scores or {}
    if scorer is None:
        if len(scores) > 1:
            raise ValueError(f"sample {sample.id} has several scores; choose a scorer")
        score = next(iter(scores.values()), None)
    else:
        score = scores.get(scorer)
    if score is None and sample.error is None:
        # A mistyped scorer or an unscored log must not become a valid zero.
        named = f" from scorer {scorer!r}" if scorer else ""
        raise ValueError(f"sample {sample.id} has no score{named}")
    return score


def _score_value(score: Score | None, key: str | None) -> float | None:
    """The selected scalar score, or None when the scorer left the sample unscored.

    Inspect's unscored value (NaN) must carry a reason; other non-numeric or
    unconvertible values are malformed and never become a silent zero.
    """
    if score is None:
        return 0.0
    value = score.value
    if isinstance(value, dict) and key is not None:
        if key not in value:
            raise ValueError(f"score has no key {key!r}")
        value = value[key]
    if isinstance(value, dict | list):
        raise ValueError("measurements require a scalar score; choose a score key")
    if isinstance(value, float) and math.isnan(value):
        if not score.reason:
            raise ValueError("an unscored result must give its reason")
        return None
    if not isinstance(value, bool | int | float | str) or (
        isinstance(value, str)
        and value not in _SENTINELS
        and value.lower() not in _BOOLEAN_TEXT
        and not _finite(value)
    ):
        raise ValueError(f"score value {value!r} is not a grade")
    return float(_to_float(value))


def _finite(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _unscored_grade(
    score: Score, grader: str, *, truncated: bool, router_failed: bool
) -> GradingResult | None:
    """The Yugos grade for a sample the scorer left unscored, as legacy grading gates it.

    A router non-answer decides first: truncation is a settled zero and a
    router failure or deadline has no grade. Otherwise the grader could not
    establish a grade (invalid reference, grader error, unusable judgment,
    unknown completion status), which needs review and leaves the category
    incomplete.
    """
    # The unscored value is NaN, which JSON evidence cannot carry.
    evidence = score.model_dump(mode="json", exclude={"value"})
    details = {"cause": str(score.reason), "inspect_score": evidence}
    if truncated:
        return GradingResult(
            score=0.0, grader=grader, code=GradeCode.TRUNCATED, status="skipped", details=details
        )
    if router_failed:
        return None
    return GradingResult(
        score=0.0,
        grader=grader,
        code=GradeCode.NEEDS_REVIEW,
        status="unavailable",
        details=details,
    )


def _failure(sample: EvalSample) -> Literal["submission", "infrastructure"] | None:
    if QUESTION_FAILURE_KEY in sample.store:
        return "infrastructure"
    if ROUTER_FAILURE_KEY in sample.store:
        return "submission"
    if sample.error is None:
        return None
    return "infrastructure"


def _grade_rule(sample: EvalSample, grade: GradingResult | None, owner: str | None) -> str:
    if grade is not None:
        return grade.code.value
    if owner is not None:
        return "router_error" if owner == "submission" else "sample_error"
    if sample.limit is not None:
        return f"{sample.limit.type}_limit"
    return ""


def measure_log(
    log: EvalLog,
    receipts: Iterable[ProviderReceipt],
    *,
    scorer: str | None = None,
    isolate_errors: bool = False,
) -> tuple[SampleMeasurement, ...]:
    """Measure each logged sample with receipt spend and pre-scoring time.

    Spend is the sum of the sample's metering receipts; Inspect's own usage
    and cost estimates are ignored. Errors, deadline limits and outputs with
    neither text nor tool calls are non-answers. A sample the scorer left
    unscored (with a reason) needs review unless the router's own non-answer
    decides it first (see
    `_unscored_grade`). Restored frozen samples carry their workload labels,
    source identity, authoritative scorer and optional scalar score key.
    Execution hosts may isolate malformed grades as incomplete questions;
    attribution and receipt-integrity errors always propagate.
    """
    samples = log.samples or []
    if any(sample.epoch != 1 for sample in samples):
        raise ValueError("competition measurements take one epoch per question")
    if len({str(sample.id) for sample in samples}) != len(samples):
        raise ValueError("duplicate sample IDs")
    spend = receipts_by_sample(receipts)
    if not spend.keys() <= {sample.uuid for sample in samples}:
        raise ValueError("a receipt cannot be attributed to a logged sample")
    rows = []
    for sample in samples:
        metadata = sample.metadata or {}
        category = metadata.get("workload_class")
        if not category:
            raise ValueError(f"sample {sample.id} has no workload class")
        owner = _failure(sample)
        output = sample.output
        finish = (
            {"max_tokens": "length", "model_length": "length"}.get(
                output.stop_reason, output.stop_reason
            )
            if output.choices
            else None
        )
        deadline = sample.limit is not None and sample.limit.type in _DEADLINE_LIMITS
        grade, error = None, None
        try:
            score = (
                None
                if owner is not None
                else _selected_score(sample, scorer or metadata.get("yugos_scorer"))
            )
            value = _score_value(score, metadata.get("yugos_score_key"))
            if score is not None:
                grader = scorer or metadata.get("yugos_scorer") or next(iter(sample.scores))
                if value is None:
                    grade = _unscored_grade(
                        score,
                        grader,
                        truncated=finish == "length",
                        router_failed=deadline,
                    )
                    value = 0.0
                else:
                    grade = GradingResult(
                        score=value,
                        grader=grader,
                        code=GradeCode.CORRECT if value > 0 else GradeCode.INCORRECT,
                        details={"inspect_score": score.model_dump(mode="json")},
                    )
        except Exception as exc:
            if not isolate_errors:
                raise
            error = f"{type(exc).__name__}: {exc}"
            value, grade, owner = 0.0, None, "infrastructure"
        with localcontext() as context:
            context.prec = PRECISION
            cost = sum((r.cost_usd for r in spend.get(sample.uuid, ())), Decimal(0))
        active = active_seconds(sample)
        answered = bool(output.completion.strip()) or bool(
            output.choices and output.message.tool_calls
        )
        outcome = Outcome(
            score=value,
            cost_usd=float(cost),
            latency_s=active,
            unrealized=owner is not None or deadline or not answered,
            finish_reason=finish,
        )
        measurement = QuestionMeasurement(
            question_id=str(sample.id),
            workload_class=category,
            source_benchmark=metadata.get("source_benchmark") or log.eval.task,
            difficulty_decile=metadata.get("difficulty_decile"),
            outcome=outcome,
            truncated=finish == "length",
            grade_rule=_grade_rule(sample, grade, owner),
        )
        rows.append(
            SampleMeasurement(
                measurement=measurement,
                sample_uuid=sample.uuid,
                grade=grade,
                spend_usd=cost,
                active_ns=round(active * 1_000_000_000),
                failure_owner=owner,
                error=error
                or (sample.error.message if sample.error else None)
                or sample.store.get(QUESTION_FAILURE_KEY)
                or sample.store.get(ROUTER_FAILURE_KEY),
            )
        )
    return tuple(rows)


def category_measurements_from_log(
    log: EvalLog | Sequence[EvalLog],
    receipts: Iterable[ProviderReceipt],
    assigned: Mapping[str, WorkloadClass],
    *,
    run_id: str,
    miner_id: str,
    submission_sha256: str,
    cohort_sha256: str,
    integrity_pass: bool,
    open_reservations: Sequence[Decimal] = (),
    scorer: str | None = None,
    isolate_errors: bool = False,
) -> tuple[CategoryMeasurement, ...]:
    """Category evidence for `yugos.competition`, from one submission's Inspect log.

    `assigned` maps every cohort question ID to its category. A category is
    complete only when the eval succeeded, every assigned question was logged,
    no infrastructure error occurred and every Yugos grade settled. The caller
    attests sandbox cleanup (`integrity_pass`). Receipt totals are a known bill.
    `open_reservations` are quotes for calls that produced no priced receipt.
    Execution time is Inspect pre-scoring working time, using the same calculation
    as the panel baseline.
    """
    logs = [log] if isinstance(log, EvalLog) else list(log)
    receipts = tuple(receipts)
    sample_ids = {s.uuid for current in logs for s in current.samples or ()}
    if not {r.sample_uuid for r in receipts} <= sample_ids:
        raise ValueError("a receipt cannot be attributed to this pack")
    measured = [
        row
        for current in logs
        for row in measure_log(
            current,
            [r for r in receipts if r.sample_uuid in {s.uuid for s in current.samples or ()}],
            scorer=scorer,
            isolate_errors=isolate_errors,
        )
    ]
    if len({row.measurement.question_id for row in measured}) != len(measured):
        raise ValueError("a question appears in more than one Inspect task")
    return category_measurements_from_samples(
        measured,
        assigned,
        run_id=run_id,
        miner_id=miner_id,
        submission_sha256=submission_sha256,
        cohort_sha256=cohort_sha256,
        integrity_pass=integrity_pass,
        open_reservations=open_reservations,
        execution_complete=all(current.status == "success" for current in logs),
    )


def category_measurements_from_samples(
    samples: Iterable[SampleMeasurement],
    assigned: Mapping[str, WorkloadClass],
    *,
    run_id: str,
    miner_id: str,
    submission_sha256: str,
    cohort_sha256: str,
    integrity_pass: bool,
    question_spend_usd: Mapping[str, Decimal] | None = None,
    open_reservations: Sequence[Decimal] = (),
    execution_complete: bool = True,
) -> tuple[CategoryMeasurement, ...]:
    """Aggregate fresh or retained samples using the same competition rules.

    Ordered retries keep the first graded outcome and count all working time.
    An executor may supply cumulative question spend, including charges for
    attempts with no completed sample. Missing timing leaves a category incomplete.
    """
    samples = tuple(samples)
    if any(row.measurement.question_id not in assigned for row in samples):
        raise ValueError("samples contain questions outside the assignment")
    if any(
        row.measurement.workload_class != assigned[row.measurement.question_id] for row in samples
    ):
        raise ValueError("sample question category conflicts with the assignment")
    rows = {}
    for row in samples:
        qid = row.measurement.question_id
        previous = rows.get(qid)
        if previous is None or previous.grade is None or previous.grade.status != "graded":
            rows[qid] = row
    with localcontext() as context:
        context.prec = PRECISION
        if question_spend_usd is None:
            costs = dict.fromkeys(assigned, Decimal(0))
            for row in samples:
                costs[row.measurement.question_id] += row.spend_usd
        else:
            costs = dict(question_spend_usd)
        if not costs.keys() <= assigned.keys():
            raise ValueError("spend contains questions outside the assignment")
        held = sum((Decimal(amount) for amount in open_reservations), Decimal(0))
    measurements = []
    for category in sorted(set(assigned.values())):
        ids = [qid for qid, assigned_category in assigned.items() if assigned_category == category]
        present = [rows[qid] for qid in ids if qid in rows]
        history = [row for row in samples if row.measurement.workload_class == category]
        spend = {qid: costs.get(qid, Decimal(0)) for qid in ids}
        with localcontext() as context:
            context.prec = PRECISION
            cost_usd = sum(spend.values(), Decimal(0))
        if held > 0:
            category_spend = OpenReservation(reservation_usd=held)
        elif held < 0:
            raise ValueError("an open reservation must be positive")
        else:
            category_spend = KnownSpend(usd=cost_usd)
        complete = (
            execution_complete
            and len(present) == len(ids)
            and all(row.active_ns is not None for row in history)
            and all(row.failure_owner != "infrastructure" for row in present)
            and all(grade_settled(row.grade) for row in present)
        )
        measurements.append(
            CategoryMeasurement(
                run_id=run_id,
                miner_id=miner_id,
                submission_sha256=submission_sha256,
                cohort_sha256=cohort_sha256,
                category=category,
                grades={
                    qid: competition_grade(rows[qid].measurement if qid in rows else None)
                    for qid in ids
                },
                question_spend_usd=spend,
                cost_usd=cost_usd,
                execution_time_ns=sum(
                    row.active_ns for row in history if row.active_ns is not None
                ),
                question_time_ns={
                    qid: sum(
                        row.active_ns
                        for row in history
                        if row.measurement.question_id == qid and row.active_ns is not None
                    )
                    for qid in ids
                },
                spend=category_spend,
                execution_complete=complete,
                integrity_pass=integrity_pass,
            )
        )
    return tuple(measurements)
