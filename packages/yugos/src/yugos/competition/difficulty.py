"""Reproducible, panel-relative question difficulty, independent of collection services."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal, localcontext
from typing import Any, Literal

from pydantic import Field, model_validator

from ..evaluation import Question
from ..evaluation.measurements import WorkloadClass
from ..inference.actions import ModelChoice
from ..inference.calls import CallLimits
from ..spend import Spend, apply_legacy_spend
from .rules import Digest, Name, Quality, Record, digest


class CalibrationCandidate(Record):
    """Identity and ranking category of a separately retained frozen question."""

    question_id: Name
    question_sha256: Digest
    workload_class: WorkloadClass


def question_identity(question: Question | CalibrationCandidate) -> str:
    """Bind prompt, gold, native grader and category without a derived difficulty label."""
    if isinstance(question, CalibrationCandidate):
        return question.question_sha256
    return digest(
        question.model_dump(mode="json", exclude={"difficulty_decile", "panel_solve_share"})
    )


class Treatment(Record):
    model: ModelChoice
    weight: Decimal = Field(gt=0)
    repeats: int = Field(ge=1, strict=True)

    @property
    def key(self) -> str:
        return digest(self.model)


class CalibrationSettings(Record):
    """Panel and calculation settings shared by full plans and their retained identities."""

    panel: tuple[Treatment, ...] = Field(min_length=1)
    grader: dict[str, str]
    execution: dict
    limits: CallLimits
    algorithm: Literal["weighted_native_mean_category_rank"] = "weighted_native_mean_category_rank"
    grouping: Literal["workload_class"] = "workload_class"
    tie_policy: Literal["seeded_order"] = "seeded_order"
    seed: Name
    minimum_pool_size: int = Field(default=10, ge=10, strict=True)
    failure_policy: Literal["exclude_missing_truncation_zero"] = "exclude_missing_truncation_zero"

    @model_validator(mode="after")
    def canonical_order(self):
        if len({q.question_id for q in self.questions}) != len(self.questions):
            raise ValueError("duplicate calibration question IDs")
        if len({t.key for t in self.panel}) != len(self.panel):
            raise ValueError("duplicate calibration treatments")
        if not self.grader or not self.execution:
            raise ValueError("calibration requires grader and execution identities")
        categories = [q.workload_class for q in self.questions]
        if any(categories.count(category) < self.minimum_pool_size for category in set(categories)):
            raise ValueError("candidate category pool is smaller than the declared minimum")
        object.__setattr__(
            self, "questions", tuple(sorted(self.questions, key=lambda q: q.question_id))
        )
        object.__setattr__(self, "panel", tuple(sorted(self.panel, key=lambda t: t.key)))
        return self

    @property
    def sha256(self) -> str:
        return digest(self)


class DifficultyCalibrationPlan(CalibrationSettings):
    """Freeze candidates and collection settings before observing the panel's answers."""

    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unlabelled_candidates(self):
        if any(
            q.difficulty_decile is not None or q.panel_solve_share is not None
            for q in self.questions
        ):
            raise ValueError("candidate questions must not contain derived labels")
        return self


class CalibrationSpec(CalibrationSettings):
    """Small plan representation; full audit reconstructs and checks its original identity."""

    questions: tuple[CalibrationCandidate, ...] = Field(min_length=1)
    source_plan_sha256: Digest

    @property
    def sha256(self) -> str:
        return self.source_plan_sha256

    @classmethod
    def from_plan(cls, plan: DifficultyCalibrationPlan):
        return cls(
            **plan.model_dump(mode="json", exclude={"questions"}),
            source_plan_sha256=plan.sha256,
            questions=tuple(
                CalibrationCandidate(
                    question_id=q.question_id,
                    question_sha256=question_identity(q),
                    workload_class=q.workload_class,
                )
                for q in plan.questions
            ),
        )


class Observation(Record):
    """One scheduled repeat, including a missing result rather than an invented zero."""

    plan_sha256: Digest
    question_id: Name
    question_sha256: Digest
    treatment: Digest
    repeat: int = Field(ge=0, strict=True)
    outcome_sha256: Digest
    score: Quality | None
    missing_reason: str | None = None

    @model_validator(mode="after")
    def missing_is_explicit(self):
        if (self.score is None) != bool(self.missing_reason and self.missing_reason.strip()):
            raise ValueError("missing scores require a reason; valid scores cannot have one")
        return self


class DifficultyLabel(Record):
    question_sha256: Digest
    pool: Name
    easiness: Quality
    difficulty: Quality
    decile: int = Field(ge=0, le=9, strict=True)
    treatment_means: dict[str, Quality]
    treatment_ranges: dict[str, tuple[Quality, Quality]]
    observations: int = Field(ge=1)
    tied_questions: int = Field(ge=1)


class DifficultyCalibrationResult(Record):
    plan_sha256: Digest
    evidence_sha256: Digest
    labels: dict[str, DifficultyLabel]
    pools: dict[str, dict]
    exclusions: dict[str, str]


class CalibrationOutcome(Observation):
    """Verified measurements of one attempt, with a reference to external evidence."""

    evidence_bytes: int = Field(gt=0, strict=True)
    active_ns: int = Field(ge=0, strict=True)
    spend_usd: Decimal = Field(ge=0)
    service_spend_usd: Decimal = Field(ge=0)
    spend: Spend
    call_ids: tuple[Name, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def load_legacy_cost(cls, data: Any) -> Any:
        return apply_legacy_spend(data, amount_key="spend_usd", drop_amount=False)

    @model_validator(mode="after")
    def known_matches_attributed_spend(self):
        if self.spend.status == "known" and self.spend.usd != self.spend_usd:
            raise ValueError("known spend must equal the attributed spend")
        return self

    @property
    def observation(self) -> Observation:
        return Observation(**{key: getattr(self, key) for key in Observation.model_fields})


class CalibrationSummary(Record):
    """Portable outcomes and labels; raw native evidence is retained separately."""

    schema_version: Literal["ycore.difficulty_calibration.v2"] = "ycore.difficulty_calibration.v2"
    plan: CalibrationSpec
    outcomes: tuple[CalibrationOutcome, ...]
    result: DifficultyCalibrationResult


def verify_summary(summary: CalibrationSummary, questions=None) -> None:
    """Reproduce labels from published outcomes without claiming to replay raw evidence."""
    calls = [call for row in summary.outcomes for call in row.call_ids]
    if len(calls) != len(set(calls)):
        raise ValueError("calibration reuses a provider call")
    if any(row.spend.status != "known" for row in summary.outcomes):
        raise ValueError("unresolved")
    reproduced = calculate(
        summary.plan,
        [row.observation for row in summary.outcomes],
        exclusions=summary.result.exclusions,
    )
    if reproduced != summary.result:
        raise ValueError("calibration result differs from its outcomes")
    if questions is not None:
        validate_labels(questions, reproduced)


def observation_key(row: Observation) -> tuple[str, str, int]:
    return row.question_id, row.treatment, row.repeat


def calculate(
    plan: DifficultyCalibrationPlan | CalibrationSpec,
    observations: Sequence[Observation],
    *,
    exclusions: dict[str, str] | None = None,
) -> DifficultyCalibrationResult:
    """Average scheduled repeats per treatment, then rank weighted means within frozen categories."""
    plan = type(plan).model_validate(plan.model_dump(mode="json"))
    plan_sha256 = plan.sha256
    rows = sorted(
        (Observation.model_validate(r.model_dump()) for r in observations), key=observation_key
    )
    exclusions = dict(exclusions or {})
    questions = {q.question_id: q for q in plan.questions}
    treatments = {t.key: t for t in plan.panel}
    if not set(exclusions) <= questions.keys() or any(not r.strip() for r in exclusions.values()):
        raise ValueError("exclusions require candidate IDs and nonempty reasons")
    grouped = defaultdict(list)
    seen, raw_ids = set(), set()
    for row in rows:
        key = observation_key(row)
        if key in seen or row.outcome_sha256 in raw_ids:
            raise ValueError("duplicate calibration observation or reused raw outcome")
        seen.add(key)
        raw_ids.add(row.outcome_sha256)
        question, treatment = questions.get(row.question_id), treatments.get(row.treatment)
        if (
            row.plan_sha256 != plan_sha256
            or question is None
            or treatment is None
            or row.question_sha256 != question_identity(question)
            or row.repeat >= treatment.repeats
        ):
            raise ValueError("observation conflicts with frozen question, panel or settings")
        grouped[row.question_id].append(row)
    labels, pools = {}, defaultdict(list)
    with localcontext() as context:
        context.prec = 28
        for qid, question in questions.items():
            if qid in exclusions:
                continue
            assigned = grouped[qid]
            expected = sum(t.repeats for t in plan.panel)
            if len(assigned) != expected or any(r.score is None for r in assigned):
                raise ValueError(f"{qid}: insufficient valid treatment/repeat coverage")
            means, ranges = {}, {}
            for key, treatment in treatments.items():
                scores = [r.score for r in assigned if r.treatment == key]
                if len(scores) != treatment.repeats:
                    raise ValueError(f"{qid}: incomplete treatment coverage")
                means[key] = sum(scores, Decimal(0)) / len(scores)
                ranges[key] = (min(scores), max(scores))
            easiness = sum((means[t.key] * t.weight for t in plan.panel), Decimal(0)) / sum(
                (t.weight for t in plan.panel), Decimal(0)
            )
            labels[qid] = DifficultyLabel(
                question_sha256=question_identity(question),
                pool=question.workload_class,
                easiness=easiness,
                difficulty=1 - easiness,
                decile=0,
                treatment_means=means,
                treatment_ranges=ranges,
                observations=expected,
                tied_questions=1,
            )
            pools[question.workload_class].append(qid)
        reports = {}
        if not labels:
            raise ValueError("calibration has no included questions")
        for pool, ids in sorted(pools.items()):
            values = [labels[qid].easiness for qid in ids]
            if len(ids) < plan.minimum_pool_size or len(set(values)) < 2:
                raise ValueError(f"{pool}: pool too small or all scores tied")
            ordered = sorted(
                ids,
                key=lambda qid: (
                    labels[qid].easiness,
                    digest({"seed": plan.seed, "question_id": qid}),
                    qid,
                ),
            )
            for index, qid in enumerate(ordered):
                labels[qid] = labels[qid].model_copy(
                    update={
                        "decile": 9 - min(9, index * 10 // len(ordered)),
                        "tied_questions": values.count(labels[qid].easiness),
                    }
                )
            reports[pool] = {
                "question_ids": sorted(ids),
                "distinct_scores": len(set(values)),
                "hard_count": sum(labels[qid].decile == 9 for qid in ids),
                "tie_broken_hard_boundary": len(ordered) > 1
                and any(
                    labels[a].decile == 9
                    and labels[b].decile < 9
                    and labels[a].easiness == labels[b].easiness
                    for a, b in zip(ordered, ordered[1:])
                ),
            }
    return DifficultyCalibrationResult(
        plan_sha256=plan_sha256,
        evidence_sha256=digest({"observations": [r.model_dump(mode="json") for r in rows]}),
        labels=labels,
        pools=reports,
        exclusions=exclusions,
    )


def validate_labels(questions: Sequence[Question], result: DifficultyCalibrationResult) -> None:
    """Reject stale identities, excluded items and substituted deciles or solve shares.

    A selected question's `panel_solve_share` must equal its label's easiness.
    """
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("calibrated selection requires unique questions")
    for question in questions:
        label = result.labels.get(question.question_id)
        if (
            label is None
            or label.question_sha256 != question_identity(question)
            or label.decile != question.difficulty_decile
            or label.easiness != question.panel_solve_share
        ):
            raise ValueError(
                f"{question.question_id}: missing, stale or substituted calibration label"
            )
