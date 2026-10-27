"""Versioned category competitions over trusted measurements, without hosting or billing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from ..evaluation.grades import GradeCode
from ..spend import Spend, apply_legacy_spend
from .frozen import FrozenModel
from .net_value import (
    PRECISION,
    CategoryGates,
    CategoryNetValue,
    NetValuePolicy,
    category_gates,
    category_net_value,
    high_water_marks,
    lower_bound,
    net_value,
    question_net_values,
    question_value,
    time_value,
)

COMPETITION_VERSION = "yugos-overall-v2"
if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..evaluation import Question
    from ..evaluation.grades import GradingResult
    from ..evaluation.measurements import QuestionMeasurement
    from .net_value import HighWaterMark
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1)]
Quality = Annotated[Decimal, Field(ge=0, le=1)]


class Record(FrozenModel):
    """A deeply immutable competition record."""


def digest(value: BaseModel | dict) -> str:
    """Content identity for a JSON-compatible competition record."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class CategoryMeasurement(Record):
    """Executor-owned measurements; none of these fields are supplied by the miner."""

    run_id: Name
    miner_id: Name
    submission_sha256: Digest
    cohort_sha256: Digest
    category: Name
    grades: dict[Name, Quality] = Field(min_length=1)
    # Spend attributed to each assigned question by its calls' question IDs.
    question_spend_usd: dict[Name, Annotated[Decimal, Field(ge=0)]]
    cost_usd: Decimal = Field(ge=0)
    execution_time_ns: int = Field(ge=0, strict=True)
    # Working time attributed to each assigned question; required when time is priced.
    question_time_ns: dict[Name, Annotated[int, Field(ge=0, strict=True)]] | None = None
    # Retained labels are opaque evidence; new measurements do not emit one.
    timing_version: str | None = None
    spend: Spend
    execution_complete: bool = Field(strict=True)
    integrity_pass: bool = Field(strict=True)
    # Evidence loaded from cost_complete keeps that commitment. New records use spend.
    _legacy_evidence: bool = PrivateAttr(default=False)
    _legacy_reservation: str | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def load_legacy_cost(cls, data: Any, handler):
        legacy = isinstance(data, dict) and "spend" not in data and "cost_complete" in data
        reservation = None
        if legacy and "reservation_usd" in data:
            reservation = data["reservation_usd"]
            reservation = reservation if isinstance(reservation, str) else str(reservation)
        if isinstance(data, dict):
            data = apply_legacy_spend(data, amount_key="cost_usd", drop_amount=False)
        model = handler(data)
        if legacy:
            object.__setattr__(model, "_legacy_evidence", True)
            object.__setattr__(model, "_legacy_reservation", reservation)
        return model

    @model_serializer(mode="wrap")
    def omit_unset_retained_label(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if self.timing_version is None:
            data.pop("timing_version", None)
        if self.question_time_ns is None:
            data.pop("question_time_ns", None)
        if self._legacy_evidence:
            # Replay hashes the stored document. Emitting spend would change it.
            data.pop("spend", None)
            data["cost_complete"] = self.spend.status == "known"
            if self.spend.status != "known" and self._legacy_reservation is not None:
                data["reservation_usd"] = self._legacy_reservation
        return data

    @model_validator(mode="after")
    def spend_is_attributed(self):
        if set(self.question_spend_usd) != set(self.grades):
            raise ValueError("question spend must cover exactly the graded questions")
        with localcontext() as context:
            context.prec = PRECISION
            if sum(self.question_spend_usd.values(), Decimal(0)) != self.cost_usd:
                raise ValueError("question spend must sum to the category cost")
        if self.spend.status == "known" and self.spend.usd != self.cost_usd:
            raise ValueError("known spend must equal the category cost")
        if self.question_time_ns is not None and (
            set(self.question_time_ns) != set(self.grades)
            or sum(self.question_time_ns.values()) != self.execution_time_ns
        ):
            raise ValueError(
                "question time must cover the graded questions and sum to category time"
            )
        return self

    @property
    def quality(self) -> Decimal:
        return sum(self.grades.values(), Decimal(0)) / len(self.grades)


def spend_error(run: CategoryMeasurement) -> str | None:
    """The one reason an unresolved bill cannot be compared."""
    if run.spend.status != "known":
        return "unresolved"
    return None


def competition_grade(measurement: QuestionMeasurement | None) -> Decimal:
    """A question's grade in comparisons: its score, or zero when missing or a non-answer."""
    if measurement is None or measurement.non_answer:
        return Decimal(0)
    return Decimal(str(measurement.outcome.score))


def grade_settled(grade: GradingResult | None) -> bool:
    """Whether a question's grade leaves its category comparable.

    A failed attempt has no grade and scores zero; truncation is a graded zero.
    Unavailable or errored grading leaves the category incomplete.
    """
    return grade is None or grade.status == "graded" or grade.code == GradeCode.TRUNCATED


def derive_hard_question_ids(
    questions: Sequence[Question], *, max_solve_share: Decimal
) -> dict[str, tuple[str, ...]]:
    """Freeze questions the difficulty panel solved at most `max_solve_share` of, per category.

    This is a threshold, not a percentile: questions with equal solve shares are
    always on the same side. A category may have no hard questions; the hard
    multiplier and hard-question floor then have no effect there.
    """
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("hard-tier selection requires unique assigned questions")
    selected: dict[str, list[str]] = {}
    for question in questions:
        if question.panel_solve_share is None:
            raise ValueError(f"missing panel solve share for {question.question_id}")
        ids = selected.setdefault(question.workload_class, [])
        if question.panel_solve_share <= max_solve_share:
            ids.append(question.question_id)
    return {category: tuple(sorted(ids)) for category, ids in sorted(selected.items())}


class CategoryRule(Record):
    """One category's frozen questions and panel-leader starting points."""

    category: Name
    cohort_sha256: Digest
    question_ids: tuple[Name, ...] = Field(min_length=1)
    hard_question_ids: tuple[Name, ...] = ()
    private_question_ids: tuple[Name, ...] = ()
    # The difficulty panel leader's measured quality; the high-water marks start here.
    initial_quality: Quality
    initial_hard_quality: Quality | None = None
    initial_private_quality: Quality | None = None
    panel_leader_time_ns: int = Field(gt=0, strict=True)
    # The anchor model's mean spend per question, for the "anchor" value basis.
    anchor_cost_per_question: Annotated[Decimal, Field(gt=0)] | None = None
    # The anchor model's mean working time per question, for priced time (`k`).
    anchor_time_per_question_ns: Annotated[Decimal, Field(gt=0)] | None = None
    # Preserve old policy digests without using the retired label in comparisons.
    timing_version: str | None = None

    @model_serializer(mode="wrap")
    def omit_unset_retained_label(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if self.timing_version is None:
            data.pop("timing_version", None)
        if self.anchor_time_per_question_ns is None:
            data.pop("anchor_time_per_question_ns", None)
        return data

    @model_validator(mode="after")
    def validate_subsets(self):
        expected = set(self.question_ids)
        if len(expected) != len(self.question_ids):
            raise ValueError("duplicate question IDs")
        for subset in (self.hard_question_ids, self.private_question_ids):
            if len(set(subset)) != len(subset) or not set(subset) <= expected:
                raise ValueError("invalid quality-check question IDs")
        for ids, initial, label in (
            (self.hard_question_ids, self.initial_hard_quality, "hard"),
            (self.private_question_ids, self.initial_private_quality, "private"),
        ):
            if bool(ids) != (initial is not None):
                raise ValueError(f"{label} questions and initial {label} quality go together")
        return self

    def measurement_errors(self, run: CategoryMeasurement) -> tuple[str, ...]:
        errors = []
        if (run.category, run.cohort_sha256) != (self.category, self.cohort_sha256):
            errors.append("incompatible cohort")
        if set(run.grades) != set(self.question_ids):
            errors.append("assigned question coverage differs")
        if not run.execution_complete:
            errors.append("execution incomplete")
        error = spend_error(run)
        if error is not None:
            errors.append(error)
        if not run.integrity_pass:
            errors.append("integrity check failed")
        if run.cost_usd <= 0 or run.execution_time_ns <= 0:
            errors.append("positive measured cost and time required")
        return tuple(errors)


class SubmissionMeasurement(Record):
    """One submission run, with a measurement for every category it attempted."""

    categories: tuple[CategoryMeasurement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def same_submission(self):
        if len({row.category for row in self.categories}) != len(self.categories):
            raise ValueError("duplicate category measurement")
        if (
            len(
                {
                    (row.run_id, row.miner_id, row.submission_sha256, row.cohort_sha256)
                    for row in self.categories
                }
            )
            != 1
        ):
            raise ValueError("category measurements must belong to one submission run")
        return self

    @property
    def run_id(self) -> str:
        return self.categories[0].run_id

    @property
    def miner_id(self) -> str:
        return self.categories[0].miner_id

    def by_category(self) -> dict[str, CategoryMeasurement]:
        return {row.category: row for row in self.categories}


class CompetitionPolicy(Record):
    """One competition: frozen categories and the net-value rule that compares routers."""

    version: Literal["yugos-overall-v2"] = COMPETITION_VERSION
    categories: tuple[CategoryRule, ...] = Field(min_length=1)
    net_value: NetValuePolicy = Field(default_factory=NetValuePolicy)

    @model_validator(mode="after")
    def coherent_categories(self):
        names = [rule.category for rule in self.categories]
        if len(set(names)) != len(names) or set(names) != set(self.net_value.categories):
            raise ValueError("net-value categories must match each category rule exactly once")
        if len({rule.cohort_sha256 for rule in self.categories}) != 1:
            raise ValueError("categories must use the same cohort")
        question_ids = [qid for rule in self.categories for qid in rule.question_ids]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("assigned questions cannot appear in multiple categories")
        if self.net_value.value_basis == "anchor" and any(
            rule.anchor_cost_per_question is None for rule in self.categories
        ):
            raise ValueError("anchor value basis requires an anchor cost in every category")
        if self.net_value.k is not None and any(
            rule.anchor_time_per_question_ns is None for rule in self.categories
        ):
            raise ValueError("priced time requires an anchor time in every category")
        return self

    def evidence_errors(self, run: SubmissionMeasurement) -> tuple[str, ...]:
        """Evidence problems that make a submission incomparable, before any gate."""
        measured = run.by_category()
        if set(measured) != set(self.net_value.categories):
            return ("submission must cover every category",)
        timed = self.net_value.k is None or all(
            row.question_time_ns is not None for row in measured.values()
        )
        return tuple(
            f"{rule.category}: {error}"
            for rule in self.categories
            for error in rule.measurement_errors(measured[rule.category])
        ) + (() if timed else ("priced time requires per-question time",))

    def question_value(self, rule: CategoryRule) -> Decimal:
        return question_value(self.net_value, rule.category, rule.anchor_cost_per_question)

    def time_value(self, rule: CategoryRule) -> Decimal | None:
        return time_value(self.net_value, rule.anchor_time_per_question_ns)


class CategoryDecision(Record):
    gates: CategoryGates
    net_value: CategoryNetValue | None = None


class ChallengeDecision(Record):
    policy_sha256: Digest
    challenger_sha256: Digest
    incumbent_sha256: Digest | None
    eligible: bool
    promoted: bool
    categories: dict[Name, CategoryDecision] = Field(default_factory=dict)
    net_value: Decimal | None = None
    lower_bound: Decimal | None = None
    min_net_value: Decimal | None = None
    reasons: tuple[str, ...]


def decide_challenge(
    policy: CompetitionPolicy,
    challenger: SubmissionMeasurement,
    champions: Sequence[SubmissionMeasurement] = (),
) -> ChallengeDecision:
    """Gate every category, then promote when the net-value lower bound clears the minimum.

    `champions` is the promoted lineage, oldest first; its last entry is the incumbent
    and the whole lineage sets the high-water-mark floors.
    """
    incumbent = champions[-1] if champions else None
    fields = dict(
        policy_sha256=digest(policy),
        challenger_sha256=digest(challenger),
        incumbent_sha256=digest(incumbent) if incumbent else None,
    )
    errors = policy.evidence_errors(challenger)
    if errors:
        return ChallengeDecision(**fields, eligible=False, promoted=False, reasons=errors)
    if incumbent is not None and policy.evidence_errors(incumbent):
        raise ValueError("incumbent evidence is not comparable under this frozen policy")
    config = policy.net_value
    marks = high_water_marks(policy, champions)
    ours = challenger.by_category()
    theirs = incumbent.by_category() if incumbent else {}
    categories, strata = {}, {}
    for rule in policy.categories:
        name = rule.category
        gates = category_gates(config, rule, ours[name], marks[name], theirs.get(name))
        value = None
        if incumbent is not None:
            terms = dict(
                question_ids=rule.question_ids,
                hard_question_ids=rule.hard_question_ids,
                value_usd=policy.question_value(rule),
                hard_multiplier=config.hard_multiplier_for(name),
                time_value_ns=policy.time_value(rule),
            )
            value = category_net_value(ours[name], theirs[name], **terms)
            strata[name] = (
                config.categories[name].weight,
                question_net_values(ours[name], theirs[name], **terms),
            )
        categories[name] = CategoryDecision(gates=gates, net_value=value)
    failures = tuple(
        f"{name}: {failure}" for name, row in categories.items() for failure in row.gates.failures
    )
    if incumbent is None:
        return ChallengeDecision(
            **fields,
            eligible=not failures,
            promoted=not failures,
            categories=categories,
            reasons=failures or ("first eligible overall champion",),
        )
    total = net_value(config.weights, {name: row.net_value for name, row in categories.items()})
    bound = None
    if not failures:
        bound = lower_bound(strata, bootstrap=config.bootstrap, confidence=config.confidence)
    promoted = bound is not None and bound > config.min_net_value
    if failures:
        reasons = failures
    elif promoted:
        reasons = ("net-value lower bound exceeds the minimum",)
    else:
        reasons = ("net-value lower bound does not exceed the minimum",)
    return ChallengeDecision(
        **fields,
        eligible=not failures,
        promoted=promoted,
        categories=categories,
        net_value=total,
        lower_bound=bound,
        min_net_value=config.min_net_value,
        reasons=reasons,
    )


class Challenge(Record):
    measurement: SubmissionMeasurement
    decision: ChallengeDecision


def _same_content(ours: Record, theirs: Record) -> bool:
    """Exact content identity. Value equality is not enough: Decimal equality ignores
    spelling such as "0.95" versus "0.950", but digests and stored decisions do not."""
    return ours is theirs or digest(ours) == digest(theirs)


class CompetitionHistory(Record):
    """Decisions on one pack, optionally seeded by a rebenchmarked incumbent."""

    policy: CompetitionPolicy
    challenges: tuple[Challenge, ...] = ()
    baseline: SubmissionMeasurement | None = None
    prior_kings: tuple[SubmissionMeasurement, ...] = Field(default=(), max_length=3)

    @model_serializer(mode="wrap")
    def retain_legacy_identity(self, handler):
        data = handler(self)
        if self.baseline is None:
            data.pop("baseline", None)
        if not self.prior_kings:
            data.pop("prior_kings", None)
        return data

    @model_validator(mode="after")
    def baseline_rebenchmarks_incumbent(self):
        if self.baseline is None:
            if self.prior_kings:
                raise ValueError("inherited kings require a comparison baseline")
            return self
        if not self.prior_kings or self.policy.evidence_errors(self.baseline):
            raise ValueError("baseline requires complete evidence and prior reward seats")
        king = self.prior_kings[-1]
        if (
            self.baseline.miner_id != king.miner_id
            or self.baseline.categories[0].submission_sha256
            != king.categories[0].submission_sha256
            or self.baseline.run_id in {row.run_id for row in self.prior_kings}
        ):
            raise ValueError("baseline must rebenchmark the incumbent submission")
        return self

    def _promoted(self) -> tuple[SubmissionMeasurement, ...]:
        return tuple(event.measurement for event in self.challenges if event.decision.promoted)

    def lineage(self) -> tuple[SubmissionMeasurement, ...]:
        """Reward-seat kings from oldest to newest, as recorded."""
        return (*self.prior_kings, *self._promoted())

    def champion(self) -> SubmissionMeasurement | None:
        kings = self.comparison_lineage()
        return kings[-1] if kings else None

    def comparison_lineage(self) -> tuple[SubmissionMeasurement, ...]:
        """Measurements on the active pack, seeded by its incumbent baseline."""
        baseline = (self.baseline,) if self.baseline else ()
        return (*baseline, *self._promoted())

    def high_water_marks(self) -> dict[str, HighWaterMark]:
        """Current category floors from the active pack's comparison lineage."""
        return high_water_marks(self.policy, self.comparison_lineage())

    def challenge(self, measurement: SubmissionMeasurement) -> CompetitionHistory:
        """Decide one new challenge against the current king and append it."""
        for event in self.challenges:
            if event.measurement.run_id == measurement.run_id:
                if not _same_content(event.measurement, measurement):
                    raise ValueError("challenge run conflicts with retained evidence")
                return self
        reserved = {row.run_id for row in self.prior_kings}
        if self.baseline:
            reserved.add(self.baseline.run_id)
        if measurement.run_id in reserved:
            raise ValueError("duplicate challenge run")
        decision = decide_challenge(self.policy, measurement, self.comparison_lineage())
        event = Challenge(measurement=measurement, decision=decision)
        return self.model_copy(update={"challenges": (*self.challenges, event)})


class CompetitionSnapshot(Record):
    """One overall champion with category breakdowns; shadow publication only."""

    schema_version: Literal["yugos-overall-shadow-v1"] = "yugos-overall-shadow-v1"
    shadow: Literal[True] = True
    should_set_weights: Literal[False] = False
    history: CompetitionHistory
    proposed_champion_run_id: str | None

    @classmethod
    def read(cls, raw: dict, *, policy_sha256: str) -> CompetitionSnapshot:
        """Check the digest and pinned policy."""
        if raw.get("id") != digest(raw.get("payload", {})):
            raise ValueError("competition snapshot digest differs")
        snapshot = cls.model_validate(raw["payload"])
        if digest(snapshot.history.policy) != policy_sha256:
            raise ValueError("snapshot differs from the pinned competition policy")
        return snapshot

    @model_validator(mode="after")
    def proposed_champion_matches(self):
        champion = self.history.champion()
        if self.proposed_champion_run_id != (champion.run_id if champion else None):
            raise ValueError("proposed overall champion does not match the history")
        return self


class CompetitionCohort(Record):
    """Validator configuration pins one policy and the hosted admission authority."""

    netuid: int = Field(gt=0, strict=True)
    policy_sha256: Digest
    scoring_version: Literal["yugos-overall-v2"] = COMPETITION_VERSION
    reward_rule: Literal["winner-take-all"] = "winner-take-all"
    submission_authority: Literal["signed-upload"] = "signed-upload"


# Current king, the king they replaced, and the king before that.
KING_SHARES = (Decimal("0.85"), Decimal("0.10"), Decimal("0.05"))


def king_weights(kings: Sequence[str]) -> dict[str, float]:
    """Pay the last three kings (oldest first in `kings`) 85%, 10%, and 5%.

    A seat that does not exist yet is not invented here. An empty older seat's
    share is added to the current king so the vector still sums to 1. The same
    hotkey keeps every seat it holds.
    """
    recent = list(reversed(kings[-3:]))
    if not recent:
        return {}
    shares = [KING_SHARES[index] for index in range(len(recent))]
    shares[0] += Decimal(1) - sum(shares, Decimal(0))
    weights: dict[str, Decimal] = {}
    for king, share in zip(recent, shares):
        weights[king] = weights.get(king, Decimal(0)) + share
    return {miner_id: float(share) for miner_id, share in weights.items()}


class CompetitionResult(Record):
    """Production result; the configured HTTPS host attests its evidence."""

    schema_version: Literal["yugos-overall-result-v2"] = "yugos-overall-result-v2"
    netuid: int = Field(gt=0, strict=True)
    history: CompetitionHistory
    reward_rule: Literal["winner-take-all"] = "winner-take-all"
    submission_authority: Literal["signed-upload"] = "signed-upload"
    paused: bool = Field(default=False, strict=True)

    def weights(self) -> dict[str, float]:
        """King weights (see `king_weights`); none while paused."""
        if self.paused:
            return {}
        return king_weights([king.miner_id for king in self.history.lineage()])


class CompetitionPublication(Record):
    """Fresh confirmation is separate from immutable result content and measurement age."""

    id: Digest
    confirmed_at: AwareDatetime
    payload: CompetitionResult

    @classmethod
    def read(
        cls,
        raw: dict,
        *,
        cohort: CompetitionCohort,
        now: datetime,
        max_age_s: float,
    ) -> CompetitionPublication:
        """Check identity, pinned policy, and freshness."""
        publication = cls.model_validate(raw)
        payload = publication.payload
        if digest(raw["payload"]) != publication.id:
            raise ValueError("competition publication digest differs")
        if (
            payload.netuid != cohort.netuid
            or payload.history.policy.version != cohort.scoring_version
            or digest(payload.history.policy) != cohort.policy_sha256
            or payload.reward_rule != cohort.reward_rule
            or payload.submission_authority != cohort.submission_authority
        ):
            raise ValueError("publication differs from the pinned competition policy")
        if not 0 <= (now - publication.confirmed_at).total_seconds() <= max_age_s:
            raise ValueError("competition confirmation is stale or dated in the future")
        return publication
