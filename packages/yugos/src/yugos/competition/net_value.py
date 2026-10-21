"""Net-value comparison of two routers on identical questions, with configurable answer values.

A category's net value is the challenger's extra correct answers, with hard answers
weighted, minus its extra spend expressed in answers at a configured dollar value
and, when the policy prices time, its extra time expressed in answers at a
configured duration. Everything here is a pure function of trusted measurements
and a frozen policy.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from statistics import NormalDist
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from .frozen import FrozenModel

if TYPE_CHECKING:
    from .rules import (
        CategoryMeasurement,
        CategoryRule,
        CompetitionPolicy,
        SubmissionMeasurement,
    )

# Decimal digits for every score, bound and floor. Replays must round identically.
PRECISION = 40
# Calibration computes panel solve shares with 28 digits; fractions such as "1/3"
# use the same precision so an exact one-third share sits on the threshold.
SHARE_PRECISION = 28

Name = Annotated[str, Field(min_length=1)]
Share = Annotated[Decimal, Field(ge=0, le=1)]
Positive = Annotated[Decimal, Field(gt=0)]


class Config(FrozenModel):
    """A deeply immutable net-value setting or result."""


class Tau(Config):
    """Floor tolerance: a share of the reference's remaining error, clamped to [min, max]."""

    share: Share = Decimal("0.25")
    min: Share = Decimal("0.02")
    max: Share = Decimal("0.05")

    @model_validator(mode="after")
    def ordered(self):
        if self.min > self.max:
            raise ValueError("tau min must not exceed tau max")
        return self


class Floor(Config):
    """Which quality a challenger must stay near, and how champions raise it."""

    basis: Literal["high_water_mark", "incumbent", "none"] = "high_water_mark"
    use_lower_bound: bool = Field(default=True, strict=True)
    tau: Tau = Field(default_factory=Tau)
    hard_floor_min_questions: int = Field(default=100, ge=1, strict=True)


class Bootstrap(Config):
    samples: int = Field(default=2000, ge=1000, le=10000, strict=True)
    seed: int = Field(default=0, ge=0, strict=True)


class TimeLimit(Config):
    multiplier: Positive = Decimal("2.0")


class CategoryValue(Config):
    """One category's score weight, answer value and optional overrides."""

    weight: Annotated[Decimal, Field(gt=0, le=1)]
    value_regular_usd: Positive | None = None
    hard_multiplier: Positive | None = None
    time_multiplier: Positive | None = None
    time_budget_seconds: Positive | None = None
    tau: Tau | None = None

    @model_validator(mode="after")
    def one_time_rule(self):
        if self.time_multiplier is not None and self.time_budget_seconds is not None:
            raise ValueError("set either time_multiplier or time_budget_seconds, not both")
        return self


def _default_categories() -> dict[str, CategoryValue]:
    return {
        "chat_qa": CategoryValue(weight=Decimal("0.40"), value_regular_usd=Decimal("1.00")),
        "math_code": CategoryValue(weight=Decimal("0.35"), value_regular_usd=Decimal("1.00")),
        "agentic": CategoryValue(weight=Decimal("0.25"), value_regular_usd=Decimal("10.00")),
    }


class NetValuePolicy(Config):
    """Every tunable number in the net-value rule. It is part of the policy digest."""

    value_basis: Literal["fixed", "anchor"] = "fixed"
    m: Positive = Decimal(30)
    # A regular answer is worth k times the anchor's mean time per question.
    # Unset, time is only a gate and is not priced.
    k: Positive | None = None
    hard_multiplier: Positive = Decimal(2)
    # Hard = no difficulty-panel model solved the question (e.g. both of two cheap models failed).
    hard_max_solve_share: Share = Decimal(0)
    min_net_value: Annotated[Decimal, Field(ge=0)] = Decimal("0.2")
    confidence: Annotated[Decimal, Field(gt=Decimal("0.5"), lt=1)] = Decimal("0.95")
    bootstrap: Bootstrap = Field(default_factory=Bootstrap)
    floor: Floor = Field(default_factory=Floor)
    time: TimeLimit = Field(default_factory=TimeLimit)
    categories: dict[Name, CategoryValue] = Field(default_factory=_default_categories, min_length=1)

    @field_validator("hard_max_solve_share", mode="before")
    @classmethod
    def fraction(cls, value):
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", 1)
            try:
                with localcontext() as context:
                    context.prec = SHARE_PRECISION
                    return Decimal(numerator.strip()) / Decimal(denominator.strip())
            except ArithmeticError as exc:
                raise ValueError(f"invalid fraction {value!r}") from exc
        return value

    @model_validator(mode="after")
    def coherent(self):
        if sum((c.weight for c in self.categories.values()), Decimal(0)) != 1:
            raise ValueError("category weights must sum to exactly one")
        values = [c.value_regular_usd for c in self.categories.values()]
        if self.value_basis == "fixed" and None in values:
            raise ValueError("fixed value basis requires value_regular_usd for every category")
        if self.value_basis == "anchor" and any(v is not None for v in values):
            raise ValueError("anchor value basis derives values; remove value_regular_usd")
        return self

    @model_serializer(mode="wrap")
    def omit_unpriced_time(self, handler: SerializerFunctionWrapHandler):
        # Policies created before time was priced keep their digests.
        data = handler(self)
        if self.k is None:
            data.pop("k", None)
        return data

    @property
    def weights(self) -> dict[str, Decimal]:
        return {name: category.weight for name, category in self.categories.items()}

    def hard_multiplier_for(self, category: str) -> Decimal:
        return self.categories[category].hard_multiplier or self.hard_multiplier

    def time_multiplier_for(self, category: str) -> Decimal:
        return self.categories[category].time_multiplier or self.time.multiplier

    def time_limit_ns(self, category: str, panel_leader_time_ns: int) -> Decimal:
        """An absolute per-category budget when configured, else a multiple of the panel leader."""
        budget = self.categories[category].time_budget_seconds
        if budget is not None:
            return budget * Decimal(1_000_000_000)
        return self.time_multiplier_for(category) * panel_leader_time_ns

    def tau_for(self, category: str) -> Tau:
        return self.categories[category].tau or self.floor.tau


def load_net_value_policy(raw: Mapping[str, Any]) -> NetValuePolicy:
    """Validate a plain mapping, such as parsed YAML or JSON, into a frozen policy."""
    return NetValuePolicy.model_validate(dict(raw))


def question_value(
    policy: NetValuePolicy, category: str, anchor_cost_per_question: Decimal | None = None
) -> Decimal:
    """Dollar value of one regular correct answer in a category (V_c)."""
    if policy.value_basis == "fixed":
        return policy.categories[category].value_regular_usd
    if anchor_cost_per_question is None or anchor_cost_per_question <= 0:
        raise ValueError(f"{category}: anchor value basis requires a positive anchor cost")
    with localcontext() as context:
        context.prec = PRECISION
        return policy.m * anchor_cost_per_question


def time_value(
    policy: NetValuePolicy, anchor_time_per_question_ns: Decimal | None = None
) -> Decimal | None:
    """Working time, in nanoseconds, worth one regular correct answer (T_c); none when unpriced."""
    if policy.k is None:
        return None
    if anchor_time_per_question_ns is None or anchor_time_per_question_ns <= 0:
        raise ValueError("priced time requires a positive anchor time per question")
    with localcontext() as context:
        context.prec = PRECISION
        return policy.k * anchor_time_per_question_ns


def tau(config: Tau, quality: Decimal) -> Decimal:
    """Allowed quality shortfall below a floor reference of the given quality."""
    with localcontext() as context:
        context.prec = PRECISION
        return min(config.max, max(config.min, config.share * (1 - quality)))


def _mean(values: Sequence[Decimal]) -> Decimal:
    with localcontext() as context:
        context.prec = PRECISION
        return sum(values, Decimal(0)) / len(values)


def _z(confidence: Decimal) -> Decimal:
    # Rounded so a last-digit libm difference cannot change a replayed floor.
    return Decimal(repr(NormalDist().inv_cdf(float(confidence)))).quantize(Decimal("1e-9"))


def quality_lower_bound(grades: Sequence[Decimal], confidence: Decimal) -> Decimal:
    """One-sided normal lower bound on mean quality: mean - z * sd / sqrt(n), at least zero."""
    if not grades:
        raise ValueError("quality requires at least one grade")
    mean = _mean(grades)
    if len(grades) < 2:
        return mean
    with localcontext() as context:
        context.prec = PRECISION
        variance = sum(((g - mean) ** 2 for g in grades), Decimal(0)) / (len(grades) - 1)
        return max(
            Decimal(0), mean - _z(confidence) * variance.sqrt() / Decimal(len(grades)).sqrt()
        )


class CategoryNetValue(Config):
    """How a category's net value was reached, in quality points (100 = every answer)."""

    n: int
    n_hard: int
    value_usd: Decimal
    hard_multiplier: Decimal
    regular_answer_delta: Decimal
    hard_answer_delta: Decimal
    spend_delta_usd: Decimal
    quality_points: Decimal
    spend_points: Decimal
    # Set only when the policy prices time.
    time_value_ns: Decimal | None = None
    time_delta_ns: int | None = None
    time_points: Decimal | None = None
    net_value: Decimal

    @model_serializer(mode="wrap")
    def omit_unpriced_time(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if self.time_value_ns is None:
            for name in ("time_value_ns", "time_delta_ns", "time_points"):
                data.pop(name, None)
        return data


def _paired(
    challenger: CategoryMeasurement,
    incumbent: CategoryMeasurement,
    question_ids: Sequence[str],
    hard_question_ids: Sequence[str],
    priced_time: bool = False,
):
    if priced_time and (challenger.question_time_ns is None or incumbent.question_time_ns is None):
        raise ValueError("priced time requires per-question time from both routers")
    hard = set(hard_question_ids)
    return [
        (
            qid in hard,
            challenger.grades[qid] - incumbent.grades[qid],
            challenger.question_spend_usd[qid] - incumbent.question_spend_usd[qid],
            challenger.question_time_ns[qid] - incumbent.question_time_ns[qid]
            if priced_time
            else 0,
        )
        for qid in question_ids
    ]


def category_net_value(
    challenger: CategoryMeasurement,
    incumbent: CategoryMeasurement,
    *,
    question_ids: Sequence[str],
    hard_question_ids: Sequence[str],
    value_usd: Decimal,
    hard_multiplier: Decimal,
    time_value_ns: Decimal | None = None,
) -> CategoryNetValue:
    """NV_c = 100 * sum(u_i * dGrade_i - dSpend_i / V - dTime_i / T) / n, on identical questions.

    The time term applies only when the policy prices time (`time_value_ns` is T).
    """
    priced = time_value_ns is not None
    with localcontext() as context:
        context.prec = PRECISION
        rows = _paired(challenger, incumbent, question_ids, hard_question_ids, priced)
        regular = sum((grade for hard, grade, _, _ in rows if not hard), Decimal(0))
        hard = sum((grade for is_hard, grade, _, _ in rows if is_hard), Decimal(0))
        spend = sum((cost for _, _, cost, _ in rows), Decimal(0))
        elapsed = sum(time for _, _, _, time in rows)
        n = len(rows)
        quality_points = 100 * (regular + hard_multiplier * hard) / n
        spend_points = 100 * spend / value_usd / n
        time_points = 100 * Decimal(elapsed) / time_value_ns / n if priced else None
        return CategoryNetValue(
            n=n,
            n_hard=sum(is_hard for is_hard, _, _, _ in rows),
            value_usd=value_usd,
            hard_multiplier=hard_multiplier,
            regular_answer_delta=regular,
            hard_answer_delta=hard,
            spend_delta_usd=spend,
            quality_points=quality_points,
            spend_points=spend_points,
            time_value_ns=time_value_ns,
            time_delta_ns=elapsed if priced else None,
            time_points=time_points,
            net_value=quality_points - spend_points - (time_points or 0),
        )


def question_net_values(
    challenger: CategoryMeasurement,
    incumbent: CategoryMeasurement,
    *,
    question_ids: Sequence[str],
    hard_question_ids: Sequence[str],
    value_usd: Decimal,
    hard_multiplier: Decimal,
    time_value_ns: Decimal | None = None,
) -> tuple[Decimal, ...]:
    """Each question's contribution, 100 * (u_i * dGrade_i - dSpend_i / V - dTime_i / T).

    Their mean is NV_c. The time term applies only when `time_value_ns` is set.
    """
    priced = time_value_ns is not None
    rows = _paired(challenger, incumbent, question_ids, hard_question_ids, priced)
    with localcontext() as context:
        context.prec = PRECISION
        return tuple(
            100
            * (
                (hard_multiplier if hard else 1) * grade
                - cost / value_usd
                - (Decimal(time) / time_value_ns if priced else 0)
            )
            for hard, grade, cost, time in rows
        )


def net_value(
    weights: Mapping[str, Decimal], categories: Mapping[str, CategoryNetValue]
) -> Decimal:
    """Overall NV = sum of weight_c * NV_c over every weighted category."""
    if set(weights) != set(categories):
        raise ValueError("net value requires every weighted category")
    with localcontext() as context:
        context.prec = PRECISION
        return sum((weights[c] * categories[c].net_value for c in sorted(weights)), Decimal(0))


def lower_bound(
    strata: Mapping[str, tuple[Decimal, Sequence[Decimal]]],
    *,
    bootstrap: Bootstrap,
    confidence: Decimal,
) -> Decimal:
    """Paired bootstrap percentile of weighted NV, resampling questions within each category.

    `strata` maps a category to its weight and per-question contributions. The seed
    fixes the draws; categories are visited in name order so the result is replayable.
    """
    names = sorted(strata)
    if not names or any(not strata[name][1] for name in names):
        raise ValueError("bootstrap requires questions in every category")
    rng = random.Random(bootstrap.seed)
    with localcontext() as context:
        context.prec = PRECISION
        draws = []
        for _ in range(bootstrap.samples):
            total = Decimal(0)
            for name in names:
                weight, values = strata[name]
                sample = rng.choices(values, k=len(values))
                total += weight * sum(sample, Decimal(0)) / len(values)
            draws.append(total)
        draws.sort()
        index = int(bootstrap.samples * (1 - confidence))
        return draws[max(index - 1, 0)]


class HighWaterMark(Config):
    """Highest champion quality so far in one category, and on its hard and private subsets."""

    quality: Decimal
    hard_quality: Decimal | None = None
    private_quality: Decimal | None = None


def _champion_quality(grades: Sequence[Decimal], policy: NetValuePolicy) -> Decimal:
    if policy.floor.use_lower_bound:
        return quality_lower_bound(grades, policy.confidence)
    return _mean(grades)


def high_water_marks(
    policy: CompetitionPolicy, champions: Sequence[SubmissionMeasurement]
) -> dict[str, HighWaterMark]:
    """Replay promoted champions, oldest first, over the frozen panel-leader starting marks.

    A mark only rises. With `floor.use_lower_bound`, a champion contributes the lower
    confidence bound of its quality rather than its mean.
    """
    config = policy.net_value
    marks = {
        rule.category: HighWaterMark(
            quality=rule.initial_quality,
            hard_quality=rule.initial_hard_quality,
            private_quality=rule.initial_private_quality,
        )
        for rule in policy.categories
    }
    for champion in champions:
        rows = champion.by_category()
        for rule in policy.categories:
            row, mark = rows[rule.category], marks[rule.category]
            quality = _champion_quality([row.grades[q] for q in rule.question_ids], config)

            def risen(current: Decimal | None, ids: Sequence[str]) -> Decimal | None:
                if not ids:
                    return current
                return max(current, _champion_quality([row.grades[q] for q in ids], config))

            marks[rule.category] = HighWaterMark(
                quality=max(mark.quality, quality),
                hard_quality=risen(mark.hard_quality, rule.hard_question_ids),
                private_quality=risen(mark.private_quality, rule.private_question_ids),
            )
    return marks


class CategoryGates(Config):
    """The floors and limit one category was held to, and which it failed."""

    quality: Decimal
    quality_reference: Decimal | None
    quality_floor: Decimal | None
    hard_quality: Decimal | None
    hard_quality_reference: Decimal | None
    hard_quality_floor: Decimal | None
    private_quality: Decimal | None
    private_quality_reference: Decimal | None
    private_quality_floor: Decimal | None
    execution_time_ns: int
    time_limit_ns: Decimal
    failures: tuple[str, ...]


def category_gates(
    policy: NetValuePolicy,
    rule: CategoryRule,
    run: CategoryMeasurement,
    mark: HighWaterMark,
    incumbent: CategoryMeasurement | None,
) -> CategoryGates:
    """Apply the time, quality, hard-subset and private-subset gates to complete evidence."""
    with localcontext() as context:
        context.prec = PRECISION
        category = rule.category
        tolerance = policy.tau_for(category)
        failures = []
        limit = policy.time_limit_ns(category, rule.panel_leader_time_ns)
        if run.execution_time_ns > limit:
            failures.append("category time limit exceeded")

        def floor(reference: Decimal | None) -> Decimal | None:
            return None if reference is None else reference - tau(tolerance, reference)

        def subset(ids: Sequence[str]) -> Decimal | None:
            return _mean([run.grades[q] for q in ids]) if ids else None

        def reference(ids: Sequence[str], marked: Decimal | None) -> Decimal | None:
            if policy.floor.basis == "none":
                return None
            if policy.floor.basis == "incumbent" and incumbent is not None:
                return _mean([incumbent.grades[q] for q in ids])
            return marked

        quality = subset(rule.question_ids)
        quality_reference = reference(rule.question_ids, mark.quality)
        hard_quality = subset(rule.hard_question_ids)
        hard_reference = None
        if len(rule.hard_question_ids) >= policy.floor.hard_floor_min_questions:
            hard_reference = reference(rule.hard_question_ids, mark.hard_quality)
        private_quality = subset(rule.private_question_ids)
        private_reference = None
        if rule.private_question_ids:
            private_reference = reference(rule.private_question_ids, mark.private_quality)
        private_floor = floor(private_reference)
        checks = (
            ("quality below floor", quality, floor(quality_reference)),
            ("hard-question quality below floor", hard_quality, floor(hard_reference)),
            ("private-question quality below floor", private_quality, private_floor),
        )
        failures.extend(
            label
            for label, value, minimum in checks
            if minimum is not None and value is not None and value < minimum
        )
        return CategoryGates(
            quality=quality,
            quality_reference=quality_reference,
            quality_floor=floor(quality_reference),
            hard_quality=hard_quality,
            hard_quality_reference=hard_reference,
            hard_quality_floor=floor(hard_reference),
            private_quality=private_quality,
            private_quality_reference=private_reference,
            private_quality_floor=private_floor,
            execution_time_ns=run.execution_time_ns,
            time_limit_ns=limit,
            failures=tuple(failures),
        )
