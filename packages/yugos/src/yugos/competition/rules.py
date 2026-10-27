"""Versioned category competitions over trusted measurements, without hosting or billing."""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidationInfo,
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


class _Replay(NamedTuple):
    """Verified state after a history's challenges: its promoted lineage and content keys."""

    policy_sha256: str
    lineage: tuple[SubmissionMeasurement, ...] = ()
    lineage_sha256: str = ""
    comparisons: tuple[SubmissionMeasurement, ...] = ()

    def advance(self, champion: SubmissionMeasurement, champion_sha256: str) -> _Replay:
        chain = digest({"previous": self.lineage_sha256, "champion": champion_sha256})
        return self._replace(
            lineage=(*self.lineage, champion), lineage_sha256=chain,
            comparisons=(*self.comparisons, champion),
        )


_CACHE_SIZE = 4096
_cache_lock = threading.Lock()
# Replayed decisions by content: policy, promoted lineage before the challenge, and challenger.
_decisions: OrderedDict[str, ChallengeDecision] = OrderedDict()
# Verified replay state by history object identity, so repeated reads skip re-hashing.
_replays: OrderedDict[int, tuple[weakref.ref, _Replay]] = OrderedDict()


def _lookup(cache: OrderedDict, key):
    with _cache_lock:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value


def _store(cache: OrderedDict, key, value):
    with _cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
    return value


def _replay_decision(
    policy: CompetitionPolicy, state: _Replay, challenger: SubmissionMeasurement, sha256: str
) -> ChallengeDecision:
    """Decide once per process for the same policy, lineage, and challenger content."""
    key = digest(
        {"policy": state.policy_sha256, "lineage": state.lineage_sha256, "challenger": sha256}
    )
    decision = _lookup(_decisions, key)
    if decision is None:
        decision = _store(_decisions, key, decide_challenge(policy, challenger, state.comparisons))
    return decision


def _same_content(ours: Record, theirs: Record) -> bool:
    """Exact content identity. Value equality is not enough: Decimal equality ignores
    spelling such as "0.95" versus "0.950", but digests and stored decisions do not."""
    return ours is theirs or digest(ours) == digest(theirs)


class PackRevision(Record):
    """An addition-only pack change with the same king and a fresh comparison baseline."""

    contest: Name
    previous_pack_sha256: Digest
    pack_sha256: Digest
    previous: CompetitionHistory
    baseline: SubmissionMeasurement

    def validate_transition(self, policy: CompetitionPolicy) -> None:
        previous = self.previous
        king = previous.champion()
        if king is None:
            raise ValueError("a pack revision requires an incumbent")
        if self.previous_pack_sha256 == self.pack_sha256:
            raise ValueError("a pack revision requires a different pack")
        if previous.pack_revision is not None and (
            previous.pack_revision.contest != self.contest
            or previous.pack_revision.pack_sha256 != self.previous_pack_sha256
        ):
            raise ValueError("pack revision does not extend the same contest and pack")
        self.validate_policy(previous.policy, policy)
        baseline = self.baseline
        if (baseline.miner_id != king.miner_id
                or baseline.categories[0].submission_sha256 != king.categories[0].submission_sha256
                or baseline.run_id in previous.run_ids()):
            raise ValueError("revision baseline must be a new run of the incumbent submission")
        if policy.evidence_errors(baseline):
            raise ValueError("revision baseline is incomplete or incompatible")
        for name, old in king.by_category().items():
            new = baseline.by_category()[name]
            if any(new.grades[q] != grade or new.question_spend_usd[q] != old.question_spend_usd[q]
                   for q, grade in old.grades.items()):
                raise ValueError("revision baseline rewrote retained grades or spend")
            if old.question_time_ns is not None and (
                new.question_time_ns is None
                or any(new.question_time_ns[q] != ns for q, ns in old.question_time_ns.items())
            ):
                raise ValueError("revision baseline rewrote retained question time")
            if new.execution_time_ns < old.execution_time_ns:
                raise ValueError("revision baseline lost retained execution time")

    @staticmethod
    def validate_policy(previous: CompetitionPolicy, policy: CompetitionPolicy) -> None:
        """Check addition-only coverage and unchanged scoring before a benchmark is funded."""
        if not _same_content(previous.net_value, policy.net_value):
            raise ValueError("a pack revision must preserve the scoring rule")
        before = {rule.category: rule for rule in previous.categories}
        after = {rule.category: rule for rule in policy.categories}
        if set(before) != set(after):
            raise ValueError("a pack revision must preserve categories")
        added = 0
        for name, old in before.items():
            new = after[name]
            kept = set(old.question_ids)
            if old.cohort_sha256 != new.cohort_sha256 or not kept <= set(new.question_ids):
                raise ValueError("a pack revision must retain the contest and all old questions")
            for subset in ("hard_question_ids", "private_question_ids"):
                if set(getattr(new, subset)) & kept != set(getattr(old, subset)):
                    raise ValueError("retained question subsets changed")
            added += len(set(new.question_ids) - kept)
        if not added:
            raise ValueError("a pack revision must add questions")


class CompetitionHistory(Record):
    """One ordered challenge history and one overall incumbent."""

    policy: CompetitionPolicy
    challenges: tuple[Challenge, ...] = ()
    pack_revision: PackRevision | None = None

    @model_serializer(mode="wrap")
    def retain_legacy_identity(self, handler):
        data = handler(self)
        if self.pack_revision is None:
            data.pop("pack_revision", None)
        return data

    def epochs(self) -> tuple[CompetitionHistory, ...]:
        """Frozen pack histories in their original scoring order."""
        prior = self.pack_revision.previous.epochs() if self.pack_revision else ()
        return (*prior, self)

    def run_ids(self) -> set[str]:
        """Every admitted challenge and baseline run across the frozen revisions."""
        result = set()
        for epoch in self.epochs():
            result.update(event.measurement.run_id for event in epoch.challenges)
            if epoch.pack_revision:
                result.add(epoch.pack_revision.baseline.run_id)
        return result

    def _initial_state(self, since=None, *, trusted=False) -> _Replay:
        if self.pack_revision is None:
            if since is not None:
                raise ValueError("history rolled back or rewrote the verified prefix")
            return _Replay(digest(self.policy))
        revision = self.pack_revision
        if trusted:
            revision.previous._remember(revision.previous._trusted_replay())
        else:
            revision.previous.verify(since=since)
        revision.validate_transition(self.policy)
        return _Replay(
            digest(self.policy), revision.previous.lineage(), digest(revision),
            (revision.baseline,),
        )

    def verify(self, *, since: CompetitionHistory | None = None) -> CompetitionHistory:
        """Replay every stored decision and return this history when they all match.

        `since` is a history the caller has already verified or otherwise trusts,
        such as the last snapshot a validator accepted. This history must repeat it
        exactly within its original policy, either in this epoch or a linked prior
        revision. Its stored decisions are trusted and only later challenges replay.
        Without it, verification starts from the first challenge. Records are deeply immutable,
        so the result is remembered for this object and later `lineage()`,
        `champion()`, and `high_water_marks()` calls do not replay again.
        """
        self._replay(since)
        return self

    def _replay(self, since: CompetitionHistory | None = None) -> _Replay:
        same_epoch = since is not None and _same_content(since.policy, self.policy)
        start = len(since.challenges) if same_epoch else 0
        same_revision = since is not None and (
            since.pack_revision is self.pack_revision
            or (since.pack_revision is not None and self.pack_revision is not None
                and _same_content(since.pack_revision, self.pack_revision))
        )
        if same_epoch and (
            not same_revision
            or len(self.challenges) < start
            or not all(map(_same_content, self.challenges, since.challenges))
        ):
            raise ValueError("history rolled back or rewrote the verified prefix")
        if since is not None and not same_epoch:
            self._initial_state(since)
        verified = self._verified()
        if verified is not None:
            return verified
        prior_ids = (
            self.pack_revision.previous.run_ids() | {self.pack_revision.baseline.run_id}
            if self.pack_revision else set()
        )
        run_ids = [event.measurement.run_id for event in self.challenges]
        if len(set(run_ids)) != len(run_ids) or prior_ids.intersection(run_ids):
            raise ValueError("duplicate challenge run")
        state = since._trusted_replay() if same_epoch else self._initial_state()
        for event in self.challenges[start:]:
            sha256 = digest(event.measurement)
            expected = _replay_decision(self.policy, state, event.measurement, sha256)
            if not _same_content(expected, event.decision):
                raise ValueError("challenge decision does not match replay")
            if expected.promoted:
                state = state.advance(event.measurement, sha256)
        return self._remember(state)

    def _trusted_replay(self) -> _Replay:
        """Replay state from stored decisions, reusing verification when this object has it."""
        verified = self._verified()
        if verified is not None:
            return verified
        state = self._initial_state(trusted=True)
        for event in self.challenges:
            if event.decision.promoted:
                state = state.advance(event.measurement, digest(event.measurement))
        return state

    def _verified(self) -> _Replay | None:
        entry = _lookup(_replays, id(self))
        return entry[1] if entry is not None and entry[0]() is self else None

    def _remember(self, state: _Replay) -> _Replay:
        # Drop states of collected histories so they do not keep their lineages alive.
        with _cache_lock:
            for key in [key for key, (ref, _) in _replays.items() if ref() is None]:
                del _replays[key]
        return _store(_replays, id(self), (weakref.ref(self), state))[1]

    def lineage(self) -> tuple[SubmissionMeasurement, ...]:
        """Promoted champions from oldest to newest. Stored decisions must replay."""
        return self._replay().lineage

    def champion(self) -> SubmissionMeasurement | None:
        kings = self.comparison_lineage()
        return kings[-1] if kings else None

    def comparison_lineage(self) -> tuple[SubmissionMeasurement, ...]:
        """Measurements on the active pack, seeded by its incumbent top-up."""
        return self._replay().comparisons

    def high_water_marks(self) -> dict[str, HighWaterMark]:
        """Current category floors, replayed from the promoted lineage."""
        return high_water_marks(self.policy, self.comparison_lineage())

    def challenge(self, measurement: SubmissionMeasurement) -> CompetitionHistory:
        """Append one decided challenge; only the new challenge is decided."""
        state = self._replay()
        for event in self.challenges:
            if event.measurement.run_id == measurement.run_id:
                if not _same_content(event.measurement, measurement):
                    raise ValueError("challenge run conflicts with retained evidence")
                return self
        sha256 = digest(measurement)
        decision = _replay_decision(self.policy, state, measurement, sha256)
        event = Challenge(measurement=measurement, decision=decision)
        if measurement.run_id in self.run_ids():
            raise ValueError("duplicate challenge run")
        history = self.model_copy(update={"challenges": (*self.challenges, event)})
        history._remember(state.advance(measurement, sha256) if decision.promoted else state)
        return history


class CompetitionSnapshot(Record):
    """One overall champion with category breakdowns; shadow publication only."""

    schema_version: Literal["yugos-overall-shadow-v1"] = "yugos-overall-shadow-v1"
    shadow: Literal[True] = True
    should_set_weights: Literal[False] = False
    history: CompetitionHistory
    proposed_champion_run_id: str | None

    @classmethod
    def read(
        cls, raw: dict, *, policy_sha256: str, since: CompetitionHistory | None = None
    ) -> CompetitionSnapshot:
        """Check the digest and pinned policy, then replay challenges after `since`."""
        if raw.get("id") != digest(raw.get("payload", {})):
            raise ValueError("competition snapshot digest differs")
        snapshot = cls.model_validate(raw["payload"], context={"verified_history": since})
        if digest(snapshot.history.policy) != policy_sha256:
            raise ValueError("snapshot differs from the pinned competition policy")
        return snapshot

    @model_validator(mode="after")
    def verify(self, info: ValidationInfo):
        since = (info.context or {}).get("verified_history")
        champion = self.history.verify(since=since).champion()
        if self.proposed_champion_run_id != (champion.run_id if champion else None):
            raise ValueError("proposed overall champion does not match replay")
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
    """Replayable production result; the configured HTTPS host attests its evidence."""

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
        since: CompetitionHistory | None = None,
    ) -> CompetitionPublication:
        """Check identity, pinned policy, and freshness, then replay challenges after `since`."""
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
        payload.history.verify(since=since)
        return publication
