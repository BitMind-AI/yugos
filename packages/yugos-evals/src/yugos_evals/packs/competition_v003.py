"""Synthetic competition pack ``competition-v0.0.3``.

Original questions in the three competition workloads: 100 ``chat_qa`` items
(five subjects of twenty), 12 ``math_code`` items, and 12 ``agentic`` items.
Counts stay large enough for a difficulty pool and are not padded to 25 per
benchmark.

Difficulty labels are the scores of scripted weak and mid answers, graded
locally (``grade_rubric_chat`` or ``grade_math``) and attributed to the
configured panel model ids (Venice DeepSeek V4 Flash at high effort, Gemini
3.8 Flash at medium effort). Those models were not called. The calibration
plan records ``collection=local-stand-in`` and ``inference=not-called``. A
strong scripted answer is kept for the local grading check and is not part
of the calibration panel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from inspect_ai import task_with
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelUsage

from yugos.competition import digest
from yugos.competition.difficulty import (
    CalibrationOutcome,
    CalibrationSpec,
    CalibrationSummary,
    DifficultyCalibrationPlan,
    Treatment,
    calculate,
    question_identity,
    verify_summary,
)
from yugos.evaluation import Question
from yugos.evaluation.packs import MANIFEST, PackManifest, canonical_json, pack_bytes, records_bytes
from yugos.inference.actions import ModelChoice
from yugos.inference.calls import CallLimits
from yugos.inspect_engine.frozen import evaluation_identity
from yugos.spend import KnownSpend
from yugos_evals.local.pricing import TokenPrices, usage_cost_microusd
from yugos_evals.prepared import prepare_task
from yugos_evals.scorers.math import grade_math
from yugos_evals.scorers.rubric_chat import contains_fact, grade_rubric_chat
from yugos_evals.tasks.boxed_math import boxed_math
from yugos_evals.tasks.rubric_chat import rubric_chat

PACK_ID = "competition-v0.0.3"
# Staging world card db6b3c56…, the card pinned for competition/v0.0.2.
WORLD_CARD_SHA256 = "db6b3c56d32faacfb94fde2af3ecc2e2f4c9c6e9e1b2fd2246f81de94c9f7525"
CHAT_SUBJECTS = (
    "quant_exact",
    "choice_logic",
    "rubric_facts",
    "procedure_steps",
    "structured_fields",
)
MATH_SUBJECT = "boxed_arithmetic"
AGENTIC_SUBJECT = "tool_trace"
SUBJECTS = (*CHAT_SUBJECTS, MATH_SUBJECT, AGENTIC_SUBJECT)
SUBJECT_COUNTS = {
    **dict.fromkeys(CHAT_SUBJECTS, 20),
    MATH_SUBJECT: 12,
    AGENTIC_SUBJECT: 12,
}
WORKLOAD = {
    **dict.fromkeys(CHAT_SUBJECTS, "chat_qa"),
    MATH_SUBJECT: "math_code",
    AGENTIC_SUBJECT: "agentic",
}
QUESTION_COUNT = sum(SUBJECT_COUNTS.values())
ROLES = ("weak", "mid", "strong")
# Panel order is the configured competition panel. Strong is local only.
PANEL = (
    ("weak", ModelChoice(provider_id="venice", model_id="deepseek-v4-flash-0731", effort="high")),
    ("mid", ModelChoice(provider_id="google", model_id="gemini-3.8-flash", effort="medium")),
)
# Public per-million token prices from that same world card.
PANEL_PRICES = {
    "deepseek-v4-flash-0731": {
        "input_usd_per_million": 0.13,
        "output_usd_per_million": 0.26,
        "cached_usd_per_million": 0.028,
    },
    "gemini-3.8-flash": {
        "input_usd_per_million": 0.75,
        "output_usd_per_million": 3.75,
        "cached_usd_per_million": 0.075,
    },
}
EXACT_SUBJECTS = frozenset({"quant_exact", "choice_logic"})


@dataclass(frozen=True)
class Item:
    """One original question and the three scripted answers used to label it."""

    subject: str
    sample_id: str
    prompt: str
    target: str
    metadata: dict
    answers: dict[str, str]

    @property
    def question_id(self) -> str:
        return f"{self.subject}:{self.sample_id}"


@dataclass(frozen=True)
class BuiltPack:
    """A verified pack plus the stand-in answers the labels were graded from."""

    archive: bytes
    sha256: str
    questions: tuple[Question, ...]
    items: tuple[Item, ...]
    summary: CalibrationSummary


def _numbers(index: int) -> tuple[int, int, int]:
    return 8 + index, 3 + (index % 7), 1 + (index % 5)


def _arithmetic(index: int) -> tuple[str, int, int, int]:
    """A short original word problem and three distinct integer results.

    The first integer is correct. The second is the mistake a weak answer
    makes. The third is a different mistake a mid answer makes on hard items.
    """
    left, right, adjust = _numbers(index)
    kind = index % 4
    if kind == 0:
        correct = left * right - adjust
        weak, mid = left * right, left * right - adjust - right
        prompt = (
            f"A dock received {left} crates with {right} radios in each crate. "
            f"Inspectors then pulled {adjust} radios out of the delivery. "
            "How many radios remain? Reply with the integer only."
        )
    elif kind == 1:
        correct = left + right - adjust
        weak, mid = left + right, left + right - 2 * adjust
        prompt = (
            f"The morning shift logged {left} tickets and the evening shift logged "
            f"{right} tickets. {adjust} tickets were logged by both shifts. "
            "How many distinct tickets were logged? Reply with the integer only."
        )
    elif kind == 2:
        correct = left * right + adjust
        weak, mid = left * right, left * (right + adjust)
        prompt = (
            f"A courier rides {left} kilometers each hour for {right} hours, then "
            f"walks another {adjust} kilometers. How many kilometers does the courier "
            "cover in all? Reply with the integer only."
        )
    else:
        correct = left * right - adjust
        weak, mid = left * right, (left - 1) * right - adjust
        prompt = (
            f"A shop sells {left} notebooks at {right} dollars each and then refunds "
            f"{adjust} dollars. What is the net amount in dollars? Reply with the integer only."
        )
    if len({correct, weak, mid}) != 3 or min(correct, weak, mid) <= 0:
        raise ValueError(f"arithmetic item {index} does not have three positive results")
    return prompt, correct, weak, mid


def _exact_answers(index: int, correct: str, weak: str, mid: str) -> dict[str, str]:
    if index < 6:
        return {role: correct for role in ROLES}
    if index < 13:
        return {"weak": weak, "mid": correct, "strong": correct}
    return {"weak": weak, "mid": mid, "strong": correct}


def _choice_item(index: int) -> Item:
    prompt, correct, weak, mid = _arithmetic(index)
    extra = correct + weak + mid + 7 + index
    values = (correct, weak, mid, extra)
    if len(set(values)) != 4:
        raise ValueError(f"choice item {index} has colliding options")
    letters = ("A", "B", "C", "D")
    order = letters[index % 4 :] + letters[: index % 4]
    by_letter = dict(zip(order, values, strict=True))
    listed = " ".join(f"{letter}) {by_letter[letter]}" for letter in letters)
    letter_of = {value: letter for letter, value in by_letter.items()}
    bare = _exact_answers(index, letter_of[correct], letter_of[weak], letter_of[mid])
    answers = {role: f"Answer: {letter}" for role, letter in bare.items()}
    body = (
        prompt.replace("Reply with the integer only.", "").strip()
        + " Which value is correct?\n"
        + listed
        + "\nPut your final answer on its own line in this exact form: Answer: L"
        + "\nReplace L with one letter: A, B, C, or D."
    )
    return Item(
        subject="choice_logic",
        sample_id=f"{index:02d}",
        prompt=body,
        target=letter_of[correct],
        metadata={"format": "choice"},
        answers=answers,
    )


def _quant_item(index: int) -> Item:
    prompt, correct, weak, mid = _arithmetic(index)
    return Item(
        subject="quant_exact",
        sample_id=f"{index:02d}",
        prompt=prompt,
        target=str(correct),
        metadata={"format": "exact"},
        answers=_exact_answers(index, str(correct), str(weak), str(mid)),
    )


def _render_hits(facts: tuple[str, ...], hits: tuple[int, int, int]) -> dict[str, str]:
    def render(count: int) -> str:
        if count <= 0:
            return "no-report"
        return "\n".join(facts[:count])

    return {role: render(count) for role, count in zip(ROLES, hits, strict=True)}


def _partial_answers(index: int, facts: tuple[str, ...]) -> dict[str, str]:
    if index < 4:
        hits = (4, 4, 4)
    elif index < 6:
        hits = (2, 4, 4)
    elif index < 12:
        hits = (0, 4, 4)
    elif index < 16:
        hits = (0, 1, 3)
    else:
        hits = (0, 0, 2)
    return _render_hits(facts, hits)


def _check_facts(prompt: str, facts: tuple[str, ...]) -> None:
    if len(set(facts)) != len(facts):
        raise ValueError("rubric facts must be unique")
    for fact in facts:
        if contains_fact(prompt, fact):
            raise ValueError(f"rubric fact is already in the prompt: {fact}")
        for other in facts:
            if other != fact and contains_fact(fact, other):
                raise ValueError(f"rubric facts overlap: {fact} / {other}")


def _rubric_item(index: int) -> Item:
    cartons, each, withdrawn = _numbers(index)
    remaining = cartons * each - withdrawn
    facts = (
        f"cartons={cartons}",
        f"kits-each={each}",
        f"withdrawn={withdrawn}",
        f"remaining={remaining}",
    )
    prompt = (
        f"A field office received {cartons} cartons. Each carton holds {each} sealed kits. "
        f"Staff withdrew {withdrawn} kits before the count was locked. Report exactly four "
        "lines, using these keys and the numbers you calculate: cartons, kits-each, "
        "withdrawn, remaining. Write each line as key=value."
    )
    _check_facts(prompt, facts)
    return Item(
        subject="rubric_facts",
        sample_id=f"{index:02d}",
        prompt=prompt,
        target="\n".join(facts),
        metadata={"format": "rubric", "rubric": list(facts)},
        answers=_partial_answers(index, facts),
    )


def _procedure_item(index: int) -> Item:
    start = 30 + index
    cut = 2 + (index % 3)
    discard = 1 + (index % 4)
    group = 4
    after_cut = start - cut
    after_dry = after_cut - discard
    after_pack = after_dry - (after_dry % group)
    shipped = after_pack // 2
    facts = (
        f"after-cut={after_cut}",
        f"after-dry={after_dry}",
        f"after-pack={after_pack}",
        f"shipped={shipped}",
    )
    prompt = (
        f"Start with {start} tiles. Cut away {cut} tiles. Of what remains, discard "
        f"{discard} tiles that failed drying. Pack the rest into groups of {group} and "
        "set aside any leftover that does not fill a group. Ship half of the packed "
        "tiles, rounded down. Report exactly four lines: after-cut, after-dry, "
        "after-pack, and shipped, each written as key=value."
    )
    _check_facts(prompt, facts)
    return Item(
        subject="procedure_steps",
        sample_id=f"{index:02d}",
        prompt=prompt,
        target="\n".join(facts),
        metadata={"format": "rubric", "rubric": list(facts)},
        answers=_partial_answers(index, facts),
    )


def _structured_item(index: int) -> Item:
    readings = (3 + index, 10 + (index % 9), 2 + ((index * 2) % 7))
    low, high = min(readings), max(readings)
    facts = (
        f"min={low}",
        f"max={high}",
        f"sum={sum(readings)}",
        f"range={high - low}",
    )
    shown = ", ".join(str(value) for value in readings)
    prompt = (
        f"Three gauges read {shown}. Report the minimum, the maximum, the sum, and "
        "the range (maximum minus minimum). Write exactly four lines: min, max, sum, "
        "and range, each as key=value."
    )
    _check_facts(prompt, facts)
    return Item(
        subject="structured_fields",
        sample_id=f"{index:02d}",
        prompt=prompt,
        target="\n".join(facts),
        metadata={"format": "rubric", "rubric": list(facts)},
        answers=_partial_answers(index, facts),
    )


def _boxed_answers(index: int, correct: int, weak: int, mid: int) -> dict[str, str]:
    def box(value: int) -> str:
        return f"\\boxed{{{value}}}"

    # 3 easy, 5 mid, 4 hard. Means: weak 0.25, mid 2/3, strong 1.
    if index < 3:
        chosen = (correct, correct, correct)
    elif index < 8:
        chosen = (weak, correct, correct)
    else:
        chosen = (weak, mid, correct)
    return {role: box(value) for role, value in zip(ROLES, chosen, strict=True)}


def _boxed_item(index: int) -> Item:
    """An original integer expression. Gold is the bare integer Math-Verify checks."""
    left, right, adjust = 6 + index, 2 + (index % 5), 1 + (index % 3)
    kind = index % 3
    if kind == 0:
        correct, weak, mid = left + right * adjust, left + right + adjust, left * right + adjust
        shown = f"{left} + {right} \\times {adjust}"
    elif kind == 1:
        correct = left * right - adjust
        weak, mid = left * right, left * right - adjust - right
        shown = f"{left} \\times {right} - {adjust}"
    else:
        correct = (left + right) * adjust
        weak, mid = left + right * adjust, left * right * adjust
        shown = f"({left} + {right}) \\times {adjust}"
    if len({correct, weak, mid}) != 3 or min(correct, weak, mid) <= 0:
        raise ValueError(f"math item {index} does not have three positive results")
    prompt = (
        f"Evaluate {shown}. Do multiplication before addition or subtraction unless "
        "parentheses say otherwise. Put only the final integer in \\boxed{}."
    )
    return Item(
        subject=MATH_SUBJECT,
        sample_id=f"{index:02d}",
        prompt=prompt,
        target=str(correct),
        metadata={"format": "math"},
        answers=_boxed_answers(index, correct, weak, mid),
    )


def _agentic_item(index: int) -> Item:
    """A short tool trace. Partial credit is the share of required lines reported."""
    slots = 4 + index
    each = 5 + (index % 4)
    hold = 1 + (index % 3)
    counted = slots * each
    free = counted - hold
    facts = (
        f"lookup=bin-{slots}",
        f"count={counted}",
        f"hold={hold}",
        f"free={free}",
    )
    prompt = (
        f"You are a warehouse agent. Look up aisle {slots}. That aisle has {slots} slots "
        f"and each slot stores {each} units. {hold} of those units are already reserved. "
        "Report the tool trace as exactly four lines, using the numbers you calculate: "
        "lookup=bin-N, count=N, hold=N, and free=N (counted units minus the reservation)."
    )
    _check_facts(prompt, facts)
    if index < 3:
        hits = (4, 4, 4)
    elif index < 6:
        hits = (2, 4, 4)
    elif index < 9:
        hits = (0, 1, 3)
    else:
        hits = (0, 0, 2)
    return Item(
        subject=AGENTIC_SUBJECT,
        sample_id=f"{index:02d}",
        prompt=prompt,
        target="\n".join(facts),
        metadata={"format": "rubric", "rubric": list(facts)},
        answers=_render_hits(facts, hits),
    )


_BUILDERS = {
    "quant_exact": _quant_item,
    "choice_logic": _choice_item,
    "rubric_facts": _rubric_item,
    "procedure_steps": _procedure_item,
    "structured_fields": _structured_item,
    MATH_SUBJECT: _boxed_item,
    AGENTIC_SUBJECT: _agentic_item,
}
_TASKS = {"chat_qa": rubric_chat, "math_code": boxed_math, "agentic": rubric_chat}


def questions() -> tuple[Item, ...]:
    """The original items, in stable subject order, with the counts in ``SUBJECT_COUNTS``."""
    items = tuple(
        _BUILDERS[subject](index)
        for subject in SUBJECTS
        for index in range(SUBJECT_COUNTS[subject])
    )
    if len(items) != QUESTION_COUNT or len({item.question_id for item in items}) != QUESTION_COUNT:
        raise ValueError(f"competition v0.0.3 requires {QUESTION_COUNT} unique questions")
    return items


def _score(item: Item, role: str) -> Decimal:
    if item.subject == MATH_SUBJECT:
        outcome = grade_math(item.target, item.answers[role])
    else:
        outcome = grade_rubric_chat(item.target, item.answers[role], item.metadata)
    if outcome.status != "graded":
        raise ValueError(f"{item.question_id} {role} did not settle: {outcome.code}")
    return Decimal(str(outcome.score))


def token_count(text: str) -> int:
    """A local token estimate used only so stand-in spend hits the price table."""
    return max(1, len(text.encode()) // 4)


def question_spend_usd(model_id: str, prompt: str, answer: str) -> Decimal:
    """Price one prompt and answer with the staging world-card rate for ``model_id``."""
    prices = TokenPrices.model_validate(PANEL_PRICES[model_id])
    micro = usage_cost_microusd(
        ModelUsage(input_tokens=token_count(prompt), output_tokens=token_count(answer)),
        prices,
    )
    if micro is None or micro <= 0:
        raise ValueError(f"no positive price for {model_id}")
    return Decimal(micro) / Decimal(1_000_000)


def _freeze(items: tuple[Item, ...]) -> tuple[Question, ...]:
    frozen: list[Question] = []
    for subject in SUBJECTS:
        selected = [item for item in items if item.subject == subject]
        samples = [
            Sample(
                id=item.sample_id,
                input=item.prompt,
                target=item.target,
                metadata=dict(item.metadata),
            )
            for item in selected
        ]
        workload = WORKLOAD[subject]
        task = task_with(_TASKS[workload](), dataset=samples)
        frozen.extend(prepare_task(task, source=subject, workload_class=workload))
    if [question.question_id for question in frozen] != [item.question_id for item in items]:
        raise ValueError("frozen question ids drifted from the source items")
    return tuple(frozen)


def _grader(unlabeled: tuple[Question, ...]) -> dict[str, str]:
    identity = evaluation_identity(unlabeled)
    return {
        "kind": identity["kind"],
        "packages": json.dumps(identity["packages"], sort_keys=True, separators=(",", ":")),
        "recipes_sha256": identity["recipes_sha256"],
    }


def _calibrate(
    items: tuple[Item, ...], unlabeled: tuple[Question, ...]
) -> tuple[tuple[Question, ...], CalibrationSummary]:
    by_id = {item.question_id: item for item in items}
    plan = DifficultyCalibrationPlan(
        questions=unlabeled,
        panel=tuple(Treatment(model=model, weight=Decimal(1), repeats=1) for _, model in PANEL),
        grader=_grader(unlabeled),
        execution={
            "collection": "local-stand-in",
            "inference": "not-called",
            "panel_model_ids": [model.model_id for _, model in PANEL],
        },
        limits=CallLimits(max_cost_usd=80, max_question_cost_usd=2),
        seed="competition-v0.0.3-stand-in",
    )
    by_model = {
        (treatment.model.provider_id, treatment.model.model_id, treatment.model.effort): treatment
        for treatment in plan.panel
    }
    treatments = {
        role: by_model[(model.provider_id, model.model_id, model.effort)] for role, model in PANEL
    }
    outcomes = []
    for question in plan.questions:
        item = by_id[question.question_id]
        for role, treatment in treatments.items():
            score = _score(item, role)
            answer = item.answers[role]
            spend = question_spend_usd(treatment.model.model_id, item.prompt, answer)
            identity = {
                "question_id": question.question_id,
                "treatment": treatment.key,
                "repeat": 0,
                "score": str(score),
            }
            outcomes.append(
                CalibrationOutcome(
                    plan_sha256=plan.sha256,
                    question_id=question.question_id,
                    question_sha256=question_identity(question),
                    treatment=treatment.key,
                    repeat=0,
                    outcome_sha256=digest(identity),
                    score=score,
                    evidence_bytes=max(1, len(answer.encode())),
                    active_ns=10_000_000,
                    spend_usd=spend,
                    service_spend_usd=Decimal(0),
                    spend=KnownSpend(usd=spend),
                    call_ids=(digest({"call": identity}),),
                )
            )
    spec = CalibrationSpec.from_plan(plan)
    result = calculate(spec, [row.observation for row in outcomes])
    labeled = tuple(
        question.model_copy(
            update={
                "difficulty_decile": result.labels[question.question_id].decile,
                "panel_solve_share": result.labels[question.question_id].easiness,
            }
        )
        for question in unlabeled
    )
    summary = CalibrationSummary(plan=spec, outcomes=tuple(outcomes), result=result)
    verify_summary(summary, labeled)
    return labeled, summary


def build_pack() -> BuiltPack:
    """Freeze, grade the stand-ins, and return a competition pack archive."""
    items = questions()
    labeled, summary = _calibrate(items, _freeze(items))
    records = records_bytes(labeled)
    calibration = canonical_json(summary.model_dump(mode="json"))
    manifest = PackManifest(
        pack_id=PACK_ID,
        records_sha256=hashlib.sha256(records).hexdigest(),
        source_releases=("synthetic-competition-v0.0.3",),
        data_visibility="private",
        purpose="competition",
        difficulty_calibration_sha256=hashlib.sha256(calibration).hexdigest(),
        world_card_sha256=WORLD_CARD_SHA256,
    )
    archive = pack_bytes(manifest, records, difficulty_calibration=calibration)
    return BuiltPack(
        archive=archive,
        sha256=hashlib.sha256(archive).hexdigest(),
        questions=labeled,
        items=items,
        summary=summary,
    )


def role_scores(built: BuiltPack) -> dict[str, dict[str, Decimal]]:
    """Stand-in score of every question, keyed by question id and role."""
    by_id = {item.question_id: item for item in built.items}
    return {
        question.question_id: {role: _score(by_id[question.question_id], role) for role in ROLES}
        for question in built.questions
    }


def mean_score(scores: Mapping[str, Decimal]) -> Decimal:
    return sum(scores.values(), Decimal(0)) / len(scores)


def subject_counts(built: BuiltPack) -> dict[str, int]:
    counts = {subject: 0 for subject in SUBJECTS}
    for question in built.questions:
        counts[question.source_benchmark] += 1
    return counts


def manifest_name() -> str:
    return MANIFEST
