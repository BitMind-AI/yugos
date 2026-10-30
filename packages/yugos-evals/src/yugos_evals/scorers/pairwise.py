"""Arena-Hard v2.0 pairwise preference against a pinned baseline, judged by the `grader` role.

The judge sees the question with the baseline and candidate answers in both
positions (baseline first, then candidate first) under the category's upstream
system prompt. Each verdict is parsed with upstream's parser; both verdicts
must be valid, complete judgments. The score is the mean of upstream's
weighted battles, a preference in [0, 1], not objective correctness. An
unusable judgment leaves the sample unscored with a reason.
"""

from __future__ import annotations

import hashlib

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
    model_roles,
)
from inspect_ai.scorer import Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver

from yugos.evaluation.grades import GradeCode
from yugos.inspect_engine import (
    RolePin,
    UnboundGraderRole,
    require_grader_roles,
    require_role_pins,
)

from . import arenahard_upstream as upstream
from ._outcome import Outcome, completion_gate, to_score

GRADER_ROLE = "grader"
# Upstream's v2.0 judge settings: deterministic sampling and a per-call timeout.
# Reasoning judges ignore the temperature.
JUDGE_CONFIG = GenerateConfig(temperature=0.0, timeout=300)


def judge_messages(category: str, question: str, baseline: str, candidate: str, round: int):
    """The upstream judge conversation; round 1 puts the candidate second, round 2 first."""
    if round not in (1, 2) or isinstance(round, bool):
        raise ValueError("Arena has exactly two answer positions")
    system = upstream.JUDGE_SETTINGS[category]["system_prompt"]
    answer_a, answer_b = (baseline, candidate) if round == 1 else (candidate, baseline)
    user = upstream.PROMPT_TEMPLATE.format(QUESTION=question, ANSWER_A=answer_a, ANSWER_B=answer_b)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_verdict(text: str, *, stop_reason: str | None) -> dict:
    """Upstream's parsed label for one judgment, and whether it is a usable, complete vote."""
    label = upstream.get_score(text, upstream.PATTERNS)
    valid = stop_reason in {"stop", "end_turn"}
    try:
        upstream.score_battles(label, label)
    except (KeyError, TypeError):
        valid = False
    return {
        "label": label,
        "stop_reason": stop_reason,
        "valid": valid,
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def preference(verdicts: list[dict]) -> Outcome:
    """Average upstream's weighted battles over both positions; any unusable vote needs review."""
    details = {
        "score_semantics": "mean_upstream_weighted_pairwise_preference",
        "judge_calls": verdicts,
    }
    if len(verdicts) != 2 or not all(v["valid"] for v in verdicts):
        return Outcome(
            GradeCode.NEEDS_REVIEW,
            status="unavailable",
            details={**details, "cause": "missing_or_invalid_arena_judgments"},
        )
    battles = upstream.score_battles(verdicts[0]["label"], verdicts[1]["label"])
    score = sum(battles) / len(battles)
    return Outcome(
        GradeCode.CORRECT if score > 0 else GradeCode.INCORRECT,
        score=score,
        details={**details, "weighted_battles": battles, "full_credit": score == 1},
    )


def grader_model(judge_model: str | None) -> Model:
    """The model bound to the `grader` role; never the model under test, never unbound."""
    tested = get_model()
    bound = model_roles()
    require_grader_roles([GRADER_ROLE], bound, model_under_test=tested)
    if judge_model is not None:
        require_role_pins({GRADER_ROLE: RolePin(model=judge_model)}, bound)
    model = get_model(role=GRADER_ROLE)
    if model is tested or str(model) == str(tested):
        raise UnboundGraderRole(f"model role {GRADER_ROLE!r} resolves to the model under test")
    return model


@solver
def judge_pin_guard(judge_model: str) -> Solver:
    """Fail the sample before the router is called unless the `grader` role is the pinned judge."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        grader_model(judge_model)
        return state

    return solve


def _chat(messages: list[dict]):
    types = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return [types[m["role"]](content=m["content"]) for m in messages]


@scorer(metrics=[mean(), stderr()])
def pairwise_judge_scorer(judge_model: str | None = None) -> Scorer:
    """Judge the answer against the sample's baseline (target) in both answer positions.

    `judge_model`, if given, pins the model the `grader` role must be bound to.
    Judge call failures propagate as sample errors; they never become a vote.
    """

    async def score(state: TaskState, target: Target):
        category = (state.metadata or {}).get("category")
        if category not in upstream.JUDGE_SETTINGS or not target.text.strip():
            return to_score(
                Outcome(
                    GradeCode.INVALID_GOLD,
                    status="unavailable",
                    details={"cause": "arena_reference_unusable"},
                )
            )
        gated = completion_gate(state)
        if gated is not None:
            return to_score(gated)
        candidate = state.output.completion
        if not candidate.strip():
            return to_score(
                Outcome(
                    GradeCode.INCORRECT,
                    details={"cause": "empty_answer"},
                )
            )
        judge = grader_model(judge_model)
        verdicts = []
        for round in (1, 2):
            messages = judge_messages(category, state.input_text, target.text, candidate, round)
            output = await judge.generate(_chat(messages), config=JUDGE_CONFIG)
            verdicts.append(parse_verdict(output.completion, stop_reason=output.stop_reason))
        return to_score(preference(verdicts), failure="grader_failed")

    return score
