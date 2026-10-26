"""Official answer extraction and HotpotQA answer/supporting-fact metrics."""

from __future__ import annotations

import json
import re
import string
from collections import Counter

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState


@scorer(metrics=[mean(), stderr()])
def longbench_answer() -> Scorer:
    """Match the answer phrase accepted by LongBench v2's published evaluator."""

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion.replace("*", "")
        match = re.search(r"The correct answer is \(([A-D])\)", response)
        if match is None:
            match = re.search(r"The correct answer is ([A-D])", response)
        answer = match.group(1) if match else ""
        return Score(value="C" if answer == target.text else "I", answer=answer)

    return score


def _normalize(answer: str) -> str:
    text = "".join(c for c in answer.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def hotpot_metrics(prediction: dict, reference: dict) -> dict[str, float]:
    """Per-question metrics from HotpotQA's published answer and evidence formulas."""
    names = (
        "em",
        "f1",
        "prec",
        "recall",
        "sp_em",
        "sp_f1",
        "sp_prec",
        "sp_recall",
        "joint_em",
        "joint_f1",
        "joint_prec",
        "joint_recall",
    )
    result = dict.fromkeys(names, 0.0)
    answer = prediction.get("answer")
    facts = prediction.get("supporting_facts")
    valid_answer = isinstance(answer, str)
    if valid_answer:
        actual, expected = _normalize(answer), _normalize(reference["answer"])
        result["em"] = float(actual == expected)
        special = {"yes", "no", "noanswer"}
        if not ((actual in special or expected in special) and actual != expected):
            actual_tokens, expected_tokens = actual.split(), expected.split()
            matches = sum((Counter(actual_tokens) & Counter(expected_tokens)).values())
            if matches:
                result["prec"] = matches / len(actual_tokens)
                result["recall"] = matches / len(expected_tokens)
                result["f1"] = (
                    2 * result["prec"] * result["recall"] / (result["prec"] + result["recall"])
                )
    valid_facts = isinstance(facts, list) and all(
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], str)
        and type(item[1]) is int
        and item[1] >= 0
        for item in facts
    )
    if valid_facts:
        actual_facts = {tuple(item) for item in facts}
        expected_facts = {tuple(item) for item in reference["supporting_facts"]}
        matches = len(actual_facts & expected_facts)
        result["sp_em"] = float(actual_facts == expected_facts)
        result["sp_prec"] = matches / len(actual_facts) if actual_facts else 0.0
        result["sp_recall"] = matches / len(expected_facts) if expected_facts else 0.0
        denominator = result["sp_prec"] + result["sp_recall"]
        result["sp_f1"] = (
            2 * result["sp_prec"] * result["sp_recall"] / denominator if denominator else 0.0
        )
    if valid_answer and valid_facts:
        result["joint_em"] = result["em"] * result["sp_em"]
        result["joint_prec"] = result["prec"] * result["sp_prec"]
        result["joint_recall"] = result["recall"] * result["sp_recall"]
        denominator = result["joint_prec"] + result["joint_recall"]
        result["joint_f1"] = (
            2 * result["joint_prec"] * result["joint_recall"] / denominator if denominator else 0.0
        )
    return result


@scorer(metrics=[mean(), stderr()])
def hotpot_answer() -> Scorer:
    """Grade one structured answer; native answer and evidence diagnostics are retained."""

    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion.strip()
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        try:
            prediction = json.loads(raw)
        except json.JSONDecodeError:
            prediction = {}
        if not isinstance(prediction, dict):
            prediction = {}
        return Score(
            value=hotpot_metrics(prediction, json.loads(target.text)),
            answer=state.output.completion,
        )

    return score
