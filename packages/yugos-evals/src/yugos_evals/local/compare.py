"""Compare two local runs on the same pack with core's net-value rule.

B is the challenger and A the incumbent. Each category's net value and the
paired-bootstrap lower bound come from `yugos.competition.net_value`; overall weights are
the policy's, renormalized over the categories the pack covers. Competition
gates (quality floors, time limits) need hosted calibration and are not applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from yugos.competition import CategoryMeasurement, derive_hard_question_ids, spend_error
from yugos.competition.net_value import (
    PRECISION,
    CategoryNetValue,
    NetValuePolicy,
    category_net_value,
    load_net_value_policy,
    lower_bound,
    net_value,
    question_net_values,
    question_value,
)
from yugos.evaluation import Question


@dataclass(frozen=True)
class Comparison:
    categories: dict[str, CategoryNetValue]
    weights: dict[str, Decimal]
    net_value: Decimal
    lower_bound: Decimal
    min_net_value: Decimal
    hard_tier: bool

    @property
    def b_wins(self) -> bool:
        return self.lower_bound > self.min_net_value

    def verdict(self) -> str:
        outcome = "B beats A" if self.b_wins else "B does not beat A"
        return (
            f"{outcome}: NV={self.net_value:.4f}, LB={self.lower_bound:.4f} "
            f"(needs LB > {self.min_net_value})"
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "b_beats_a": self.b_wins,
            "net_value": str(self.net_value),
            "lower_bound": str(self.lower_bound),
            "min_net_value": str(self.min_net_value),
            "weights": {k: str(v) for k, v in self.weights.items()},
            "hard_tier": self.hard_tier,
            "categories": {k: v.model_dump(mode="json") for k, v in self.categories.items()},
        }


def load_policy(path: Path | None) -> NetValuePolicy:
    """The competition's default net-value policy, or one from a JSON/YAML file."""
    if path is None:
        return NetValuePolicy()
    text = Path(path).read_text()
    if Path(path).suffix in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("net_value"), dict):
        raw = raw["net_value"]
    return load_net_value_policy(raw)


def _categories(run: dict[str, Any], label: str) -> dict[str, CategoryMeasurement]:
    rows = {}
    for raw in run["categories"]:
        row = CategoryMeasurement.model_validate(raw)
        problems = []
        if not row.execution_complete:
            problems.append("execution incomplete")
        error = spend_error(row)
        if error is not None:
            problems.append(error)
        if not row.integrity_pass:
            problems.append("worker cleanup unconfirmed")
        if problems:
            raise ValueError(f"run {label}, {row.category}: {', '.join(problems)}")
        rows[row.category] = row
    return rows


def _hard_ids(run: dict[str, Any], policy: NetValuePolicy) -> dict[str, tuple[str, ...]] | None:
    """Hard questions by the policy's panel threshold, when the pack carries panel shares."""
    shares = run.get("panel_solve_shares") or {}
    if not shares or any(value is None for value in shares.values()):
        return None
    categories = {qid: c["category"] for c in run["categories"] for qid in c["grades"]}
    questions = [
        Question.model_construct(
            question_id=qid,
            workload_class=categories[qid],
            panel_solve_share=Decimal(share),
        )
        for qid, share in shares.items()
    ]
    return derive_hard_question_ids(questions, max_solve_share=policy.hard_max_solve_share)


def compare_runs(a: dict[str, Any], b: dict[str, Any], policy: NetValuePolicy) -> Comparison:
    """Net value of B over A, per category and overall, with its bootstrap lower bound."""
    if a["pack"]["records_sha256"] != b["pack"]["records_sha256"]:
        raise ValueError("the runs used different packs; compare runs of the same pack")
    if a["question_ids"] != b["question_ids"]:
        raise ValueError("the runs evaluated different questions (check --limit)")
    ours, theirs = _categories(b, "B"), _categories(a, "A")
    if set(ours) != set(theirs):
        raise ValueError("the runs measured different categories")
    unknown = sorted(set(ours) - set(policy.categories))
    if unknown:
        raise ValueError(f"the policy has no value for categories {unknown}")
    hard = _hard_ids(a, policy)
    categories, strata = {}, {}
    with localcontext() as context:
        context.prec = PRECISION
        total_weight = sum((policy.categories[c].weight for c in ours), Decimal(0))
        weights = {c: policy.categories[c].weight / total_weight for c in sorted(ours)}
    for name in sorted(ours):
        terms = dict(
            question_ids=tuple(sorted(ours[name].grades)),
            hard_question_ids=(hard or {}).get(name, ()),
            value_usd=question_value(policy, name),
            hard_multiplier=policy.hard_multiplier_for(name),
        )
        categories[name] = category_net_value(ours[name], theirs[name], **terms)
        strata[name] = (weights[name], question_net_values(ours[name], theirs[name], **terms))
    return Comparison(
        categories=categories,
        weights=weights,
        net_value=net_value(weights, categories),
        lower_bound=lower_bound(strata, bootstrap=policy.bootstrap, confidence=policy.confidence),
        min_net_value=policy.min_net_value,
        hard_tier=hard is not None,
    )


def format_comparison(a: dict[str, Any], b: dict[str, Any], result: Comparison) -> str:
    rows = [("CATEGORY", "N", "QUALITY A", "QUALITY B", "COST/Q A", "COST/Q B", "NV")]
    for name, value in result.categories.items():
        qa, qb = a["summary"][name], b["summary"][name]
        rows.append(
            (
                name,
                str(value.n),
                f"{Decimal(qa['quality']):.3f}",
                f"{Decimal(qb['quality']):.3f}",
                f"{Decimal(qa['cost_per_question_usd']):.6f}",
                f"{Decimal(qb['cost_per_question_usd']):.6f}",
                f"{value.net_value:.4f}",
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    if not result.hard_tier:
        lines.append("no panel solve shares in this pack: every question counts as regular")
    lines.append(result.verdict())
    return "\n".join(lines)
