"""`compare` applies core's net-value rule to two local runs of the same pack."""

import json
from decimal import Decimal

import pytest

from yugos.competition import CategoryMeasurement
from yugos.competition.net_value import NetValuePolicy
from yugos_evals.cli import main
from yugos_evals.local.compare import compare_runs, load_policy
from yugos_evals.local.runner import RUN_SCHEMA, summarize

N = 60


def synthetic_run(correct: dict[str, int], cost: str, *, records="a" * 64, complete=True):
    categories = []
    for category, right in correct.items():
        ids = [f"{category}:{i}" for i in range(N)]
        spend = {qid: Decimal(cost) for qid in ids}
        categories.append(
            CategoryMeasurement(
                run_id="r",
                miner_id="local",
                submission_sha256="b" * 64,
                cohort_sha256=records,
                category=category,
                grades={qid: Decimal(1 if i < right else 0) for i, qid in enumerate(ids)},
                question_spend_usd=spend,
                cost_usd=sum(spend.values(), Decimal(0)),
                execution_time_ns=N * 1_000_000,
                cost_complete=True,
                execution_complete=complete,
                integrity_pass=True,
            )
        )
    return {
        "schema": RUN_SCHEMA,
        "pack": {"records_sha256": records},
        "question_ids": [qid for row in categories for qid in row.grades],
        "panel_solve_shares": {qid: None for row in categories for qid in row.grades},
        "categories": [row.model_dump(mode="json") for row in categories],
        "questions": [],
        "summary": summarize(categories),
    }


def test_better_and_cheaper_challenger_wins():
    a = synthetic_run({"chat_qa": 30, "math_code": 30}, "0.01")
    b = synthetic_run({"chat_qa": 50, "math_code": 45}, "0.005")
    result = compare_runs(a, b, NetValuePolicy())
    assert result.b_wins and result.net_value > result.lower_bound > Decimal("0.2")
    assert result.verdict().startswith("B beats A: NV=")
    # Weights renormalize over the pack's categories (0.40 and 0.35 of the policy).
    assert sum(result.weights.values()) == pytest.approx(1)
    reverse = compare_runs(b, a, NetValuePolicy())
    assert not reverse.b_wins and reverse.verdict().startswith("B does not beat A")


def test_spend_counts_against_quality_at_the_policy_answer_value():
    a = synthetic_run({"chat_qa": 30}, "0.00")
    b = synthetic_run({"chat_qa": 30}, "0.10")
    [value] = compare_runs(a, b, NetValuePolicy()).categories.values()
    # 10 cents per question at $1 per answer is 10 quality points.
    assert value.net_value == Decimal(-10)


@pytest.mark.parametrize(
    ("b", "message"),
    [
        (synthetic_run({"chat_qa": 30}, "0.01", records="c" * 64), "different packs"),
        (synthetic_run({"chat_qa": 30}, "0.01", complete=False), "execution incomplete"),
    ],
)
def test_incomparable_runs_are_refused(b, message):
    a = synthetic_run({"chat_qa": 30}, "0.01")
    with pytest.raises(ValueError, match=message):
        compare_runs(a, b, NetValuePolicy())


def test_policy_file_overrides_the_default(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"net_value": {"min_net_value": "50"}}))
    assert load_policy(path).min_net_value == 50
    a = synthetic_run({"chat_qa": 30, "math_code": 30}, "0.01")
    b = synthetic_run({"chat_qa": 50, "math_code": 45}, "0.005")
    assert not compare_runs(a, b, load_policy(path)).b_wins


def test_compare_command_prints_the_verdict(tmp_path, capsys):
    a = synthetic_run({"chat_qa": 30}, "0.01")
    b = synthetic_run({"chat_qa": 50}, "0.01")
    for name, run in (("a", a), ("b", b)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "measurements.json").write_text(json.dumps(run))
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("B beats A: NV=33.3333")
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "missing")]) == 2
