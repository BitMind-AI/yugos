"""Net-value arithmetic, floors and configuration, checked against hand-worked examples."""

from decimal import Decimal

import pytest
from test_competition import CATEGORIES, config, ids, policy, submission

from yugos.competition import CategoryMeasurement
from yugos.competition.net_value import (
    Bootstrap,
    NetValuePolicy,
    Tau,
    category_net_value,
    high_water_marks,
    load_net_value_policy,
    lower_bound,
    net_value,
    quality_lower_bound,
    question_net_values,
    question_value,
    tau,
    time_value,
)

QUESTIONS = [f"q{i}" for i in range(1000)]
HARD = QUESTIONS[:100]


def row(grades, spend, run_id="r", time_ns=None):
    grades = {q: Decimal(grades.get(q, 0)) for q in QUESTIONS}
    spend = {q: Decimal(spend.get(q, 0)) for q in QUESTIONS}
    times = None if time_ns is None else dict.fromkeys(QUESTIONS, time_ns)
    return CategoryMeasurement(
        run_id=run_id,
        miner_id=run_id,
        submission_sha256="a" * 64,
        cohort_sha256="b" * 64,
        category="chat_qa",
        grades=grades,
        question_spend_usd=spend,
        cost_usd=sum(spend.values(), Decimal(0)),
        execution_time_ns=1 if times is None else sum(times.values()),
        question_time_ns=times,
        cost_complete=True,
        execution_complete=True,
        integrity_pass=True,
    )


def compare(challenger, incumbent, **terms):
    terms = (
        dict(
            question_ids=QUESTIONS,
            hard_question_ids=HARD,
            value_usd=Decimal(1),
            hard_multiplier=Decimal(2),
        )
        | terms
    )
    return (
        category_net_value(challenger, incumbent, **terms),
        question_net_values(challenger, incumbent, **terms),
    )


def test_extra_hard_answers_can_pay_for_extra_spend():
    # +6 regular and +8 hard answers at H=2, for $21 more: 100 * (6 + 16 - 21) / 1000.
    wins = {q: 1 for q in QUESTIONS[100:106] + HARD[:8]}
    value, per_question = compare(row(wins, {"q999": 21}), row({}, {}))
    assert (value.n, value.n_hard, value.value_usd) == (1000, 100, 1)
    assert (value.regular_answer_delta, value.hard_answer_delta) == (6, 8)
    assert (value.quality_points, value.spend_points) == (Decimal("2.2"), Decimal("2.1"))
    assert value.net_value == Decimal("0.1")
    assert sum(per_question) / len(per_question) == value.net_value


def test_half_price_router_can_lose_answers_and_still_win():
    # Incumbent spends $29; half price saves $14.50 and loses 10 regular answers.
    king_spend = {q: Decimal("0.029") for q in QUESTIONS}
    cheap_spend = {q: Decimal("0.0145") for q in QUESTIONS}
    king_grades = {q: 1 for q in QUESTIONS[100:200]}
    cheap_grades = {q: 1 for q in QUESTIONS[110:200]}
    value, _ = compare(row(cheap_grades, cheap_spend), row(king_grades, king_spend))
    assert value.spend_delta_usd == Decimal("-14.5")
    assert value.net_value == Decimal("0.45")


def test_priced_time_converts_extra_time_to_answers():
    # T = 100 s per answer. 10 more answers for 1 s more on each of 1,000 questions:
    # 100 * (10 - 1000 * 1 / 100) / 1000 = 0, so the time cancels the answers.
    second, value_ns = 1_000_000_000, Decimal(100_000_000_000)
    wins = {q: 1 for q in QUESTIONS[100:110]}
    value, per_question = compare(
        row(wins, {}, time_ns=2 * second), row({}, {}, time_ns=second), time_value_ns=value_ns
    )
    assert value.time_delta_ns == 1000 * second
    assert (value.quality_points, value.time_points, value.net_value) == (1, 1, 0)
    assert sum(per_question) / len(per_question) == value.net_value
    # The same comparison without a time value ignores time entirely.
    unpriced, _ = compare(row(wins, {}, time_ns=2 * second), row({}, {}, time_ns=second))
    assert unpriced.net_value == 1 and unpriced.time_points is None


def test_priced_time_needs_per_question_time_from_both_routers():
    with pytest.raises(ValueError, match="per-question time from both"):
        compare(row({}, {}, time_ns=1), row({}, {}), time_value_ns=Decimal(1))


def test_unpriced_time_leaves_existing_records_unchanged():
    # Stored policies, measurements and decisions are replayed by digest.
    assert "k" not in NetValuePolicy().model_dump(mode="json")
    assert "question_time_ns" not in row({}, {}).model_dump(mode="json")
    value, _ = compare(row({}, {}), row({}, {}))
    assert not {"time_value_ns", "time_delta_ns", "time_points"} & set(
        value.model_dump(mode="json")
    )


def test_question_time_must_cover_and_sum_to_category_time():
    timed = row({}, {}, time_ns=1).model_dump(mode="json")
    with pytest.raises(ValueError, match="sum to category time"):
        CategoryMeasurement.model_validate(timed | {"execution_time_ns": 999})
    partial = dict(list(timed["question_time_ns"].items())[:999])
    with pytest.raises(ValueError, match="cover the graded questions"):
        CategoryMeasurement.model_validate(
            timed | {"execution_time_ns": 999, "question_time_ns": partial}
        )


def test_time_value_is_k_times_anchor_time():
    assert time_value(NetValuePolicy(), Decimal(5)) is None
    assert time_value(config(k="300"), Decimal("8.4e9")) == Decimal("2.520e12")
    with pytest.raises(ValueError, match="anchor time"):
        time_value(config(k="300"))


def test_hard_multiplier_and_answer_value_scale_their_own_terms():
    wins = {HARD[0]: 1}
    base, _ = compare(row(wins, {"q999": 1}), row({}, {}))
    tripled, _ = compare(row(wins, {"q999": 1}), row({}, {}), hard_multiplier=Decimal(3))
    pricier, _ = compare(row(wins, {"q999": 1}), row({}, {}), value_usd=Decimal(10))
    assert (base.quality_points, tripled.quality_points) == (Decimal("0.2"), Decimal("0.3"))
    assert tripled.spend_points == base.spend_points
    assert pricier.spend_points == base.spend_points / 10
    assert pricier.quality_points == base.quality_points


def test_per_category_overrides_fall_back_to_global_values():
    categories = NetValuePolicy().model_dump(mode="json")["categories"]
    categories["agentic"] |= {
        "hard_multiplier": "4",
        "time_multiplier": "3",
        "tau": {"share": "0.5", "min": "0.01", "max": "0.1"},
    }
    policy = config(categories=categories)
    assert policy.hard_multiplier_for("agentic") == 4
    assert policy.hard_multiplier_for("chat_qa") == policy.hard_multiplier == 2
    assert policy.time_multiplier_for("agentic") == 3
    assert policy.time_multiplier_for("math_code") == policy.time.multiplier
    assert policy.tau_for("agentic").max == Decimal("0.1")
    assert policy.tau_for("chat_qa") == policy.floor.tau


def test_absolute_time_budget_overrides_the_panel_leader_multiple():
    categories = NetValuePolicy().model_dump(mode="json")["categories"]
    categories["chat_qa"] |= {"time_budget_seconds": "900"}
    policy = config(categories=categories)
    assert policy.time_limit_ns("chat_qa", 1_000_000_000) == 900_000_000_000
    assert policy.time_limit_ns("math_code", 1_000_000_000) == 2 * 1_000_000_000
    categories["chat_qa"] |= {"time_multiplier": "3"}
    with pytest.raises(ValueError, match="either time_multiplier or time_budget_seconds"):
        config(categories=categories)


@pytest.mark.parametrize(
    "quality,expected", [("0.97", "0.02"), ("0.90", "0.025"), ("0.74", "0.05"), ("1", "0.02")]
)
def test_tau_is_a_clamped_share_of_remaining_error(quality, expected):
    assert tau(Tau(), Decimal(quality)) == Decimal(expected)


def test_quality_lower_bound_uses_sample_spread_and_confidence():
    assert quality_lower_bound([Decimal("0.9")] * 50, Decimal("0.95")) == Decimal("0.9")
    grades = [Decimal(i % 2) for i in range(100)]
    # sd = sqrt(25/99) ~ 0.50252; z(0.95) = 1.644853627.
    assert float(quality_lower_bound(grades, Decimal("0.95"))) == pytest.approx(0.41734, abs=1e-5)
    assert quality_lower_bound(grades, Decimal("0.99")) < quality_lower_bound(
        grades, Decimal("0.95")
    )
    assert quality_lower_bound([Decimal(0), Decimal(1)], Decimal("0.99")) == 0


def test_high_water_marks_can_follow_mean_quality():
    mean_policy = policy(net_value=config(floor={"use_lower_bound": False}))
    mixed = {c: {"grades": {q: Decimal(i % 2) for i, q in enumerate(ids(c))}} for c in CATEGORIES}
    top = submission("top", quality="0.97")
    marks = high_water_marks(mean_policy, (top, submission("second", overrides=mixed)))
    assert marks["chat_qa"].quality == Decimal("0.97")
    assert high_water_marks(mean_policy, ())["chat_qa"].quality == Decimal("0.95")


def test_bootstrap_is_seeded_and_stratified():
    strata = {
        "chat_qa": (Decimal("0.6"), [Decimal(i % 3 - 1) for i in range(90)]),
        "agentic": (Decimal("0.4"), [Decimal(2)] * 10),
    }
    first = lower_bound(strata, bootstrap=Bootstrap(), confidence=Decimal("0.95"))
    assert first == lower_bound(
        dict(reversed(strata.items())), bootstrap=Bootstrap(), confidence=Decimal("0.95")
    )
    assert first < Decimal("0.8")  # The point value is 0.6 * 0 + 0.4 * 2.
    assert lower_bound(strata, bootstrap=Bootstrap(seed=3), confidence=Decimal("0.95")) != first
    constant = {"chat_qa": (Decimal(1), [Decimal("0.5")] * 20)}
    assert lower_bound(constant, bootstrap=Bootstrap(), confidence=Decimal("0.95")) == Decimal(
        "0.5"
    )


def test_overall_net_value_is_the_weighted_sum():
    value, _ = compare(row({QUESTIONS[500]: 1}, {}), row({}, {}))
    assert net_value({"a": Decimal("0.25"), "b": Decimal("0.75")}, {"a": value, "b": value}) == (
        value.net_value
    )
    with pytest.raises(ValueError, match="every weighted category"):
        net_value({"a": Decimal(1)}, {"b": value})


def test_question_value_is_fixed_or_m_times_anchor():
    assert question_value(NetValuePolicy(), "agentic") == 10
    anchored = config(
        value_basis="anchor",
        m="30",
        categories={c: {"weight": w} for c, w in zip(CATEGORIES, ("0.4", "0.35", "0.25"))},
    )
    assert question_value(anchored, "chat_qa", Decimal("0.004")) == Decimal("0.12")
    with pytest.raises(ValueError, match="anchor cost"):
        question_value(anchored, "chat_qa")


def test_loader_accepts_parsed_yaml_numbers_and_fraction_thresholds():
    raw = {
        "min_net_value": 0.2,
        "hard_max_solve_share": "1/3",
        "time": {"multiplier": 2.0},
        "categories": {
            "chat_qa": {"weight": 0.40, "value_regular_usd": 1.00},
            "math_code": {"weight": 0.35, "value_regular_usd": 1.00},
            "agentic": {"weight": 0.25, "value_regular_usd": 10.00},
        },
    }
    loaded = load_net_value_policy(raw)
    # Floats keep their shortest decimal spelling, so the weights still sum to exactly one.
    assert loaded.categories["math_code"].weight == Decimal("0.35")
    # A panel of three equally weighted models with one solve sits exactly on 1/3.
    assert Decimal(1) / Decimal(3) <= loaded.hard_max_solve_share < Decimal("0.3334")
    # By default a question is hard only when no panel model solved it.
    assert NetValuePolicy().hard_max_solve_share == 0


@pytest.mark.parametrize(
    "change,message",
    [
        (
            {"categories": {"chat_qa": {"weight": "0.5", "value_regular_usd": "1"}}},
            "sum to exactly one",
        ),
        (
            {
                "categories": {
                    "chat_qa": {"weight": "0.4", "value_regular_usd": "1"},
                    "math_code": {"weight": "0.35", "value_regular_usd": "1"},
                    "agentic": {"weight": "0.2501", "value_regular_usd": "10"},
                }
            },
            "sum to exactly one",
        ),
        (
            {
                "categories": {
                    "chat_qa": {"weight": "1", "value_regular_usd": "0"},
                }
            },
            "greater than 0",
        ),
        ({"categories": {"chat_qa": {"weight": "1"}}}, "requires value_regular_usd"),
        (
            {
                "value_basis": "anchor",
                "categories": {"chat_qa": {"weight": "1", "value_regular_usd": "1"}},
            },
            "remove value_regular_usd",
        ),
        ({"floor": {"tau": {"share": "0.25", "min": "0.06", "max": "0.05"}}}, "tau min"),
        ({"confidence": "0.5"}, "greater than 0.5"),
        ({"confidence": "1"}, "less than 1"),
        ({"m": "0"}, "greater than 0"),
        ({"k": "0"}, "greater than 0"),
        ({"hard_multiplier": "-1"}, "greater than 0"),
        ({"time": {"multiplier": "0"}}, "greater than 0"),
        ({"bootstrap": {"samples": 10, "seed": 0}}, "greater than or equal to 1000"),
        ({"min_net_value": "-0.1"}, "greater than or equal to 0"),
        ({"hard_max_solve_share": "1.5"}, "less than or equal to 1"),
        ({"hard_max_solve_share": "1/0"}, "invalid fraction"),
        ({"hard_max_solve_share": "one/3"}, "invalid fraction"),
        ({"unknown": 1}, "Extra inputs"),
    ],
)
def test_invalid_configuration_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        load_net_value_policy(change)
