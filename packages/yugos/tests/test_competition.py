import copy
import json
import pickle
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from yugos.competition import (
    CategoryMeasurement,
    CategoryRule,
    CompetitionCohort,
    CompetitionHistory,
    CompetitionPolicy,
    CompetitionPublication,
    CompetitionResult,
    CompetitionSnapshot,
    SubmissionMeasurement,
    decide_challenge,
    derive_hard_question_ids,
    digest,
)
from yugos.competition import rules as competition
from yugos.competition.net_value import NetValuePolicy, load_net_value_policy
from yugos.evaluation import Question

CATEGORIES = ("chat_qa", "math_code", "agentic")
N = 100


def ids(category):
    return [f"{category}-{i}" for i in range(N)]


def measured(
    run_id="king",
    *,
    category="chat_qa",
    quality="0.95",
    cost="10",
    seconds=600,
    grades=None,
    spend=None,
    **changes,
):
    grades = grades or {qid: Decimal(quality) for qid in ids(category)}
    spend = spend or {qid: Decimal(cost) / len(grades) for qid in grades}
    return CategoryMeasurement(
        **(
            dict(
                run_id=run_id,
                miner_id=run_id,
                submission_sha256="a" * 64,
                cohort_sha256="b" * 64,
                category=category,
                grades=grades,
                question_spend_usd=spend,
                cost_usd=sum(spend.values(), Decimal(0)),
                execution_time_ns=int(seconds * 1_000_000_000),
                cost_complete=True,
                execution_complete=True,
                integrity_pass=True,
            )
            | changes
        )
    )


def submission(run_id="king", *, overrides=None, **settings):
    return SubmissionMeasurement(
        categories=tuple(
            measured(run_id, category=category, **(settings | (overrides or {}).get(category, {})))
            for category in CATEGORIES
        )
    )


def policy(net_value=None, **changes):
    rules = tuple(
        CategoryRule(
            **(
                dict(
                    category=category,
                    cohort_sha256="b" * 64,
                    question_ids=tuple(ids(category)),
                    hard_question_ids=(f"{category}-0", f"{category}-1"),
                    private_question_ids=(f"{category}-2", f"{category}-3"),
                    initial_quality=Decimal("0.95"),
                    initial_hard_quality=Decimal("0.95"),
                    initial_private_quality=Decimal("0.95"),
                    panel_leader_time_ns=600_000_000_000,
                )
                | changes
            )
        )
        for category in CATEGORIES
    )
    return CompetitionPolicy(categories=rules, net_value=net_value or NetValuePolicy())


def config(**changes):
    return load_net_value_policy(NetValuePolicy().model_dump(mode="json") | changes)


def test_hard_tier_is_a_solve_share_threshold_that_keeps_ties_together():
    questions = tuple(
        Question(
            question_id=name,
            prompt="Question",
            expected_answer="A",
            inspect_task={"builder": "fixture"},
            inspect_sample={"id": "fixture"},
            workload_class=category,
            panel_solve_share=share,
        )
        for name, category, share in (
            ("tied-b", "chat_qa", Decimal(1) / Decimal(3)),
            ("easy", "chat_qa", Decimal("0.9")),
            ("near-hard", "chat_qa", Decimal("0.34")),
            ("tied-a", "chat_qa", Decimal(1) / Decimal(3)),
            ("unsolved", "chat_qa", Decimal(0)),
            ("math-hard", "math_code", Decimal("0.2")),
        )
    )
    threshold = Decimal(1) / Decimal(3)
    expected = {"chat_qa": ("tied-a", "tied-b", "unsolved"), "math_code": ("math-hard",)}
    assert derive_hard_question_ids(questions, max_solve_share=threshold) == expected
    assert derive_hard_question_ids(questions[::-1], max_solve_share=threshold) == expected
    # A percentile would split the tie; a lower threshold drops both tied questions.
    assert derive_hard_question_ids(questions, max_solve_share=Decimal("0.3"))["chat_qa"] == (
        "unsolved",
    )
    missing = questions[1].model_copy(update={"question_id": "unknown", "panel_solve_share": None})
    with pytest.raises(ValueError, match="missing panel solve share"):
        derive_hard_question_ids((*questions, missing), max_solve_share=threshold)
    easy_math = questions[-1].model_copy(update={"panel_solve_share": Decimal("0.5")})
    # A category without hard questions is allowed; the hard multiplier has no effect there.
    no_hard = derive_hard_question_ids((*questions[:-1], easy_math), max_solve_share=threshold)
    assert no_hard["math_code"] == ()
    with pytest.raises(ValueError, match="unique"):
        derive_hard_question_ids((*questions, questions[0]), max_solve_share=threshold)


def test_categories_without_hard_questions_still_decide():
    rules = policy(hard_question_ids=(), initial_hard_quality=None)
    decision = decide_challenge(rules, submission("cheap", cost="8"), (submission(),))
    assert decision.promoted
    for category in decision.categories.values():
        assert category.net_value.n_hard == 0 and category.net_value.hard_answer_delta == 0


@pytest.mark.parametrize(
    "reworded,min_pairs,enforce,failed,promoted",
    [
        ("0", 20, True, True, False),  # wording-keyed router is gated
        ("0", 20, False, True, True),  # report-only: probe losses leave net value alone
        ("1", 20, True, False, True),
        ("0", 100, True, False, True),  # too few pairs to judge
    ],
)
def test_wording_probe_gates_the_paired_drop_but_not_net_value(
    reworded, min_pairs, enforce, failed, promoted
):
    pairs = tuple((f"chat_qa-{i}", f"chat_qa-{i + 40}") for i in range(10, 50))
    probe = competition.WordingProbe(
        pairs=pairs, reference_drop=Decimal("0.02"), min_pairs=min_pairs, enforce=enforce
    )
    frozen = policy().model_copy(update={"wording": probe})
    frozen = CompetitionPolicy.model_validate(frozen.model_dump(mode="json"))
    grades = {qid: Decimal(1) for qid in ids("chat_qa")} | {
        reworded_id: Decimal(reworded) for _, reworded_id in pairs
    }
    challenger = submission("cheap", cost="8", overrides={"chat_qa": {"grades": grades}})
    king = submission(overrides={"chat_qa": {"quality": "1"}})
    decision = decide_challenge(frozen, challenger, (king,))
    assert (decision.wording.failed, decision.promoted) == (failed, promoted)
    assert decision.categories["chat_qa"].gates.quality == 1
    assert "wording" not in policy().model_dump(mode="json")


def test_cheaper_equal_quality_challenger_wins_and_breakdown_is_reported():
    decision = decide_challenge(policy(), submission("cheap", cost="8"), (submission(),))
    assert decision.promoted and decision.eligible
    chat = decision.categories["chat_qa"].net_value
    # $2 saved at $1 per answer over 100 questions is +2 points; agentic answers are worth $10.
    assert (chat.spend_delta_usd, chat.spend_points, chat.net_value) == (-2, -2, 2)
    assert decision.categories["agentic"].net_value.net_value == Decimal("0.2")
    assert decision.net_value == Decimal("1.55")
    assert decision.lower_bound == decision.net_value  # Uniform savings have no variance.
    assert decision.min_net_value == Decimal("0.2")


def test_net_value_is_antisymmetric_so_two_routers_cannot_beat_each_other():
    a = submission("a", overrides={"chat_qa": {"cost": "6"}, "agentic": {"cost": "40"}})
    b = submission("b", overrides={"math_code": {"cost": "4"}})
    ab, ba = decide_challenge(policy(), a, (b,)), decide_challenge(policy(), b, (a,))
    assert ab.net_value == -ba.net_value
    assert not (ab.promoted and ba.promoted)


def test_identical_router_never_wins_even_with_zero_minimum():
    rules = policy(net_value=config(min_net_value="0"))
    decision = decide_challenge(rules, submission("copy"), (submission(),))
    assert decision.eligible and decision.net_value == 0 and decision.lower_bound == 0
    assert not decision.promoted


def test_uncertain_gain_does_not_promote_but_the_bootstrap_is_deterministic():
    # One more regular answer per category (+1 point) at equal spend, but 98 questions differ.
    king_grades = {c: {q: Decimal(i >= 50) for i, q in enumerate(ids(c))} for c in CATEGORIES}
    noisy_grades = {c: {q: Decimal(2 <= i < 53) for i, q in enumerate(ids(c))} for c in CATEGORIES}
    rules = policy(
        net_value=config(floor={"basis": "none"}),
        initial_quality=Decimal("0.5"),
        initial_private_quality=Decimal("0"),
    )
    king = submission(overrides={c: {"grades": king_grades[c]} for c in CATEGORIES})
    noisy = submission("noisy", overrides={c: {"grades": noisy_grades[c]} for c in CATEGORIES})
    decision = decide_challenge(rules, noisy, (king,))
    assert decision.net_value > decision.min_net_value > decision.lower_bound
    assert not decision.promoted
    assert decide_challenge(rules, noisy, (king,)) == decision
    reseeded = policy(
        net_value=config(floor={"basis": "none"}, bootstrap={"samples": 2000, "seed": 7}),
        initial_quality=Decimal("0.5"),
        initial_private_quality=Decimal("0"),
    )
    assert decide_challenge(reseeded, noisy, (king,)).lower_bound != decision.lower_bound


@pytest.mark.parametrize("seconds,passes", [(1200, True), (1201, False)])
def test_time_gate_is_a_multiple_of_the_panel_leader_time(seconds, passes):
    candidate = submission("slow", cost="1", overrides={"agentic": {"seconds": seconds}})
    decision = decide_challenge(policy(), candidate, (submission(),))
    assert decision.promoted is passes
    if not passes:
        assert decision.reasons == ("agentic: category time limit exceeded",)
        assert decision.lower_bound is None and decision.net_value is not None


def test_per_category_time_override_only_changes_that_category():
    rules = policy(
        net_value=config(
            categories=NetValuePolicy().model_dump(mode="json")["categories"]
            | {"agentic": {"weight": "0.25", "value_regular_usd": "10", "time_multiplier": "3"}}
        )
    )
    fast_chat = submission("x", cost="1", overrides={"agentic": {"seconds": 1800}})
    assert decide_challenge(rules, fast_chat, (submission(),)).promoted
    slow_chat = submission("y", cost="1", overrides={"chat_qa": {"seconds": 1800}})
    assert decide_challenge(rules, slow_chat, (submission(),)).reasons == (
        "chat_qa: category time limit exceeded",
    )


@pytest.mark.parametrize("hard_quality,eligible", [("0.93", True), ("0.92", False)])
def test_hard_floor_cannot_be_offset_by_savings_once_enough_hard_questions(hard_quality, eligible):
    grades = {q: Decimal("0.95") for q in ids("chat_qa")}
    grades |= {"chat_qa-0": Decimal(hard_quality), "chat_qa-1": Decimal(hard_quality)}
    candidate = submission("cheap", cost="1", overrides={"chat_qa": {"grades": grades}})
    strict = policy(net_value=config(floor={"hard_floor_min_questions": 2}))
    decision = decide_challenge(strict, candidate, (submission(),))
    assert decision.eligible is eligible and decision.promoted is eligible
    if not eligible:
        assert decision.reasons == ("chat_qa: hard-question quality below floor",)
    # Below the minimum hard-question count only the weighted net value sees them.
    assert decide_challenge(policy(), candidate, (submission(),)).eligible


def test_first_submission_must_pass_the_panel_leader_floor():
    first = CompetitionHistory(policy=policy()).challenge(submission("weak", quality="0.92"))
    decision = first.challenges[0].decision
    assert not decision.promoted and first.champion() is None
    assert decision.reasons == (
        "chat_qa: quality below floor",
        "chat_qa: private-question quality below floor",
        "math_code: quality below floor",
        "math_code: private-question quality below floor",
        "agentic: quality below floor",
        "agentic: private-question quality below floor",
    )
    first = first.challenge(submission("ok", quality="0.93"))
    assert first.champion().run_id == "ok"
    assert first.challenges[-1].decision.reasons == ("first eligible overall champion",)
    assert first.challenges[-1].decision.net_value is None


def test_high_water_mark_rises_with_champion_lower_bounds_and_never_falls():
    rules = policy(initial_quality=Decimal("0.5"), initial_private_quality=Decimal("0"))
    mixed = {
        c: {"grades": {q: Decimal(i % 10 != 0) for i, q in enumerate(ids(c))}} for c in CATEGORIES
    }  # Mean 0.9 with variance, so its lower bound is below 0.9.
    history = CompetitionHistory(policy=rules)
    assert history.high_water_marks()["chat_qa"].quality == Decimal("0.5")
    history = history.challenge(submission("first", overrides=mixed))
    mark = history.high_water_marks()["chat_qa"].quality
    assert Decimal("0.8") < mark < Decimal("0.9")
    assert history.high_water_marks()["chat_qa"].hard_quality == Decimal("0.95")
    cheaper = {
        c: {"grades": {q: Decimal(i % 10 != 0 and i % 50 != 5) for i, q in enumerate(ids(c))}}
        | {"cost": "1"}
        for c in CATEGORIES
    }  # Mean 0.88: a cheaper champion whose lower bound is below the mark.
    worse = {
        c: {"grades": {q: Decimal(i % 10 > 1) for i, q in enumerate(ids(c))}, "cost": "1"}
        for c in CATEGORIES
    }  # Mean 0.8: within tau of 0.5, below the new mark's floor.
    history = history.challenge(submission("cheaper", overrides=cheaper))
    assert history.champion().run_id == "cheaper"
    assert history.high_water_marks()["chat_qa"].quality == mark
    history = history.challenge(submission("worse", overrides=worse))
    assert history.champion().run_id == "cheaper"
    assert "chat_qa: quality below floor" in history.challenges[-1].decision.reasons


def test_floor_basis_selects_the_reference():
    incumbent = submission(quality="0.90")
    challenger = submission("c", quality="0.89", cost="5")
    # An earlier stronger champion set the mark; under "incumbent" only the current one matters.
    lineage = (submission("old", quality="1"), incumbent)
    rules = {
        basis: policy(
            net_value=config(floor={"basis": basis}),
            initial_private_quality=Decimal("0.5"),
        )
        for basis in ("high_water_mark", "incumbent", "none")
    }
    by_mark = decide_challenge(rules["high_water_mark"], challenger, lineage)
    assert by_mark.categories["chat_qa"].gates.quality_reference == 1
    assert "chat_qa: quality below floor" in by_mark.reasons
    by_incumbent = decide_challenge(rules["incumbent"], challenger, lineage)
    gates = by_incumbent.categories["chat_qa"].gates
    assert (gates.quality_reference, gates.quality_floor) == (Decimal("0.9"), Decimal("0.875"))
    assert by_incumbent.eligible
    unchecked = decide_challenge(rules["none"], challenger, lineage)
    assert unchecked.categories["chat_qa"].gates.quality_floor is None and unchecked.eligible


def test_private_floor_rises_with_champions_like_the_main_floor():
    grades = {q: Decimal("0.95") for q in ids("math_code")}
    grades |= {"math_code-2": Decimal("0.9"), "math_code-3": Decimal("0.9")}
    candidate = submission("p", cost="1", overrides={"math_code": {"grades": grades}})
    # Starts at the panel leader's private quality (0.95 - 0.02).
    decision = decide_challenge(policy(), candidate, (submission(),))
    assert decision.reasons == ("math_code: private-question quality below floor",)
    assert decision.categories["math_code"].gates.private_quality_floor == Decimal("0.93")
    # A champion with perfect private answers raises the private mark to 1 (floor 0.98).
    perfect = {q: Decimal("0.95") for q in ids("math_code")}
    perfect |= {"math_code-2": Decimal("1"), "math_code-3": Decimal("1")}
    raised = (submission(), submission("sharp", overrides={"math_code": {"grades": perfect}}))
    near = dict(perfect) | {"math_code-2": Decimal("0.96"), "math_code-3": Decimal("0.96")}
    challenger = submission("near", cost="1", overrides={"math_code": {"grades": near}})
    gates = decide_challenge(policy(), challenger, raised).categories["math_code"].gates
    assert (gates.private_quality_reference, gates.private_quality_floor) == (1, Decimal("0.98"))
    assert "private-question quality below floor" in gates.failures[0]
    # Under floor basis "none" there is no private floor either.
    unchecked = policy(net_value=config(floor={"basis": "none"}))
    assert (
        decide_challenge(unchecked, challenger, raised)
        .categories["math_code"]
        .gates.private_quality_floor
        is None
    )


def test_anchor_basis_values_answers_at_m_times_anchor_cost():
    anchored = policy(
        net_value=config(
            value_basis="anchor",
            m="20",
            categories={
                "chat_qa": {"weight": "0.40"},
                "math_code": {"weight": "0.35"},
                "agentic": {"weight": "0.25"},
            },
        ),
        anchor_cost_per_question=Decimal("0.05"),
    )
    fixed = policy(
        net_value=config(
            categories={
                c: {"weight": w, "value_regular_usd": "1"}
                for c, w in zip(CATEGORIES, ("0.40", "0.35", "0.25"))
            }
        )
    )
    candidate, king = submission("c", cost="8"), (submission(),)
    a, f = decide_challenge(anchored, candidate, king), decide_challenge(fixed, candidate, king)
    assert a.categories["agentic"].net_value.value_usd == 1
    assert a.net_value == f.net_value == 2 and a.lower_bound == f.lower_bound
    with pytest.raises(ValueError, match="anchor cost in every category"):
        policy(net_value=anchored.net_value)


def timed(run_id="king", seconds_per_question=6, **settings):
    """A submission whose per-question time is recorded, as priced time requires."""
    ns = int(seconds_per_question * 1_000_000_000)
    return submission(
        run_id,
        overrides={
            c: {"question_time_ns": dict.fromkeys(ids(c), ns), "seconds": ns * N / 1e9}
            for c in CATEGORIES
        },
        **settings,
    )


def test_priced_time_charges_a_slower_challenger_in_answers():
    priced = policy(net_value=config(k="100"), anchor_time_per_question_ns=Decimal(6e9))
    # T = 100 * 6 s = 600 s per answer; 6 s more on every question costs 1 point.
    slower = decide_challenge(priced, timed("c", 12, cost="8"), (timed(),))
    category = slower.categories["chat_qa"].net_value
    # It saves 2 points of spend and pays 1 point of time.
    assert (category.spend_points, category.time_points) == (-2, 1)
    assert category.net_value == 1
    # Unpriced, the same challenger is only cheaper.
    unpriced = decide_challenge(policy(), timed("c", 12, cost="8"), (timed(),))
    assert unpriced.categories["chat_qa"].net_value.net_value == 2


def test_priced_time_requires_anchor_time_and_per_question_evidence():
    with pytest.raises(ValueError, match="anchor time in every category"):
        policy(net_value=config(k="100"))
    priced = policy(net_value=config(k="100"), anchor_time_per_question_ns=Decimal(6e9))
    decision = decide_challenge(priced, submission("c"), (timed(),))
    assert not decision.eligible
    assert "priced time requires per-question time" in decision.reasons
    assert digest(priced) != digest(policy(anchor_time_per_question_ns=Decimal(6e9)))


def test_policy_categories_must_match_net_value_categories():
    two = config(
        categories={
            "chat_qa": {"weight": "0.5", "value_regular_usd": "1"},
            "math_code": {"weight": "0.5", "value_regular_usd": "1"},
        }
    )
    with pytest.raises(ValueError, match="categories must match"):
        policy(net_value=two)


def test_every_configured_number_changes_the_policy_identity():
    base = digest(policy())
    for change in (
        {"m": "31"},
        {"hard_multiplier": "3"},
        {"min_net_value": "0.3"},
        {"floor": {"tau": {"share": "0.3", "min": "0.02", "max": "0.05"}}},
        {"bootstrap": {"samples": 2000, "seed": 1}},
        {"time": {"multiplier": "2.5"}},
    ):
        assert digest(policy(net_value=config(**change))) != base


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"cost_complete": False}, "unresolved"),
        ({"execution_complete": False}, "execution incomplete"),
        ({"integrity_pass": False}, "integrity check failed"),
        ({"grades": {"missing": Decimal(1)}}, "assigned question coverage differs"),
        ({"cost": "0"}, "positive measured cost and time required"),
        ({"execution_time_ns": 0}, "positive measured cost and time required"),
        ({"cohort_sha256": "c" * 64}, "incompatible cohort"),
    ],
)
def test_invalid_evidence_in_any_category_cannot_win(changes, reason):
    result = decide_challenge(policy(), submission("bad", **changes), (submission(),))
    assert not result.eligible and not result.promoted and result.net_value is None
    assert any(reason in error for error in result.reasons)


def test_missing_category_blocks_the_whole_challenge():
    partial = SubmissionMeasurement(categories=(measured("partial"),))
    decision = decide_challenge(policy(), partial, (submission(),))
    assert not decision.eligible and decision.reasons == ("submission must cover every category",)


def test_question_spend_must_cover_and_sum_to_category_cost():
    row = measured()
    with pytest.raises(ValueError, match="sum to the category cost"):
        CategoryMeasurement.model_validate(row.model_dump() | {"cost_usd": Decimal("10.01")})
    with pytest.raises(ValueError, match="cover exactly"):
        CategoryMeasurement.model_validate(
            row.model_dump() | {"question_spend_usd": {"chat_qa-0": row.cost_usd}}
        )


def test_duplicate_delivery_and_conflicting_evidence():
    history = CompetitionHistory(policy=policy()).challenge(submission())
    assert history.challenge(submission()) == history
    with pytest.raises(ValueError, match="conflicts"):
        history.challenge(submission(cost="1"))


def test_snapshot_has_one_verified_champion_and_no_category_rewards():
    history = CompetitionHistory(policy=policy()).challenge(submission())
    snapshot = CompetitionSnapshot(history=history, proposed_champion_run_id="king")
    raw = {"id": digest(snapshot), "payload": snapshot.model_dump(mode="json")}
    assert CompetitionSnapshot.read(raw, policy_sha256=digest(history.policy)) == snapshot
    assert not snapshot.should_set_weights and "allocations" not in raw["payload"]
    with pytest.raises(ValueError, match="pinned"):
        CompetitionSnapshot.read(raw, policy_sha256="b" * 64)
    raw["payload"]["history"]["challenges"][0]["decision"]["promoted"] = False
    with pytest.raises(ValueError, match="replay"):
        CompetitionSnapshot.model_validate(raw["payload"])


def test_categories_cannot_mix_different_miners_or_runs():
    with pytest.raises(ValueError, match="one submission"):
        SubmissionMeasurement(categories=(measured("a"), measured("b", category="math_code")))


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-1"])
def test_nonfinite_or_negative_costs_are_rejected(bad):
    with pytest.raises(ValueError):
        measured(spend={q: Decimal(bad) for q in ids("chat_qa")})


def test_json_object_key_order_does_not_change_replay():
    candidate = submission(
        "mixed",
        overrides={
            "chat_qa": {"cost": "7.19", "seconds": 451},
            "math_code": {"cost": "10.73", "seconds": 631},
            "agentic": {"cost": "8.37", "seconds": 583},
        },
    )
    history = CompetitionHistory(policy=policy()).challenge(submission()).challenge(candidate)
    assert history.champion() == candidate
    # JSONB/database and proxy serializers may reorder object keys.
    raw = json.loads(history.model_dump_json())

    def reverse_objects(value):
        if isinstance(value, dict):
            return {k: reverse_objects(v) for k, v in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reverse_objects(v) for v in value]
        return value

    for reordered in (reverse_objects(raw), json.loads(json.dumps(raw, sort_keys=True))):
        restored = CompetitionHistory.model_validate(reordered)
        assert restored.champion() == history.champion()


def kings(*costs):
    history = CompetitionHistory(policy=policy())
    for index, cost in enumerate(costs):
        history = history.challenge(submission(f"miner-{index}", cost=cost))
    return history


def test_last_three_kings_receive_85_10_and_5_percent():
    history = kings("10", "8", "6", "4").challenge(submission("loser", cost="20"))
    assert [king.miner_id for king in history.lineage()] == [
        "miner-0",
        "miner-1",
        "miner-2",
        "miner-3",
    ]
    weights = CompetitionResult(netuid=7, history=history).weights()
    assert weights == pytest.approx({"miner-3": 0.85, "miner-2": 0.10, "miner-1": 0.05})
    assert "miner-0" not in weights
    assert "loser" not in weights
    assert CompetitionResult(netuid=7, history=history, paused=True).weights() == {}


def test_missing_older_king_share_stays_with_the_current_king():
    solo = CompetitionResult(netuid=7, history=kings("10")).weights()
    assert solo == pytest.approx({"miner-0": 1.0})
    assert CompetitionResult(netuid=7, history=kings("10", "8")).weights() == pytest.approx(
        {"miner-1": 0.90, "miner-0": 0.10}
    )


def test_a_hotkey_that_reclaims_the_throne_keeps_both_seats():
    history = CompetitionHistory(policy=policy())
    for run_id, miner, cost in (
        ("run-a", "miner-a", "10"),
        ("run-b", "miner-b", "8"),
        ("run-a2", "miner-a", "6"),
    ):
        history = history.challenge(submission(run_id, miner_id=miner, cost=cost))
    assert CompetitionResult(netuid=7, history=history).weights() == pytest.approx(
        {"miner-a": 0.90, "miner-b": 0.10}
    )


@pytest.fixture
def decisions(monkeypatch):
    """Count challenge decisions with empty replay caches."""
    monkeypatch.setattr(competition, "_decisions", competition.OrderedDict())
    monkeypatch.setattr(competition, "_replays", competition.OrderedDict())
    calls = []
    real = competition.decide_challenge

    def counted(*args, **kwargs):
        calls.append(args[1].run_id)
        return real(*args, **kwargs)

    monkeypatch.setattr(competition, "decide_challenge", counted)
    return calls


def tampered(history, index, **decision):
    raw = history.model_dump(mode="json")
    raw["challenges"][index]["decision"].update(decision)
    return CompetitionHistory.model_validate(raw)


def test_appending_and_reading_decide_each_challenge_once(decisions):
    history = kings("10", "8", "12", "6", "4")
    assert len(decisions) == 5
    for _ in range(3):
        assert history.champion().run_id == "miner-4"
        history.high_water_marks()
        CompetitionResult(netuid=7, history=history).weights()
    assert len(decisions) == 5


def test_same_content_replays_from_cache_and_equality_ignores_verification(decisions):
    history = kings("10", "8", "12")
    raw = history.model_dump(mode="json")
    unverified = CompetitionHistory.model_validate(raw)
    assert unverified == history
    assert unverified.lineage() == history.lineage() and len(decisions) == 3
    assert unverified == history and unverified.model_dump(mode="json") == raw
    # A cached decision still has to match what is stored.
    with pytest.raises(ValueError, match="replay"):
        tampered(history, 1, promoted=False).champion()
    assert len(decisions) == 3


def test_incremental_verification_replays_only_the_new_suffix(decisions):
    prior = kings("10", "8", "12")
    newer = prior.challenge(submission("miner-3", cost="6")).challenge(submission("late", cost="5"))
    raw = newer.model_dump(mode="json")
    decisions.clear()
    competition._decisions.clear()
    trusted = CompetitionHistory.model_validate(prior.model_dump(mode="json"))
    verified = CompetitionHistory.model_validate(raw).verify(since=trusted)
    assert decisions == ["miner-3", "late"]
    assert verified.champion() == newer.champion() and len(decisions) == 2
    with pytest.raises(ValueError, match="replay"):
        tampered(newer, 4, reasons=["forged"]).verify(since=trusted)
    with pytest.raises(ValueError, match="replay"):
        tampered(newer, 4, reasons=["forged"]).verify()
    with pytest.raises(ValueError, match="rolled back"):
        prior.verify(since=newer)
    with pytest.raises(ValueError, match="rolled back"):
        kings("10", "9", "12", "6").verify(since=prior)


def test_verified_prefix_is_trusted_as_given(decisions):
    """Only the caller's own verified history belongs in `since`; its decisions are not replayed."""
    forged = tampered(kings("10", "8"), 0, reasons=["forged"])
    extended = CompetitionHistory.model_validate(
        {
            **forged.model_dump(mode="json"),
            "challenges": [
                *forged.model_dump(mode="json")["challenges"],
                kings("10", "8", "6").model_dump(mode="json")["challenges"][-1],
            ],
        }
    )
    decisions.clear()
    assert extended.verify(since=forged).champion().run_id == "miner-2"
    # A history the caller stored itself can be adopted without any replay.
    stored = CompetitionHistory.model_validate(extended.model_dump(mode="json"))
    assert stored.verify(since=stored).champion().run_id == "miner-2"
    assert decisions == []
    with pytest.raises(ValueError, match="replay"):
        CompetitionHistory.model_validate(extended.model_dump(mode="json")).verify()


def test_respelled_policy_cannot_use_a_verified_prefix_or_poison_the_cache(decisions):
    """Equal-valued policies with different digests must not share cached decisions."""
    history = kings("10", "8", "12")
    raw = history.model_dump(mode="json")
    raw["policy"]["net_value"]["min_net_value"] = "0.20"
    respelled = CompetitionHistory.model_validate(raw).policy
    assert respelled == history.policy and digest(respelled) != digest(history.policy)
    forged = CompetitionHistory(policy=respelled)
    for event in history.challenges:
        forged = forged.challenge(event.measurement)
    competition._decisions.clear()
    with pytest.raises(ValueError, match="rolled back"):
        CompetitionHistory.model_validate(forged.model_dump(mode="json")).verify(
            since=CompetitionHistory(policy=history.policy)
        )
    honest = CompetitionHistory.model_validate(history.model_dump(mode="json"))
    assert honest.verify().champion() == history.champion()


@pytest.mark.parametrize(
    ("part", "path", "spelling"),
    [
        ("measurement", ("categories", 0, "grades", "chat_qa-0"), "0.950"),
        ("decision", ("min_net_value",), "0.20"),
    ],
)
def test_respelled_prefix_fails_fresh_and_incremental_verification(decisions, part, path, spelling):
    """Decimal equality ignores spelling; digests, and so stored decisions, do not."""
    old = kings("10", "8")
    raw = old.model_dump(mode="json")
    node = raw["challenges"][1][part]
    for key in path[:-1]:
        node = node[key]
    assert Decimal(node[path[-1]]) == Decimal(spelling) and node[path[-1]] != spelling
    node[path[-1]] = spelling
    new = CompetitionHistory.model_validate(raw)
    assert new.challenges == old.challenges
    with pytest.raises(ValueError, match="replay"):
        new.verify()
    with pytest.raises(ValueError, match="rolled back"):
        new.verify(since=old)
    shorter = CompetitionHistory(policy=old.policy, challenges=old.challenges[:1])
    with pytest.raises(ValueError, match="replay"):
        new.verify(since=shorter)

    snapshot = CompetitionSnapshot(history=old, proposed_champion_run_id="miner-1")
    payload = snapshot.model_dump(mode="json") | {"history": raw}
    with pytest.raises(ValueError, match="rolled back"):
        CompetitionSnapshot.read(
            {"id": digest(payload), "payload": payload}, policy_sha256=digest(policy()), since=old
        )
    now = datetime.now(timezone.utc)
    result = CompetitionResult(netuid=7, history=old).model_dump(mode="json") | {"history": raw}
    published = {"id": digest(result), "confirmed_at": now.isoformat(), "payload": result}
    for since in (old, None):
        with pytest.raises(ValueError, match="rolled back" if since else "replay"):
            CompetitionPublication.read(
                published,
                cohort=CompetitionCohort(netuid=7, policy_sha256=digest(policy())),
                now=now,
                max_age_s=60,
                since=since,
            )


def test_records_are_deeply_immutable_so_verification_cannot_go_stale(decisions):
    history = kings("10", "8")
    assert history.verify().champion().run_id == "miner-1"
    with pytest.raises(TypeError):
        history.challenges[0].measurement.categories[0].grades["chat_qa-0"] = Decimal("0")
    row = history.challenges[1].measurement.categories[0]
    mappings = (
        row.grades,
        row.question_spend_usd,
        history.challenges[1].decision.categories,
        history.policy.net_value.categories,
    )
    mutations = (
        lambda m, key: m.__setitem__(key, m[key]),
        lambda m, key: m.__delitem__(key),
        lambda m, key: m.pop(key),
        lambda m, key: m.popitem(),
        lambda m, key: m.setdefault("new", m[key]),
        lambda m, key: m.update({key: m[key]}),
        lambda m, key: m.__ior__({key: m[key]}),
        lambda m, key: m.clear(),
    )
    for mapping in mappings:
        key = next(iter(mapping))
        for mutate in mutations:
            with pytest.raises(TypeError):
                mutate(mapping, key)
    assert history.verify().champion().run_id == "miner-1" and len(decisions) == 2

    # Copies stay frozen; dumps stay plain, mutable JSON-compatible dicts.
    for copied in (
        row.model_copy(update={"grades": dict(row.grades)}),
        copy.deepcopy(row),
        pickle.loads(pickle.dumps(row)),
    ):
        assert copied == row and hash(copied) == hash(row)
        with pytest.raises(TypeError):
            copied.grades["chat_qa-0"] = Decimal("0")
    dumped = history.model_dump()
    dumped["policy"]["net_value"]["categories"].clear()
    assert type(dumped["challenges"][1]["measurement"]["categories"][0]["grades"]) is dict


def test_collected_histories_release_their_verified_lineage(decisions):
    import gc

    raw = kings("10", "8").model_dump(mode="json")
    competition._replays.clear()
    for _ in range(5):
        CompetitionHistory.model_validate(raw).verify()
    gc.collect()
    kept = CompetitionHistory.model_validate(raw).verify()
    assert list(competition._replays) == [id(kept)]


def test_snapshot_read_can_verify_only_challenges_after_an_accepted_history(decisions):
    prior = kings("10", "8")
    history = prior.challenge(submission("miner-2", cost="6"))
    snapshot = CompetitionSnapshot(history=history, proposed_champion_run_id="miner-2")
    raw = {"id": digest(snapshot), "payload": snapshot.model_dump(mode="json")}
    competition._decisions.clear()
    decisions.clear()
    read = CompetitionSnapshot.read(raw, policy_sha256=digest(policy()), since=prior)
    assert read == snapshot and decisions == ["miner-2"]
    raw["payload"]["history"]["challenges"][-1]["decision"]["promoted"] = False
    raw["id"] = digest(raw["payload"])
    with pytest.raises(ValueError, match="replay"):
        CompetitionSnapshot.read(raw, policy_sha256=digest(policy()), since=prior)


def test_retained_timing_labels_do_not_control_admission_or_change_evidence():
    # Old receipts/policies remain digest-stable, but their retired labels cannot
    # introduce another compatibility switch alongside the pack's identity.
    stored_policy = policy().model_dump(mode="json")
    stored_run = submission().model_dump(mode="json")
    for rule in stored_policy["categories"]:
        rule["timing_version"] = "inspect-working-v1"
    for measurement in stored_run["categories"]:
        measurement["timing_version"] = "category-active-wall-v1"
    restored_policy = CompetitionPolicy.model_validate(stored_policy)
    restored_run = SubmissionMeasurement.model_validate(stored_run)
    assert digest(restored_policy) == digest(stored_policy)
    assert digest(restored_run) == digest(stored_run)
    evidence_ids = {"policy_sha256", "challenger_sha256", "incumbent_sha256"}
    assert decide_challenge(restored_policy, restored_run).model_dump(exclude=evidence_ids) == (
        decide_challenge(policy(), submission()).model_dump(exclude=evidence_ids)
    )


def test_new_policy_and_measurement_records_have_no_timing_label():
    for row in (
        *policy().model_dump(mode="json")["categories"],
        *submission().model_dump(mode="json")["categories"],
    ):
        assert "timing_version" not in row
    raw = policy().model_dump(mode="json")
    assert digest(CompetitionPolicy.model_validate(raw)) == digest(raw)


def test_previous_format_challenge_history_replays_with_the_same_commitment():
    # Generated by the pre-spend competition code. Loading must not rewrite the digest.
    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "legacy-competition-history.json").read_text()
    )
    history = CompetitionHistory.model_validate(raw)
    assert digest(history) == digest(raw)
    assert history.verify().champion().miner_id == "miner-1"
