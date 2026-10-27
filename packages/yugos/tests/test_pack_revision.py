"""Pack amendments preserve decisions and reward seats while replacing comparisons."""

from decimal import Decimal

import pytest

from yugos.competition import (
    CategoryMeasurement,
    CategoryRule,
    CompetitionHistory,
    CompetitionPolicy,
    CompetitionResult,
    PackRevision,
    SubmissionMeasurement,
    digest,
)
from yugos.competition.net_value import NetValuePolicy
from yugos.spend import KnownSpend


def policy(ids=("old",)):
    config = NetValuePolicy().model_dump(mode="json")
    config["categories"] = {"chat_qa": config["categories"]["chat_qa"] | {"weight": "1"}}
    return CompetitionPolicy(
        net_value=NetValuePolicy.model_validate(config),
        categories=(
            CategoryRule(
                category="chat_qa",
                cohort_sha256="b" * 64,
                question_ids=ids,
                initial_quality="0.9",
                panel_leader_time_ns=10_000_000_000,
            ),
        ),
    )


def measured(run="a", miner=None, ids=("old",), price="1"):
    spend = {q: Decimal(price) for q in ids}
    return SubmissionMeasurement(
        categories=(
            CategoryMeasurement(
                run_id=run,
                miner_id=miner or run,
                submission_sha256="a" * 64,
                cohort_sha256="b" * 64,
                category="chat_qa",
                grades={q: "1" for q in ids},
                question_spend_usd=spend,
                cost_usd=sum(spend.values()),
                spend=KnownSpend(usd=sum(spend.values())),
                question_time_ns={q: 1000 for q in ids},
                execution_time_ns=1000 * len(ids),
                execution_complete=True,
                integrity_pass=True,
            ),
        )
    )


def upgraded(previous=None):
    previous = previous or CompetitionHistory(policy=policy()).challenge(measured())
    baseline = measured("top-up", miner=previous.champion().miner_id, ids=("old", "new"))
    return CompetitionHistory(
        policy=policy(("old", "new")),
        pack_revision=PackRevision(
            contest="contest",
            previous_pack_sha256="1" * 64,
            pack_sha256="2" * 64,
            previous=previous,
            baseline=baseline,
        ),
    )


def test_revision_does_not_promote_and_next_challenge_uses_expanded_king():
    previous = CompetitionHistory(policy=policy()).challenge(measured())
    new = upgraded(previous).verify(since=previous)
    assert new.lineage() == previous.lineage()
    assert new.champion().run_id == "top-up"
    assert new.comparison_lineage() == (new.pack_revision.baseline,)
    assert CompetitionResult(netuid=34, history=new).weights() == {"a": 1.0}
    next_history = new.challenge(measured("b", ids=("old", "new"), price="0.1"))
    assert next_history.challenges[-1].decision.incumbent_sha256 == digest(new.champion())
    assert next_history.challenges[-1].decision.promoted
    assert CompetitionResult(netuid=34, history=next_history).weights() == {"b": 0.9, "a": 0.1}
    assert next_history.pack_revision.previous.challenges == previous.challenges
    CompetitionHistory.model_validate(next_history.model_dump(mode="json")).verify(since=previous)


def test_revision_preserves_three_reward_seats():
    old = CompetitionHistory(policy=policy())
    for name, price in (("a", "3"), ("b", "2"), ("c", "1")):
        old = old.challenge(measured(name, price=price))
    new = upgraded(old).verify()
    assert CompetitionResult(netuid=34, history=new).weights() == {"c": 0.85, "b": 0.1, "a": 0.05}
    assert len(new.lineage()) == 3


@pytest.mark.parametrize(
    "change", ["identity", "grade", "spend", "time", "removed", "scoring", "history"]
)
def test_revision_rejects_rewritten_evidence_or_rules(change):
    new = upgraded()
    raw = new.model_dump(mode="json")
    baseline = raw["pack_revision"]["baseline"]["categories"][0]
    if change == "identity":
        baseline["miner_id"] = "other"
    elif change == "grade":
        baseline["grades"]["old"] = "0"
    elif change == "spend":
        baseline["question_spend_usd"].update(old="0.5", new="1.5")
    elif change == "time":
        baseline["question_time_ns"].update(old=500, new=1500)
    elif change == "removed":
        raw["policy"]["categories"][0]["question_ids"] = ["new"]
    elif change == "scoring":
        raw["policy"]["net_value"]["m"] = "123"
    else:
        raw["pack_revision"]["previous"]["challenges"][0]["decision"]["reasons"] = ["forged"]
    with pytest.raises(ValueError):
        CompetitionHistory.model_validate(raw).verify()


def test_revision_rejects_rollback_even_after_cached_verification():
    new = upgraded().verify()
    unrelated = CompetitionHistory(policy=policy()).challenge(measured("different"))
    with pytest.raises(ValueError, match="prefix"):
        new.verify(since=unrelated)


def test_full_reset_still_has_no_king_or_inherited_history():
    old = upgraded().verify()
    reset = CompetitionHistory(policy=old.policy)
    assert reset.lineage() == () and reset.champion() is None
    assert CompetitionResult(netuid=34, history=reset).weights() == {}
    assert "pack_revision" not in reset.model_dump(mode="json")
    assert reset.challenge(measured("fresh", ids=("old", "new"))).champion().miner_id == "fresh"


def test_multiple_revisions_verify_the_original_prefix_without_extra_reward_seats():
    first = upgraded().verify()
    second = CompetitionHistory(
        policy=policy(("old", "new", "next")),
        pack_revision=PackRevision(
            contest="contest",
            previous_pack_sha256="2" * 64,
            pack_sha256="3" * 64,
            previous=first,
            baseline=measured("second-top-up", miner="a", ids=("old", "new", "next")),
        ),
    ).verify(since=first.pack_revision.previous)
    assert len(second.epochs()) == 3
    assert len(second.lineage()) == 1 and second.champion().run_id == "second-top-up"
    with pytest.raises(ValueError, match="duplicate"):
        second.challenge(second.pack_revision.baseline)
