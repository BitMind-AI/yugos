import hashlib
from decimal import Decimal

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.scorer import Score  # noqa: E402
from inspect_fakes import ScriptedRouter, exact_question, pack_task, reply, run  # noqa: E402

from yugos.competition import (  # noqa: E402
    CategoryRule,
    CompetitionPolicy,
    SubmissionMeasurement,
    decide_challenge,
)
from yugos.competition.net_value import CategoryValue, NetValuePolicy  # noqa: E402
from yugos.evaluation.grades import GradeCode, GradingResult  # noqa: E402
from yugos.evaluation.measurements import Outcome, QuestionMeasurement  # noqa: E402
from yugos.inspect_engine import (  # noqa: E402
    FairnessProfile,
    ProviderReceipt,
    SampleMeasurement,
    category_measurements_from_log,
    category_measurements_from_samples,
    competition_eval_options,
    measure_log,
    receipt_digest,
    router_model,
    run_competition_eval,
)

PACK = [
    exact_question("mmlu-1", "C", source_benchmark="mmlu_pro"),
    exact_question("mmlu-2", "J", source_benchmark="mmlu_pro"),
    exact_question("mmlu-3", "A", source_benchmark="mmlu_pro"),
    exact_question("gpqa-1", "B", source_benchmark="gpqa"),
    exact_question("gpqa-2", "D", source_benchmark="gpqa"),
    exact_question("gpqa-3", "A", source_benchmark="gpqa"),
]
COHORT = "c" * 64


def test_host_owned_work_clock_can_disable_inspect_wall_limits():
    options = competition_eval_options(FairnessProfile(
        attempt_timeout_s=None, call_timeout_s=None, working_limit_s=None,
    ))
    assert options["attempt_timeout"] is None
    assert options["timeout"] is None
    assert options["working_limit"] is None
    with pytest.raises(ValueError):
        FairnessProfile(working_limit_s=0)


def recorded_sample(qid, score, active_ns, *, unavailable=False):
    return SampleMeasurement(
        measurement=QuestionMeasurement(
            question_id=qid,
            workload_class="chat_qa",
            outcome=Outcome(score=score, cost_usd=0, latency_s=0),
        ),
        sample_uuid=f"{qid}-{active_ns}",
        grade=GradingResult(
            score=score,
            grader="fixture",
            code=GradeCode.GRADER_UNAVAILABLE
            if unavailable
            else GradeCode.CORRECT
            if score
            else GradeCode.INCORRECT,
            status="unavailable" if unavailable else "graded",
        ),
        spend_usd=Decimal("0.01"),
        active_ns=active_ns,
        failure_owner=None,
    )


def aggregate_samples(samples, **kwargs):
    return category_measurements_from_samples(
        samples,
        {"a": "chat_qa", "b": "chat_qa"},
        run_id="run",
        miner_id="miner",
        submission_sha256="a" * 64,
        cohort_sha256=COHORT,
        integrity_pass=True,
        **kwargs,
    )[0]


def test_retained_retries_keep_accepted_grades_and_all_work():
    measured = aggregate_samples(
        [
            recorded_sample("a", 1, 100),
            recorded_sample("b", 0, 200, unavailable=True),
            recorded_sample("b", 1, 300),
            recorded_sample("a", 0, 50),
        ]
    )
    assert measured.grades == {"a": Decimal(1), "b": Decimal(1)}
    assert measured.question_spend_usd == {"a": Decimal("0.02"), "b": Decimal("0.02")}
    assert measured.execution_time_ns == 650
    # Every attempt's work belongs to its question, so priced time sees retries too.
    assert measured.question_time_ns == {"a": 150, "b": 500}
    assert measured.execution_complete is True


def test_cumulative_billing_includes_work_without_a_completed_sample():
    measured = aggregate_samples(
        [recorded_sample("a", 1, 100)],
        question_spend_usd={"a": Decimal("0.03"), "b": Decimal("0.02")},
    )
    assert measured.cost_usd == Decimal("0.05")
    assert measured.grades == {"a": Decimal(1), "b": Decimal(0)}
    assert measured.execution_complete is False


@pytest.mark.parametrize("missing", ["time", "grade"])
def test_retained_missing_evidence_stays_incomplete(missing):
    measured = aggregate_samples(
        [
            recorded_sample("a", 1, 100),
            recorded_sample(
                "b",
                0 if missing == "grade" else 1,
                None if missing == "time" else 200,
                unavailable=missing == "grade",
            ),
        ]
    )
    assert measured.execution_complete is False


def answers(correct: set[str]):
    """Scripted final answers: the reference letter for `correct` IDs, else a wrong letter."""
    return {q.prompt: (q.expected_answer if q.question_id in correct else "Z") for q in PACK}


def measure(log, router, run_id):
    return SubmissionMeasurement(
        categories=category_measurements_from_log(
            log,
            router.receipts.receipts(),
            {q.question_id: q.workload_class for q in PACK},
            run_id=run_id,
            miner_id=run_id,
            submission_sha256=hashlib.sha256(run_id.encode()).hexdigest(),
            cohort_sha256=COHORT,
            integrity_pass=True,
        )
    )


def test_router_runs_become_a_net_value_decision(tmp_path):
    # Catches a break anywhere between Inspect execution and the competition decision.
    incumbent = ScriptedRouter(answers({"mmlu-1", "gpqa-1"}), cost_usd="0.001")
    challenger = ScriptedRouter(answers({q.question_id for q in PACK}), cost_usd="0.002")
    task = pack_task(PACK, name="pack")
    king = measure(run(task, incumbent, tmp_path, submission_id="king"), incumbent, "king")
    rival = measure(run(task, challenger, tmp_path, submission_id="rival"), challenger, "rival")

    [chat] = rival.categories
    assert chat.execution_complete and chat.integrity_pass
    assert chat.grades == {q.question_id: Decimal(1) for q in PACK}
    assert chat.question_spend_usd == {q.question_id: Decimal("0.002") for q in PACK}
    assert chat.execution_time_ns > 0
    [chat] = king.categories
    assert sum(chat.grades.values()) == 2

    policy = CompetitionPolicy(
        categories=(
            CategoryRule(
                category="chat_qa",
                cohort_sha256=COHORT,
                question_ids=tuple(q.question_id for q in PACK),
                initial_quality=Decimal(0),
                panel_leader_time_ns=60_000_000_000,
            ),
        ),
        net_value=NetValuePolicy(
            categories={"chat_qa": CategoryValue(weight=Decimal(1), value_regular_usd=Decimal(1))}
        ),
    )
    assert "timing_version" not in chat.model_dump(mode="json")
    assert "timing_version" not in policy.categories[0].model_dump(mode="json")
    first = decide_challenge(policy, king)
    assert first.promoted
    decision = decide_challenge(policy, rival, [king])
    assert decision.promoted, decision.reasons
    assert decision.categories["chat_qa"].net_value.spend_delta_usd == Decimal("0.006")


def test_spend_comes_from_receipts_and_must_be_attributable(tmp_path):
    # Catches Inspect usage estimates replacing metered spend, or orphan receipts being dropped.
    question = exact_question("q", "Paris")
    router = ScriptedRouter({question.prompt: "Paris"}, cost_usd="0.125")
    log = run(pack_task([question], name="pack"), router, tmp_path)
    [row] = measure_log(log, router.receipts.receipts())
    assert row.spend_usd == Decimal("0.125") and row.measurement.outcome.cost_usd == 0.125

    orphan = router.receipts.receipts()[0].model_copy(update={"sample_uuid": "elsewhere"})
    for isolate in (False, True):
        with pytest.raises(ValueError, match="attributed"):
            measure_log(log, [orphan], isolate_errors=isolate)


def test_a_missing_score_is_rejected_rather_than_measured_as_zero(tmp_path):
    # Catches a mistyped scorer name or an unscored log becoming a valid all-zero result.
    question = exact_question("q", "Paris")
    router = ScriptedRouter({question.prompt: "Paris"})
    log = run(pack_task([question], name="pack"), router, tmp_path)
    [row] = measure_log(log, router.receipts.receipts(), scorer="exact")
    assert row.measurement.outcome.score == 1
    with pytest.raises(ValueError, match="no score from scorer 'yugos-scorer'"):
        measure_log(log, router.receipts.receipts(), scorer="yugos-scorer")
    unscored = log.model_copy(
        update={"samples": [s.model_copy(update={"scores": None}) for s in log.samples]}
    )
    with pytest.raises(ValueError, match="no score"):
        measure_log(unscored, router.receipts.receipts())


def rescored(log, score, **metadata):
    """`log` with every sample's authoritative score replaced by `score`."""
    samples = [
        s.model_copy(update={"scores": {"exact": score}, "metadata": {**s.metadata, **metadata}})
        for s in log.samples
    ]
    return log.model_copy(update={"samples": samples})


def chat_categories(log, router, question_ids):
    return category_measurements_from_log(
        log,
        router.receipts.receipts(),
        dict.fromkeys(question_ids, "chat_qa"),
        run_id="r",
        miner_id="m",
        submission_sha256="a" * 64,
        cohort_sha256=COHORT,
        integrity_pass=True,
    )


@pytest.mark.parametrize("finish_reason", ["stop", "content_filter"])
def test_an_unscored_sample_needs_review_and_voids_only_its_category(tmp_path, finish_reason):
    # Catches one grader failure aborting measurement for a whole category, or it
    # becoming the router's graded zero instead of an incomplete, ineligible run.
    graded, failed = exact_question("graded", "Paris"), exact_question("failed", "Paris")
    router = ScriptedRouter(
        {graded.prompt: "Paris", failed.prompt: reply("Paris", finish_reason=finish_reason)}
    )
    log = run(pack_task([graded, failed], name="pack"), router, tmp_path)
    [ok, bad] = sorted(log.samples, key=lambda s: s.id != "graded")
    unscored = Score.unscored(reason="grader_failed", metadata={"code": "needs_review"})
    log = log.model_copy(
        update={"samples": [ok, bad.model_copy(update={"scores": {"exact": unscored}})]}
    )

    rows = {
        row.measurement.question_id: row for row in measure_log(log, router.receipts.receipts())
    }
    grade = rows["failed"].grade
    assert (grade.code, grade.status, grade.score) == ("needs_review", "unavailable", 0.0)
    assert grade.details["cause"] == "grader_failed"
    assert rows["failed"].failure_owner is None
    assert rows["failed"].measurement.grade_rule == "needs_review"
    assert rows["graded"].grade.code == "correct"

    [chat] = chat_categories(log, router, ["graded", "failed"])
    assert chat.grades == {"graded": Decimal(1), "failed": Decimal(0)}
    assert not chat.execution_complete
    rule = CategoryRule(
        category="chat_qa",
        cohort_sha256=COHORT,
        question_ids=("graded", "failed"),
        initial_quality=Decimal(0),
        panel_leader_time_ns=60_000_000_000,
    )
    assert "execution incomplete" in rule.measurement_errors(chat)


def test_an_unscored_truncated_answer_is_the_routers_settled_zero(tmp_path):
    # Catches a router's truncation voiding the run instead of counting as its non-answer.
    question = exact_question("q", "Paris")
    router = ScriptedRouter({question.prompt: reply("Par", finish_reason="length")})
    log = rescored(
        run(pack_task([question], name="pack"), router, tmp_path),
        Score.unscored(reason="scoring_failed"),
    )
    [row] = measure_log(log, router.receipts.receipts())
    assert (row.grade.code, row.grade.status) == ("truncated", "skipped")
    assert row.measurement.non_answer and row.failure_owner is None
    [chat] = chat_categories(log, router, ["q"])
    assert chat.execution_complete and chat.grades == {"q": Decimal(0)}


@pytest.mark.parametrize(
    "score,key,match",
    [
        (Score(value=float("nan")), None, "reason"),
        (Score(value=1.5), None, r"\[0, 1\]"),
        (Score(value=-0.5), None, r"\[0, 1\]"),
        (Score(value=float("inf")), None, r"\[0, 1\]"),
        (Score(value="garbage"), None, "not a grade"),
        (Score(value="nan"), None, "not a grade"),
        (Score(value={"accuracy": None}), "accuracy", "not a grade"),
        (Score(value={"accuracy": 1.0}), "pass", "no key 'pass'"),
        (Score(value={"accuracy": 1.0}), None, "scalar"),
    ],
)
def test_malformed_scores_fail_closed(tmp_path, score, key, match):
    # Catches a broken scorer's output being measured as a grade or a silent zero.
    question = exact_question("q", "Paris")
    router = ScriptedRouter({question.prompt: "Paris"})
    log = run(pack_task([question], name="pack"), router, tmp_path)
    metadata = {"yugos_score_key": key} if key else {}
    invalid = rescored(log, score, **metadata)
    with pytest.raises(ValueError, match=match):
        measure_log(invalid, router.receipts.receipts())
    [row] = measure_log(invalid, router.receipts.receipts(), isolate_errors=True)
    assert row.failure_owner == "infrastructure" and row.grade is None
    assert row.measurement.outcome.unrealized
    assert row.spend_usd == router.receipts.receipts()[0].cost_usd


def test_infrastructure_errors_leave_the_category_incomplete(tmp_path):
    # Catches operator failures being scored as the miner's zero instead of a void run.
    fine, broken = exact_question("fine", "Paris"), exact_question("broken", "Paris")
    router = ScriptedRouter({fine.prompt: "Paris", broken.prompt: KeyError("proxy state lost")})
    log = run(pack_task([fine, broken], name="pack"), router, tmp_path)
    [chat] = category_measurements_from_log(
        log,
        router.receipts.receipts(),
        {"fine": "chat_qa", "broken": "chat_qa"},
        run_id="r",
        miner_id="m",
        submission_sha256="a" * 64,
        cohort_sha256=COHORT,
        integrity_pass=True,
    )
    assert log.status == "success"
    assert chat.grades == {"fine": Decimal(1), "broken": Decimal(0)}
    assert not chat.execution_complete


def test_receipt_digest_is_stable_across_reruns_and_replays(tmp_path):
    # Catches digests that depend on random IDs, timestamps or log bytes.
    task = pack_task(PACK, name="pack")
    digests, uuids = [], []
    for attempt in ("first", "second"):
        router = ScriptedRouter(answers({"mmlu-1", "gpqa-2"}))
        log = run(task, router, tmp_path / attempt)
        receipts = router.receipts.receipts()
        uuids.append({s.uuid for s in log.samples})
        digests.append(receipt_digest(log, receipts, dataset_revision="pack-v1"))
        replay = read_eval_log(log.location)
        digests.append(receipt_digest(replay, receipts, dataset_revision="pack-v1"))
    assert len(set(digests)) == 1
    assert uuids[0].isdisjoint(uuids[1])

    [first, *rest] = receipts
    changed = [first.model_copy(update={"cost_usd": Decimal("0.5")}), *rest]
    assert receipt_digest(log, changed, dataset_revision="pack-v1") != digests[0]
    assert receipt_digest(log, receipts, dataset_revision="pack-v2") != digests[0]
    other = ScriptedRouter(answers(set()))
    other_log = run(task, other, tmp_path / "other")
    assert (
        receipt_digest(other_log, other.receipts.receipts(), dataset_revision="pack-v1")
        != (digests[0])
    )
    edited = log.model_copy(update={"invalidated": True})
    with pytest.raises(ValueError, match="edited"):
        receipt_digest(edited, receipts, dataset_revision="pack-v1")


def test_every_submission_runs_with_identical_fairness_settings(tmp_path):
    # Catches per-submission drift in concurrency, retries, limits or error handling.
    question = exact_question("q", "Paris")
    task = pack_task([question], name="pack")
    logs = [
        run(task, ScriptedRouter({question.prompt: text}), tmp_path, submission_id=name)
        for name, text in (("sub-a", "Paris"), ("sub-b", "Rome"))
    ]
    configs = [(log.eval.config, log.plan.config) for log in logs]
    assert configs[0] == configs[1]
    eval_config, generate_config = configs[0]
    assert generate_config.adaptive_connections is False
    assert generate_config.max_connections == eval_config.max_samples
    assert generate_config.max_retries is not None and generate_config.timeout is not None
    assert eval_config.fail_on_error is False and eval_config.retry_on_error == 0
    assert eval_config.working_limit is not None and eval_config.time_limit is None
    assert competition_eval_options() == competition_eval_options(FairnessProfile())

    model = router_model(ScriptedRouter({}), submission_id="sub-c")
    with pytest.raises(ValueError, match="fairness profile"):
        run_competition_eval(task, model, log_dir=str(tmp_path), max_connections=100)


def test_receipt_model_rejects_negative_spend():
    with pytest.raises(ValueError):
        ProviderReceipt(
            receipt_id="r",
            sample_uuid="s",
            provider="p",
            model="m",
            cost_usd=Decimal("-0.01"),
            started_ns=0,
            ended_ns=1,
        )
