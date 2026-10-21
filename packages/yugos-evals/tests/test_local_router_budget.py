"""Candidate and grader admissions must consume the same local allowance."""

import time
from decimal import Decimal

import pytest

from yugos.inference.actions import CallAction, ModelChoice
from yugos.inference.calls import CallError, CallLimits, ModelResponse, QuestionLimitExceeded
from yugos.inference.chat import ModelInput
from yugos.inspect_engine import MemoryReceipts, RouterCall, SampleScope
from yugos_evals.local.router import LocalRouter


@pytest.mark.parametrize("service_first", [False, True])
@pytest.mark.parametrize(
    "cap,error,match",
    [
        ({"max_cost_usd": 0.15}, CallError, "evaluation budget"),
        ({"max_question_cost_usd": 0.15}, QuestionLimitExceeded, "question budget"),
        ({"max_calls_per_question": 1}, QuestionLimitExceeded, "question call limit"),
    ],
)
def test_candidate_and_grader_share_admission_limits(cap, error, match, service_first):
    class Provider:
        def __init__(self):
            self.requests = []

        def quote(self, request):
            return 0.1

        def complete(self, request):
            self.requests.append(request)
            return ModelResponse(text="answer", cost_usd=0.1, latency_s=0)

    candidate, grader = Provider(), Provider()
    router = LocalRouter(
        session=None,
        run_id="local",
        router_client=candidate,
        service_client=grader,
        limits=CallLimits(**{"max_cost_usd": 1, "max_question_cost_usd": 1, **cap}),
        receipts=MemoryReceipts(),
    )
    call = RouterCall(
        submission_id="test",
        scope=SampleScope(sample_uuid="uuid", sample_id="q1", epoch=1, eval_id="eval"),
        request=ModelInput.from_text("question"),
    )
    action = CallAction(model=ModelChoice(provider_id="test", model_id="test"), input=call.request)
    meters = [router.meter, router.service_meter]
    if service_first:
        meters.reverse()
    router._dispatch(meters[0], action, call, time.monotonic() + 10)
    with pytest.raises(error, match=match):
        router._dispatch(meters[1], action, call, time.monotonic() + 10)
    assert len(candidate.requests) + len(grader.requests) == 1
    assert sum(c.response.spend.usd for c in router.meter.budget.calls) == Decimal("0.1")
