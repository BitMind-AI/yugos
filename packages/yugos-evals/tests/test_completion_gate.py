"""The completion gate grades an agent stopped at its message limit mid tool call."""

import pytest
from evals_fakes import task_state
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import scorer
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool

from yugos.evaluation.grades import GradeCode
from yugos_evals.scorers._outcome import completion_gate, to_score


def tool_call_state(messages, message_limit):
    state = task_state("", stop_reason="tool_calls")
    state.messages = [ChatMessageUser(content="q") for _ in range(messages)]
    state.message_limit = message_limit
    return state


def test_tool_call_at_the_message_limit_is_graded_incorrect():
    outcome = completion_gate(tool_call_state(20, 20))
    assert outcome.code == GradeCode.INCORRECT
    assert outcome.status == "graded"
    assert outcome.details == {"cause": "message_limit_reached", "stop_reason": "tool_calls"}


@pytest.mark.parametrize("messages,limit", [(19, 20), (5, None)])
def test_other_tool_call_endings_still_need_review(messages, limit):
    outcome = completion_gate(tool_call_state(messages, limit))
    assert outcome.code == GradeCode.NEEDS_REVIEW
    assert outcome.status == "unavailable"


def test_an_agent_stopped_by_inspect_at_its_message_limit_is_graded(tmp_path):
    @tool
    def lookup():
        async def run(key: str) -> str:
            """Look up a key.

            Args:
              key: The key.
            """
            return "value"

        return run

    @scorer(metrics=[])
    def gated():
        async def score(state, target):
            return to_score(completion_gate(state))

        return score

    calls = [ModelOutput.for_tool_call("mockllm/model", "lookup", {"key": "k"})] * 20
    task = Task(
        dataset=[Sample(input="q", target="x")],
        solver=[use_tools(lookup()), generate()],
        scorer=gated(),
        message_limit=6,
    )
    [log] = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=calls),
        display="none",
        log_dir=str(tmp_path),
    )
    [sample] = log.samples
    assert sample.limit.type == "message"
    [score] = sample.scores.values()
    assert score.value == 0
    assert score.metadata["code"] == GradeCode.INCORRECT
