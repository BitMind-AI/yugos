"""Tool controls survive freezing and run with private values and continuation intact."""

import json

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.log import read_eval_log

from yugos.evaluation import Question
from yugos.inference.chat import PROVIDER_STATE_KEY, ProviderState, provider_state, public_message
from yugos.inspect_engine import (
    RouterReply,
    RouterToolCall,
    freeze_task,
    redacted_log,
    restore_tasks,
    router_model,
    run_competition_eval,
)
from yugos.inspect_engine.controls import lookup_task


def test_frozen_control_executes_tool_and_retains_only_provider_continuation(tmp_path):
    secret = "private-lookup-value"
    [question] = freeze_task(
        lookup_task("record-1", secret),
        builder="yugos.lookup",
        builder_args={"record_id": "record-1", "value": secret},
        source="control",
        workload_class="chat_qa",
    )
    [(task, _)] = restore_tasks([Question.model_validate_json(question.model_dump_json())])
    state = ProviderState(provider_id="fixture", model_id="model", data={"signed": "opaque"})
    calls = []

    async def client(call):
        calls.append(call)
        if len(calls) == 1:
            assert secret not in call.request.model_dump_json()
            return RouterReply(
                finish_reason="tool_calls",
                tool_calls=(
                    RouterToolCall(
                        id="t1", function="lookup_record", arguments={"record_id": "record-1"}
                    ),
                ),
                provider_state=state,
            )
        assistant, result = call.request.messages[-2:]
        assert provider_state(assistant) == state
        assert result.tool_call_id == "t1"
        assert json.loads(result.text) == {"record_id": "record-1", "value": secret}
        return RouterReply(content=secret, finish_reason="stop", provider_state=state)

    log = run_competition_eval(
        task, router_model(client, submission_id="control"), log_dir=str(tmp_path), display="none"
    )
    assert len(calls) == 2 and log.samples[0].scores["exact"].value == "C"
    saved = read_eval_log(log.location, resolve_attachments=True)
    assistant = saved.samples[0].messages[1]
    assistant.metadata["operator-secret"] = "hidden"
    assert provider_state(public_message(assistant)) == state
    assert "operator-secret" not in (public_message(assistant).metadata or {})
    public = redacted_log(saved, data_visibility="private").model_dump_json()
    assert PROVIDER_STATE_KEY not in public and "opaque" not in public
    assert "opaque" in saved.samples[0].output.model_dump_json()
