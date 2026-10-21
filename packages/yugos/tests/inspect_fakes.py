"""Scripted routers, receipts and packs for Inspect engine tests; no provider calls."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal

import anyio
from inspect_ai import Task, task_with
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from yugos.evaluation import Question
from yugos.inspect_engine import (
    MemoryReceipts,
    ProviderReceipt,
    RouterCall,
    RouterReply,
    restore_tasks,
    router_model,
    run_competition_eval,
)


def exact_question(question_id: str, reference: str, **fields) -> Question:
    [question] = freeze_fixture(
        Task(
            dataset=[
                Sample(
                    id=question_id,
                    input=f"Question {question_id}: name the reference.",
                    target=reference,
                )
            ],
            solver=generate(),
            scorer=exact(),
            name="fixture",
        ),
        source=fields.pop("source_benchmark", "fixture"),
        workload_class=fields.pop("workload_class", "chat_qa"),
    )
    return question.model_copy(update={"question_id": question_id, **fields})


def pack_task(questions, *, name):
    [(task, _)] = restore_tasks(questions)
    return task_with(task, name=name)


def reply(content: str, finish_reason: str = "stop") -> RouterReply:
    return RouterReply(
        content=content, finish_reason=finish_reason, input_tokens=7, output_tokens=3
    )


class ScriptedRouter:
    """Replay a scripted reply per prompt and meter one fake provider call per generate.

    Receipt IDs and clocks depend only on the sample and call order, so reruns
    produce identical receipts apart from Inspect's random sample UUIDs.
    """

    def __init__(
        self,
        replies: Mapping[str, RouterReply | str | BaseException],
        *,
        cost_usd: str = "0.001",
        delay_s: float = 0.0,
        receipts: MemoryReceipts | None = None,
    ) -> None:
        self.replies = replies
        self.cost_usd = Decimal(cost_usd)
        self.delay_s = delay_s
        self.receipts = receipts if receipts is not None else MemoryReceipts()
        self.calls: list[RouterCall] = []
        self._counts: defaultdict[tuple[str, int], itertools.count] = defaultdict(itertools.count)

    def scripted(self, call: RouterCall) -> RouterReply | str | BaseException:
        prompt = call.request.messages[-1].text
        if isinstance(prompt, list):
            prompt = "".join(part["text"] for part in prompt)
        return self.replies[prompt]

    async def __call__(self, call: RouterCall) -> RouterReply:
        self.calls.append(call)
        scripted = self.scripted(call)
        if isinstance(scripted, BaseException):
            raise scripted
        index = next(self._counts[(call.scope.sample_id, call.scope.epoch)])
        if self.delay_s:
            await anyio.sleep(self.delay_s)
        self.receipts.append(
            ProviderReceipt(
                receipt_id=f"{call.scope.sample_id}:{call.scope.epoch}:{index}",
                sample_uuid=call.scope.sample_uuid,
                provider="fake",
                model="fake/model",
                reasoning_effort="low",
                cost_usd=self.cost_usd,
                started_ns=1_000 * index,
                ended_ns=1_000 * index + 500,
            )
        )
        return reply(scripted) if isinstance(scripted, str) else scripted


class SequenceRouter(ScriptedRouter):
    """Replay scripted replies in call order, for agents whose prompts grow each turn."""

    def __init__(self, replies, **kwargs) -> None:
        super().__init__({}, **kwargs)
        self.pending = list(replies)

    def scripted(self, call: RouterCall) -> RouterReply | str | BaseException:
        return self.pending.pop(0)


def run(task, router: ScriptedRouter, tmp_path, *, submission_id: str = "sub-a", **options):
    """Run `task` against `router` under the competition fairness profile."""
    model = router_model(router, submission_id=submission_id)
    return run_competition_eval(
        task, model, log_dir=str(tmp_path / submission_id), display="none", **options
    )


# Test task code remains in Python; the pack selects a fixture constructor by name.
_NATIVE_FIXTURES = {}


def freeze_fixture(task, **options):
    from inspect_ai._util.registry import registry_info
    from inspect_ai.scorer._scorer import unique_scorer_name

    from yugos.inspect_engine.frozen import freeze_task as prepare

    try:
        name = registry_info(task).name
    except ValueError:
        name = ""
    if name.startswith(("inspect_evals/", "yugos_evals/")):
        from yugos_evals.prepared import prepare_task

        return prepare_task(task, **options)
    names = []
    for scorer in task.scorer or ():
        names.append(unique_scorer_name(scorer, names))
    case = (task.name if task.name == "fixture" else options["source"]) + ":" + "+".join(names)
    _NATIVE_FIXTURES[case] = task
    from unittest.mock import patch

    from yugos.inspect_engine import frozen

    with patch.object(frozen, "_task_builder", return_value=_fixture_registration()):
        return prepare(task, builder="tests.yugos", builder_args={"case": case}, **options)


def _fixture_registration():
    from importlib.metadata import version
    from types import SimpleNamespace

    from inspect_ai import task_with

    def build(*, options, case):
        from copy import deepcopy

        return task_with(deepcopy(_NATIVE_FIXTURES[case]), **options)

    return SimpleNamespace(
        dist=SimpleNamespace(name="pytest", version=version("pytest")), load=lambda: build
    )


def install_task_builders(monkeypatch):
    from yugos.inspect_engine import frozen

    original = frozen._task_builder
    monkeypatch.setattr(
        frozen,
        "_task_builder",
        lambda name: _fixture_registration() if name == "tests.yugos" else original(name),
    )
