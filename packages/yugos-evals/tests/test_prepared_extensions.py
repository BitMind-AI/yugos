"""Installed private constructors restore prepared recipes without source acquisition."""

from types import SimpleNamespace

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from yugos_evals import prepared


def test_installed_constructor_receives_the_frozen_samples_and_plain_arguments(monkeypatch):
    sample = Sample(id="frozen", input="question", target="reference")
    options = {"dataset": [sample], "name": "private"}
    received = []

    def construct(**values):
        received.append(values)
        return Task(**values["options"], solver=generate(), scorer=exact())

    def registered(**query):
        assert query == {"group": "yugos_evals.prepared_tasks", "name": "private/example"}
        return [SimpleNamespace(load=lambda: construct)]

    monkeypatch.setattr(prepared, "entry_points", registered)
    task = prepared.build(options=options, task="private/example", args={"mode": "fixed"})
    assert task.dataset[0] == sample
    assert received == [{"options": options, "args": {"mode": "fixed"}, "fewshot_samples": ()}]


@pytest.mark.parametrize("registrations", [[], [object(), object()]])
def test_missing_or_ambiguous_constructors_cannot_choose_an_import(monkeypatch, registrations):
    monkeypatch.setattr(prepared, "entry_points", lambda **kwargs: registrations)
    with pytest.raises(ValueError, match="one installed constructor"):
        prepared.build(options={}, task="untrusted/arbitrary.module", args={})
