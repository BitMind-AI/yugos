"""Native task builders for offline fixtures."""

import pytest


@pytest.fixture(autouse=True)
def native_fixture_builders(monkeypatch):
    from evals_fakes import install_task_builders

    install_task_builders(monkeypatch)


@pytest.fixture(autouse=True)
def scripted_providers_need_no_sdk(monkeypatch):
    """Tests script provider responses; they never need an installed provider SDK."""
    from yugos_evals.local import providers

    monkeypatch.setattr(providers, "_SDK_CHECKS", {})
