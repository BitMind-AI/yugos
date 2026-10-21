"""Native task builders for offline fixtures."""

import pytest


@pytest.fixture(autouse=True)
def native_fixture_builders(monkeypatch):
    from inspect_fakes import install_task_builders

    install_task_builders(monkeypatch)
