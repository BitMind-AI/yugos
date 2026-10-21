"""Installed entry points let Inspect and frozen Yugos recipes find this package."""

from importlib.metadata import entry_points


def test_inspect_entry_point_loads_the_task_registry():
    [entry] = [e for e in entry_points(group="inspect_ai") if e.name == "yugos_evals"]
    entry.load()


def test_installed_core_satisfies_the_declared_range_and_reports_its_version():
    # A core bump without the matching range (or __version__) would install a
    # yugos-evals that pip cannot resolve, or report a stale version.
    from importlib.metadata import requires, version

    from packaging.requirements import Requirement

    import yugos
    import yugos_evals

    [core] = [
        r for r in map(Requirement, requires("yugos-evals")) if r.name == "yugos" and not r.marker
    ]
    assert core.specifier.contains(version("yugos"))
    assert yugos.__version__ == version("yugos")
    assert yugos_evals.__version__ == version("yugos-evals")
