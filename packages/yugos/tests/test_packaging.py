"""Package metadata stays consistent with the Inspect versions the engine is verified against."""

import re
import tomllib
from pathlib import Path

from yugos.inspect_engine.versions import INSPECT_SANDBOXES_VERSION, PINNED

ROOT = Path(__file__).resolve().parents[3]


def requirements(path: Path) -> list[str]:
    project = tomllib.loads(path.read_text())["project"]
    found = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        found.extend(extra)
    return found


def pins(path: Path) -> dict[str, str]:
    """Exact `name==version` requirements declared anywhere in a pyproject."""
    found = {}
    for requirement in requirements(path):
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^;\s]+)", requirement)
        if match:
            found[match[1].lower()] = match[2]
    return found


def test_every_declared_inspect_pin_matches_the_engine():
    # A mismatch would install one Inspect release while frozen recipes and the
    # engine's version check expect another, failing only at evaluation time.
    expected = {**PINNED, "inspect-sandboxes": INSPECT_SANDBOXES_VERSION}
    for path in (ROOT / "pyproject.toml", ROOT / "packages/yugos-evals/pyproject.toml"):
        declared = pins(path)
        for package, version in expected.items():
            if package in declared:
                assert declared[package] == version, f"{path.name}: {package}"


def test_core_and_benchmark_packages_ship_separate_modules():
    # Two distributions that both install yugos_evals would overwrite each other.
    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert core["packages"]["find"]["where"] == ["packages/yugos/src"]
    assert not (ROOT / "packages/yugos/src/yugos_evals").exists()
