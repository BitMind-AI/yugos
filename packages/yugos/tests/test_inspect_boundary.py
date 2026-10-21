import ast
import subprocess
import sys
from pathlib import Path

import yugos


def test_core_evaluation_does_not_import_inspect():
    # Catches core Yugos gaining a dependency on the optional Inspect engine.
    code = (
        "import sys\n"
        "import yugos.competition, yugos.evaluation, yugos.competition.net_value, yugos.evaluation.packs\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in {'inspect_ai', 'inspect_evals'}]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_never_imports_the_evals_package():
    # Dependencies point one way: yugos-evals builds on core, never the reverse.
    source = Path(yugos.__file__).parent
    offenders = []
    for path in sorted(source.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value]  # Dynamic imports name the module in a string.
            else:
                continue
            if any(n == "yugos_evals" or n.startswith("yugos_evals.") for n in names):
                offenders.append(f"{path.relative_to(source)}:{node.lineno}")
    assert not offenders
