"""A single output prediction, graded as a Python expression in isolation."""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from ..scorers.cruxeval import cruxeval_output_scorer
from ..sources.cruxeval import load_cruxeval
from .code_tests import code_sandbox


@task
def cruxeval(image: str, sandbox_type: str = "docker", data_dir: str | None = None) -> Task:
    """Use a prepared Python image containing the shared Arrow/RPyC supervisor dependencies."""
    loaded = load_cruxeval(data_dir=data_dir)
    return Task(
        dataset=loaded.samples,
        solver=generate(),
        scorer=cruxeval_output_scorer(),
        sandbox=code_sandbox(image, sandbox_type),
        metadata=loaded.metadata(),
        version="1-A",
    )
