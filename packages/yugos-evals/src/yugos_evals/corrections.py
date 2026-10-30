"""Versioned fixes to pinned native benchmark samples and graders."""

from inspect_ai import Task

REVISION = 1
TASKS = {"math", "humaneval", "mbpp", "simpleqa", "simpleqa_verified"}


def corrected_scorer(name: str, args: dict):
    """Return the revised scorer, or None when the native scorer is retained."""
    if name == "math":
        from .scorers.math_equivalence import math_equivalence

        return math_equivalence()
    if name == "mbpp":
        from .scorers.benchmark_fixes import mbpp_with_imports

        return mbpp_with_imports()
    if name in {"simpleqa", "simpleqa_verified"} and args.get("scorer", "tool") == "tool":
        from inspect_evals.simpleqa.grader_templates import (
            SIMPLEQA_GRADER_TEMPLATE,
            SIMPLEQA_VERIFIED_GRADER_TEMPLATE,
        )

        from .scorers.benchmark_fixes import simpleqa_checked

        return simpleqa_checked(
            SIMPLEQA_VERIFIED_GRADER_TEMPLATE
            if name.endswith("verified")
            else SIMPLEQA_GRADER_TEMPLATE
        )
    return None


def correct_task(task: Task, name: str, args: dict):
    """Apply reviewed source corrections once, before samples are frozen."""
    if name not in TASKS or (task.metadata or {}).get("yugos_grading_revision") == REVISION:
        return
    replacement = corrected_scorer(name, args)
    if replacement is not None:
        task.scorer = [replacement]
    for sample in task.dataset:
        metadata = sample.metadata or {}
        if name == "humaneval" and str(sample.id) == "HumanEval/125":
            metadata["test"] += (
                '\n    for whitespace in "\\t\\n\\r\\v\\f":\n'
                '        assert candidate("abc" + whitespace + "def") == ["abc", "def"]\n'
            )
            sample.target = sample.target.replace(
                'if " " in txt:', "if any(c.isspace() for c in txt):"
            )
        elif name == "mbpp" and str(sample.id) == "143":
            tests = list(sample.target)
            tests[2] = "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1],)) == 1"
            sample.target = tests
            metadata.update(
                test_list=tests,
                test_list_str="\n".join(tests),
                code="def find_lists(items):\n    return sum(isinstance(item, list) for item in items)\n",
            )
        elif name == "mbpp" and str(sample.id) == "248":
            sample.input = "Write a function harmonic_sum(n) returning 1 + 1/2 + ... + 1/n for a positive integer n."
            metadata["prompt"] = sample.input
        sample.metadata = metadata or sample.metadata
    task.metadata = {**(task.metadata or {}), "yugos_grading_revision": REVISION}
    task.version = f"{task.version}+yugos-grading.{REVISION}"
