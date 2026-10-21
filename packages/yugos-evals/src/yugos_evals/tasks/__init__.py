"""Inspect tasks for catalog benchmarks that Inspect Evals lacks.

Tasks register as `yugos_evals/<name>` and use only portable, registered
components so that Ycore can freeze them into competition packs. Each factory
loads its public source when called, at preparation; frozen packs restore
the selected samples without calling it again.
"""

from .boxed_math import boxed_math
from .code_tests import code_tests
from .math_answer import math_answer
from .pairwise_judge import pairwise_judge
from .rubric_chat import rubric_chat

__all__ = ["boxed_math", "code_tests", "math_answer", "pairwise_judge", "rubric_chat"]
