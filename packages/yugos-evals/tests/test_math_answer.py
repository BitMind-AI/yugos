"""`math_answer`: Math-Verify parity with the legacy grader, loaders, scoring and freezing."""

import hashlib
import json
import math
import os
from pathlib import Path

import anyio
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evals_fakes import ScriptedRouter, assert_portable_task_config, run, task_state
from inspect_ai.scorer import Target

from yugos.evaluation.grades import GradeCode
from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task as freeze_task
from yugos_evals.scorers._math_compare import grade_math as symbolic_grade
from yugos_evals.scorers.math import grade_math, math_answer_scorer, reference_exclusion
from yugos_evals.sources import _fetch, arxivmath, livemathbench
from yugos_evals.sources._math import boxed_prompt
from yugos_evals.tasks import math_answer

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE = [
    json.loads(line) for line in (FIXTURES / "math_baseline.jsonl").read_text().splitlines()
]
# Responses beyond the baseline fixture; their legacy grades are recorded in LEGACY.
RESPONSES = [
    "The answer is \\boxed{42}.",
    "Final answer: 41 or 42",
    "\\boxed{\\frac{1}{2}}",
    "0.5",
    "\\boxed{x^2 + 2x + 1}",
    "\\boxed{(x+1)^2}",
    "\\boxed{\\ln 2}",
    "\\boxed{[0, 1)}",
    "\\boxed{\\{1, 2\\}}",
    "\\boxed{}",
    "I don't know.",
    "x" * 70_000,
]
REFERENCES = [
    "42",
    "\\frac{1}{2}",
    "(x+1)^2",
    "\\ln 2",
    "\\log 2",
    "\\log_{2} 8",
    "\\log _{2} 8",
    "\\text{yes}",
    "\\text{ }5",
    "\\operatorname{lcm}(a, b)",
    "1, 2",
    "(1, 2)",
    "\\{1, 2\\}",
    "[0, 1)",
    "5 \\text{ cm}",
    "3\\sqrt{2}",
    "\\infty",
    "",
    "\\frac{",
    "a=3",
]


def _ported(outcome) -> dict:
    return {
        "score": outcome.score,
        "code": outcome.code.value,
        "status": outcome.status,
        "details": outcome.details,
    }


def _response_key(response: str) -> str:
    return hashlib.sha256(response.encode()).hexdigest()[:16]


# The legacy grader's outputs for REFERENCES x RESPONSES, recorded from `yugos.grading.math`
# (yugos cb20e3c) before it was deleted.
LEGACY = json.loads((FIXTURES / "math_legacy_parity.json").read_text())


@pytest.mark.parametrize("case", BASELINE, ids=lambda case: case["name"])
def test_grading_matches_the_recorded_legacy_baseline(case):
    assert _ported(grade_math(case["gold"], case["response"])) == case["expected"]


def test_grading_matches_the_recorded_legacy_grader():
    responses = {_response_key(r): r for r in RESPONSES}
    assert {(c["reference"], c["response_sha256_16"]) for c in LEGACY["grades"]} == {
        (r, _response_key(s)) for r in REFERENCES if r for s in RESPONSES
    }
    for case in LEGACY["grades"]:
        response = responses[case["response_sha256_16"]]
        assert _ported(grade_math(case["reference"], response)) == case["expected"], case


def test_reference_exclusion_matches_the_recorded_legacy_screen():
    assert [c["reference"] for c in LEGACY["exclusions"]] == REFERENCES
    for case in LEGACY["exclusions"]:
        assert reference_exclusion(case["reference"]) == case["expected"], case["reference"]


def _score(completion, target="42", stop_reason="stop"):
    score = math_answer_scorer()
    return anyio.run(score, task_state(completion, stop_reason=stop_reason), Target(target))


def test_scorer_settles_answers_and_leaves_grader_failures_unscored():
    assert _score("\\boxed{42}").value == 1.0
    wrong = _score("\\boxed{41}")
    assert wrong.value == 0.0 and wrong.metadata["code"] == "incorrect"
    truncated = _score("\\boxed{42}", stop_reason="max_tokens")
    assert truncated.value == 0.0 and truncated.metadata["code"] == "truncated"
    # An answer that stopped with no content is an attempt (it may have only reasoned).
    for blank in ("", " \n\t"):
        empty = _score(blank)
        assert empty.value == 0.0 and empty.metadata["code"] == "incorrect"
        assert empty.metadata["cause"] == "empty_answer"
    # An invalid reference is never a silent zero.
    invalid = _score("\\boxed{42}", target="")
    assert math.isnan(invalid.value) and invalid.metadata["code"] == "invalid_gold"
    # A provider block is the answer as given: graded incorrect, even if it
    # happens to contain the reference.
    blocked = _score("\\boxed{42}", stop_reason="content_filter")
    assert blocked.value == 0.0 and blocked.metadata["code"] == "incorrect"
    assert blocked.metadata["cause"] == "provider_blocked"
    review = _score("\\boxed{42}", stop_reason="tool_calls")
    assert math.isnan(review.value) and review.metadata["code"] == "needs_review"


def test_a_sample_ended_before_any_answer_is_not_attempted():
    from inspect_ai.model import ModelOutput

    # A limit (time, spend) can end the sample at its first turn: Inspect
    # leaves no choice, whose stop reason must not be read.
    state = task_state("")
    state.output = ModelOutput(model="mockllm/model", choices=[])
    score = anyio.run(math_answer_scorer(), state, Target("42"))
    assert score.value == 0.0 and score.metadata["code"] == "not_attempted"
    assert score.reason == "no_response"


def _month_fixture(tmp_path, monkeypatch, rows):
    month = arxivmath.MONTHS[0]
    path = tmp_path / month.dataset / arxivmath.DATA_FILE
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    pinned = arxivmath.ArxivMathMonth(
        month.dataset, month.revision, _fetch.sha256_file(path), path.stat().st_size
    )
    monkeypatch.setattr(arxivmath, "MONTHS", (pinned,))
    return pinned


ROWS = [
    {"problem_idx": 1, "problem": "Compute 6*7.", "answer": "42", "source": 2512.01234},
    {"problem_idx": 2, "problem": "Name it.", "answer": "\\text{yes}", "source": 2512.0123},
    {"problem_idx": 3, "problem": "Evaluate.", "answer": "\\log 8", "source": 2512.01236},
]


def test_arxivmath_loader_screens_references_and_records_provenance(tmp_path, monkeypatch):
    month = _month_fixture(tmp_path, monkeypatch, ROWS)
    loaded = arxivmath.load_arxivmath(data_dir=tmp_path)
    [sample] = loaded.samples
    assert sample.id == "arxivmath-2512-1" and sample.target == "42"
    assert sample.input == boxed_prompt("Compute 6*7.")
    assert sample.metadata["arxiv_id"] == "2512.01234"
    assert [e["id"] for e in loaded.excluded] == ["arxivmath-2512-2", "arxivmath-2512-3"]
    assert loaded.excluded[0]["reason"].startswith("textual_reference")
    assert loaded.excluded[1]["reason"].startswith("ambiguous_log_base")
    assert loaded.files == [
        {
            "source": month.dataset,
            "revision": month.revision,
            "file": arxivmath.DATA_FILE,
            "sha256": month.sha256,
        }
    ]
    with pytest.raises(ValueError, match="unknown ArXivMath months"):
        arxivmath.load_arxivmath(["9901"], data_dir=tmp_path)


def test_arxivmath_loader_refuses_changed_files_and_unreviewed_columns(tmp_path, monkeypatch):
    _month_fixture(tmp_path, monkeypatch, [{**ROWS[0], "hint": "extra"}])
    with pytest.raises(ValueError, match="invalid ArXivMath row"):
        arxivmath.load_arxivmath(data_dir=tmp_path)
    month = arxivmath.MONTHS[0]
    monkeypatch.setattr(
        arxivmath, "MONTHS", (month.__class__(month.dataset, month.revision, "0" * 64, 1),)
    )
    with pytest.raises(ValueError, match="not the pinned"):
        arxivmath.load_arxivmath(data_dir=tmp_path)


def test_livemathbench_requires_a_token_without_printing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(PermissionError, match="gated.*HF_TOKEN"):
        livemathbench.load_livemathbench()
    subset = tmp_path / "202505" / "all_en.jsonl"
    subset.parent.mkdir()
    subset.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"question": "What is 2+2?", "answer": "4", "question_type": "Problem-Solving"},
                {"question": "Which is larger?", "answer": "\\text{the first}"},
            )
        )
    )
    loaded = livemathbench.load_livemathbench(data_dir=tmp_path)
    assert [s.id for s in loaded.samples] == ["livemathbench-202505-all-en-1"]
    assert [e["id"] for e in loaded.excluded] == ["livemathbench-202505-all-en-2"]


def test_livemathbench_access_errors_do_not_leak_the_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_value")

    def refused(*args, **kwargs):
        raise _fetch.SourceAccessError("all_en.jsonl: access refused (401)")

    monkeypatch.setattr(livemathbench, "hf_file", refused)
    with pytest.raises(PermissionError) as error:
        livemathbench.load_livemathbench()
    assert "hf_secret_token_value" not in str(error.value)


def test_task_freezes_and_restores_without_its_loader(tmp_path, monkeypatch):
    _month_fixture(tmp_path, monkeypatch, ROWS)
    task = math_answer(source="arxivmath", data_dir=str(tmp_path))
    assert [e["id"] for e in task.metadata["excluded"]] == ["arxivmath-2512-2", "arxivmath-2512-3"]
    questions = freeze_task(task, source="arxivmath", workload_class="math_code")
    [question] = [q.model_validate_json(q.model_dump_json()) for q in questions]
    assert_portable_task_config(question)
    monkeypatch.setattr(arxivmath, "MONTHS", ())
    [(restored, definition)] = restore_tasks([question])
    assert definition.scorer == "math_answer_scorer"
    router = ScriptedRouter("6*7 = 42.\n\\boxed{42}")
    log = run(restored, router, tmp_path)
    [sample] = log.samples
    assert sample.scores["math_answer_scorer"].value == 1.0
    # The reference never reaches the router.
    assert '"42"' not in router.seen() and "arxivmath" not in router.seen()


def test_math_answer_rejects_mismatched_selections():
    with pytest.raises(ValueError, match="months"):
        math_answer(source="livemathbench", months=["2512"])
    with pytest.raises(ValueError, match="subsets"):
        math_answer(source="arxivmath", subsets=["202505/all_en"])


@pytest.mark.skipif(
    not os.environ.get("YUGOS_EVALS_NETWORK_TESTS"), reason="downloads pinned sources (~180 KB)"
)
def test_every_pinned_arxivmath_month_loads_and_matches_the_reference_screen():
    loaded = arxivmath.load_arxivmath()
    assert {row["sha256"] for row in loaded.files} == {m.sha256 for m in arxivmath.MONTHS}
    excluded = {e["id"]: e["reason"] for e in loaded.excluded}
    rows = [
        row
        for month in arxivmath.MONTHS
        for row in arxivmath._rows(
            month, _fetch.cache_dir() / "sources" / month.sha256 / Path(arxivmath.DATA_FILE).name
        )
    ]
    assert len(rows) == len(loaded.samples) + len(excluded)
    for sample_id, _, answer, _ in rows:
        assert excluded.get(sample_id) == reference_exclusion(answer.strip()), sample_id
    print(f"arxivmath: {len(rows)} rows, {len(loaded.samples)} kept, {len(excluded)} excluded")


@pytest.mark.parametrize(
    "reference,answer",
    [
        (r"2! \cdot 4! \cdots 2^{2023}!", r"\prod_{k=1}^{2024}(2^{k-1})!"),
        (r"\prod_{i=2}^{8}(3^i)!", r"\prod_{j=1}^{7}(3^{j+1})!"),
        (r"\prod_{i=1}^{5}(2^i)!", r"\prod_{i=1}^{5}(4^{i}/2^i)!"),
        (r"\prod_{i=1}^{4}i!", r"2!\cdot3!\cdot4!"),
    ],
)
def test_equivalent_products_normalize_without_factorial_expansion(reference, answer, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a factorial was expanded")

    monkeypatch.setattr("math.factorial", forbidden)
    monkeypatch.setattr("sympy.factorial", forbidden)
    result = symbolic_grade(reference, rf"\boxed{{{answer}}}")
    assert result.settled and result.score == 1


@pytest.mark.parametrize("answer", [r"\prod_{i=1}^{6}(2^i)!", r"(2^{9})!", "0", "14"])
def test_proven_mismatches_are_incorrect(answer):
    result = symbolic_grade(r"\prod_{i=1}^{8}(2^i)!", rf"\boxed{{{answer}}}")
    assert result.settled and result.score == 0


def test_explicit_indexed_definition_is_resolved():
    result = symbolic_grade(
        r"\prod_{i=1}^{6}(2^i)!",
        r"Let \(m_j = 2^{j+1}\). Thus \boxed{\prod_{j=0}^{5}m_j!}",
    )
    assert result.settled and result.score == 1


@pytest.mark.parametrize(
    "answer",
    [
        r"\prod_{i=1}^{8}(2^i)!+1",
        r"\prod_{i=1}^{8}(2^i)!\cdot x",
        r"\prod_{i=1}^{8}(2^i)!=0",
        r"\prod_{i=1}^{8}(3^{i-1})!",
        r"\prod_{i=1}^{1000000}(2^i)!",
        r"\prod_{i=1}^{8}(2^{1000000})!",
        r"\left(\prod_{i=0}^{2024}m_i!\right)"
        r"\left(\left\lfloor\frac{F}{Q^E}\right\rfloor\bmod Q\right)",
    ],
)
def test_unresolved_or_oversized_expressions_never_invent_a_zero(answer):
    result = symbolic_grade(r"\prod_{i=1}^{8}(2^i)!", rf"\boxed{{{answer}}}")
    assert not result.settled
    assert result.details["cause"] == "symbolic_comparison_unresolved"


def test_conflicting_definitions_are_unresolved():
    result = symbolic_grade(
        r"\prod_{i=1}^{6}(2^i)!",
        r"m_j = 2^j\n m_j = 3^j\n \boxed{\prod_{j=1}^{6}m_j!}",
    )
    assert not result.settled


@pytest.mark.parametrize("definition", [r"m_0=99, m_j=2^j", r"m_j=2^j\quad(j\ge1)"])
def test_index_definitions_do_not_drop_exceptions_or_domain_restrictions(definition):
    result = symbolic_grade(r"\prod_{i=0}^{5}(2^i)!", definition + r"\boxed{\prod_{j=0}^{5}m_j!}")
    assert not result.settled


def test_reference_screening_uses_the_same_complete_symbolic_parser(monkeypatch):
    from yugos_evals.scorers import _math_compare

    monkeypatch.setattr(_math_compare, "parse", lambda *a, **kw: pytest.fail("expanding parser"))
    assert _math_compare.reference_exclusion(r"\prod_{i=1}^{2023}(2^i)!") is None
    assert _math_compare.reference_exclusion(r"\prod_{i=1}^{n}(2^i)!") is not None


def test_zero_coefficients_and_unary_precedence_do_not_invent_inequalities():
    assert symbolic_grade(r"0*\prod_{i=1}^{4}i!", r"\boxed{0*99!}").score == 1
    result = symbolic_grade(r"\prod_{i=1}^{4}i!", r"\boxed{(-2^2)!}")
    assert not result.settled


def test_floor_modulo_and_explicit_scalar_definitions_keep_exact_arithmetic():
    result = symbolic_grade(
        r"4*\prod_{i=1}^{4}i!",
        r"Let \(F=123456\), \(Q=10\), \(E=2\). "
        r"\boxed{(\prod_{i=1}^{4}i!)(\lfloor\frac{F}{Q^E}\rfloor\bmod Q)}",
    )
    assert result.settled and result.score == 1


def test_symbolic_worker_timeout_remains_a_saved_unresolved_comparison(monkeypatch):
    from yugos_evals.scorers import math
    from yugos_evals.scorers._math_process import MathWorkerFailed

    def timeout(*args, **kwargs):
        raise MathWorkerFailed("timeout")

    monkeypatch.setattr(math, "compare", timeout)
    result = math.grade_math(r"\prod_{i=1}^{8}(2^i)!", r"\boxed{1}")
    assert not result.settled and result.status == "unavailable"
    assert result.details["cause"] == "symbolic_comparison_unresolved"


def test_every_settled_small_product_agrees_with_exact_integer_arithmetic():
    from math import factorial, prod

    for base in (2, 3, 4):
        reference = rf"\prod_{{i=1}}^{{3}}({base}^i)!"
        expected = prod(factorial(base**i) for i in range(1, 4))
        for other in (2, 3, 4):
            for lower, upper in ((0, 3), (1, 2), (1, 3), (1, 4)):
                answer = rf"\prod_{{j={lower}}}^{{{upper}}}({other}^j)!"
                outcome = symbolic_grade(reference, rf"\boxed{{{answer}}}")
                actual = prod(factorial(other**j) for j in range(lower, upper + 1))
                if outcome.settled:
                    assert outcome.score == int(expected == actual)


def test_small_factorial_products_still_accept_their_exact_integer_answers():
    assert symbolic_grade(r"\prod_{i=1}^{3}i!", r"\boxed{12}").score == 1
    assert symbolic_grade(r"\prod_{i=1}^{3}i!", r"\boxed{13}").code == GradeCode.INCORRECT


# Retained regressions from the former hosted factorial-product repair.
REFERENCE = r"2! \cdot 4! \cdots 2^{2023}!"


@pytest.mark.parametrize(
    "answer",
    [
        r"(2^0)!",
        r"(2^{2023})!",
        r"(2^{2024})!",
        r"2^{2024}!",
        r"(2^{99999})!",
        r"(3^{2024})!",
    ],
)
def test_single_power_factorials_are_incorrect_without_expansion(answer, monkeypatch):
    def forbid_expansion(*args, **kwargs):
        pytest.fail("symbolic comparison must not expand factorials")

    monkeypatch.setattr("math.factorial", forbid_expansion)
    monkeypatch.setattr("sympy.factorial", forbid_expansion)
    outcome = symbolic_grade(REFERENCE, rf"\boxed{{{answer}}}")
    assert outcome.code == GradeCode.INCORRECT
    assert outcome.score == 0
    assert outcome.settled


def test_last_box_and_malformed_box():
    correct = rf"\boxed{{{REFERENCE}}}"
    wrong = r"\boxed{\prod_{i=0}^{2022}(2^{i})!}"
    assert symbolic_grade(REFERENCE, correct + wrong).code == GradeCode.INCORRECT
    assert not symbolic_grade(REFERENCE, correct + r"\boxed{").settled
    assert symbolic_grade(REFERENCE, "").code == GradeCode.NOT_ATTEMPTED
    assert not symbolic_grade(REFERENCE, "unboxed prose").settled


def test_long_reasoning_keeps_final_answer_and_oversized_boxes_need_review():
    from yugos_evals.scorers._symbolic_products import MAX_TEXT

    answer = rf"\boxed{{{REFERENCE}}}"
    assert symbolic_grade(REFERENCE, "x" * (MAX_TEXT + 1) + answer).score == 1
    assert not symbolic_grade(REFERENCE, r"\boxed{" + "x" * MAX_TEXT + "}").settled


@pytest.mark.parametrize("answer", ["0", "1", "-1", "+2024", "9" * 5000])
def test_integer_answers_below_the_factorial_product_settle_without_expansion(
    answer, monkeypatch
):
    # The observed saved answer was boxed 1; parsing huge numeric references must
    # neither require a product-shaped answer nor expand factorials to compare it.
    monkeypatch.setattr("math.factorial", lambda *a: pytest.fail("factorial expansion"))
    monkeypatch.setattr(
        "sympy.factorial", lambda *a: pytest.fail("factorial expansion")
    )
    outcome = symbolic_grade(REFERENCE, rf"\boxed{{{answer}}}")
    assert outcome.settled and outcome.code == GradeCode.INCORRECT


@pytest.mark.parametrize("answer", ["1+x", "1=1", r"\frac{1}{x}", "1e999999999"])
def test_numeric_prefix_does_not_discard_unparsed_math(answer):
    assert not symbolic_grade(REFERENCE, rf"\boxed{{{answer}}}").settled


def test_symbolic_completion_gate_is_preserved_by_native_scorer():
    result = _score(r"\boxed{(2^{2024})!}", target=REFERENCE, stop_reason="max_tokens")
    assert result.metadata["code"] == GradeCode.TRUNCATED.value


def test_saved_symbolic_reference_selection_rejects_incomplete_products():
    from yugos_evals.scorers.math import supports_symbolic_product

    assert supports_symbolic_product(REFERENCE)
    assert supports_symbolic_product(r"\prod_{i=1}^{3}i!")
    assert not supports_symbolic_product("4")
    assert not supports_symbolic_product(r"\prod_{i=1}^{n}i!")
    assert not supports_symbolic_product(REFERENCE + "=0")


@pytest.mark.parametrize("answer,score", [("0", 0), (r"\prod_{j=1}^{4}j!", 1)])
def test_symbolic_answer_line_overrides_earlier_box(answer, score):
    result = symbolic_grade(
        r"\prod_{i=1}^{4}i!",
        r"\boxed{\prod_{i=1}^{4}i!}" + "\nANSWER: " + answer,
        answer_format="answer_line",
    )
    assert result.settled and result.score == score


def test_symbolic_answer_line_keeps_explicit_definition_context():
    result = symbolic_grade(
        r"\prod_{i=1}^{3}(2^i)!",
        "m_j=2^j\nANSWER: " + r"\prod_{j=1}^{3}m_j!",
        answer_format="answer_line",
    )
    assert result.settled and result.score == 1


@pytest.mark.parametrize("definitions", [r"m_j=2^j, m_{j}=3^j, ", r"m_{j}=3^j, m_j=2^j, "])
def test_conflicting_braced_indices_cannot_settle(definitions):
    result = symbolic_grade(r"\prod_{i=1}^{3}(2^i)!", definitions + r"\boxed{\prod_{j=1}^{3}m_j!}")
    assert not result.settled
