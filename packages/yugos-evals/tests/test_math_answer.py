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

from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task as freeze_task
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
    empty = _score("")
    assert empty.value == 0.0 and empty.reason == "no_response"
    # An invalid reference is never a silent zero.
    invalid = _score("\\boxed{42}", target="")
    assert math.isnan(invalid.value) and invalid.metadata["code"] == "invalid_gold"
    review = _score("\\boxed{42}", stop_reason="content_filter")
    assert math.isnan(review.value) and review.metadata["code"] == "needs_review"


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
