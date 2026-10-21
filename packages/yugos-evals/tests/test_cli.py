"""The command prints the catalog and fails clearly on bad local-run arguments."""

import json

from yugos_evals.catalog import CATALOG
from yugos_evals.cli import main


def test_list_prints_every_entry(capsys):
    assert main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:2] == ["ID", "CATEGORY"]
    assert [line.split()[0] for line in lines[1:]] == [e.id for e in CATALOG]


def test_list_filters_and_emits_json(capsys):
    assert main(["list", "--json", "--category", "agentic", "--status", "planned"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and {(r["category"], r["status"]) for r in rows} == {("agentic", "planned")}


def test_run_defaults_the_spend_cap_to_100_dollars():
    from yugos_evals.cli import build_parser

    args = build_parser().parse_args(["run", "--pack", "p.tar", "--bundle", "b.zip"])
    assert args.max_cost == 100


def test_run_reports_setup_errors_without_a_traceback(tmp_path, capsys):
    argv = ["run", "--pack", str(tmp_path / "missing.tar"), "--bundle", "b.zip"]
    assert main([*argv, "--max-cost", "1", "--log-dir", str(tmp_path / "run")]) == 2
    assert "yugos-evals run:" in capsys.readouterr().err
    assert main([*argv, "--max-cost", "1", "--grader-model", "grader=gpt"]) == 2
    assert "provider/model" in capsys.readouterr().err
