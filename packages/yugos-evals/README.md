# yugos-evals

Local benchmarks for Yugos, on [Inspect](https://inspect.aisi.org.uk/) and
[Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals). Use it to
list the catalog and run a router on your own machine. Local runs spend your
API keys and are not competition results.

The live competition pack is prepared separately. This package builds public
development packs only.

Math-Verify comparisons run in disposable processes with a 30-second wall
deadline and, on Linux, a 1 GiB address-space limit. A timeout, resource failure
or lost worker leaves the answer unscored as a grader error; it never becomes
an incorrect answer. Reference screening uses the same limits. Successful
comparisons retain the existing extraction and strict verification rules.

## Install

```sh
uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git" \
  "yugos-evals @ git+https://github.com/BitMind-AI/yugos.git#subdirectory=packages/yugos-evals"
```

That installs the benchmark catalog and the Inspect Evals tasks it runs. In a
checkout, `uv sync` at the repository root installs the same tools.

## Catalog

```sh
yugos-evals list
yugos-evals list --category agentic --json
```

`supported` means the task and its settings exist in the pinned packages.
`planned` entries have no task yet. Filter with `--status supported`.

Categories are `chat_qa`, `math_code`, and `agentic`. Each entry names an
Inspect task (`inspect_evals/...` or `yugos_evals/...`), the settings Yugos
selects, and the score key when a task reports several scores.

Tasks Yugos maintains because Inspect Evals does not:

| Task | Benchmarks |
| --- | --- |
| `yugos_evals/math_answer` | ArXivMath, LiveMathBench |
| `yugos_evals/code_tests` | LiveCodeBench v6 additions, LiveCodeBench-Plus |
| `yugos_evals/pairwise_judge` | Arena-Hard v2.0 |
| `yugos_evals/reading` | Belebele, LongBench v2, HotpotQA distractor validation |
| `yugos_evals/cruxeval` | CRUXEval output prediction |

AppWorld and Terminal-Bench are `planned`; Terminal-Bench 4.0 needs a separate
verifier environment and longer hosted invocations.

Reading sources are pinned to dataset commits. LongBench retains complete
contexts and uses the published answer extraction from
[LongBench](https://github.com/THUDM/LongBench). HotpotQA retains answer,
supporting-fact and joint metrics from the
[official evaluation contract](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py);
the catalog scores joint F1. Belebele metadata records a master question identity
shared across translations, allowing a private selector to exclude translated
duplicates. The package does not impose competition sampling quotas.

CRUXEval uses a single output expression, without the upstream few-shot or CoT
prompt. It preserves Python value equality and the
[official exclusion of the given function call](https://github.com/facebookresearch/cruxeval/blob/main/evaluation/utils_general.py).
Only expressions are accepted; statements appended to an expression are invalid.
The shared isolated Python supervisor keeps gold assertions outside the
candidate process. Its image needs `pyarrow` and `rpyc`; candidate code is never
evaluated on the preparation host.

## Run a router locally

```sh
yugos pack ./my-router --output router.zip
yugos-evals pack build --benchmark aime2025 --limit 20 --out aime.pack.tar
export OPENAI_API_KEY=...
yugos-evals run --pack aime.pack.tar --bundle router.zip --log-dir runs/a
yugos-evals compare runs/a runs/b
inspect view --log-dir runs/a/logs
```

`yugos pack` builds the miner ZIP. `yugos-evals pack build` builds a
development `pack.tar` from public data. `--limit` and `--seed` choose a
deterministic subset. `--arg NAME=JSON` passes extra task arguments.

`yugos-evals pack assemble packs/ --out combined.pack.tar` merges the `.tar`
packs directly in that directory, including `.pack.tar`, sorted by filename.
Other files are ignored. The directory must contain at least two packs. Pass
files instead to list them: `yugos-evals pack assemble a.tar b.tar --out
combined.pack.tar`. Do not mix files and a directory. Packs must share purpose.
An omitted `data_visibility` adopts the visibility other packs set; two
different visibilities still fail. Records are sorted by question id.
`--calibration PATH` stores that file as `difficulty-calibration.json`.

**`run`** checks the pack, the price card, declared models, role bindings, and
API keys before it spends. `--max-cost` is a USD cap on router and grader
spend together. It defaults to 100. The run stops when the next call would
exceed it, and still writes what it finished.

Results in `--log-dir`:

- `measurements.json` — grades, spend, and time
- `receipts.jsonl` — one receipt per router call
- `service-receipts.jsonl` — grader and simulator calls, excluded from router spend
- `calls.jsonl`, `worker.log`, and `logs/` (Inspect)

Errors in an individual question leave that question incomplete while later
questions and tasks continue. This includes provider, tool, simulator and grader
errors, and unfamiliar exceptions. Potentially billed failures are not retried;
unknown bills keep their reservations held against the remaining budget. Shared
worker failures, failed persistence and violated accounting bounds stop the run.
An incomplete question remains visible in the logs and final summary, even if
its task catches the error and returns a score.

`--world-card PATH|URL` selects prices. The default is the service's active
card, cached under `~/.cache/yugos-evals/world-cards/`. A pack that pins a card digest accepts only that card.

`--grader-model ROLE=PROVIDER/MODEL[@EFFORT]` fills a role the pack did not
pin. Pinned roles use the pack's model and settings. A pin that needs any
generation field other than `temperature`, `seed`, `max_tokens`, or
`reasoning_effort` is refused.

`--limit N` runs the first N questions in the pack.

Sandboxed tasks use Inspect's local Docker provider and the pack's pinned
images. Docker has to be installed.

**`compare RUN_A RUN_B`** requires both runs to cover the same questions and
to be complete. It prints category net value and the overall lower bound, and
reports when B beats A under the default policy (`--policy FILE` overrides
the numbers). Quality floors and hosted time limits are not applied locally.

### API keys

Each provider reads its key from the environment: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `VENICE_API_KEY`,
`DEEPSEEK_API_KEY`, `XAI_API_KEY` (provider id `spacexai`), `NOUS_API_KEY`,
`CHUTES_API_KEY`, `GM_API_KEY`, `ENGY_API_KEY`, `NEAR_AI_API_KEY`, `ZEROG_API_KEY`. Install the SDK Inspect expects
(`openai`, `anthropic`, or `google-genai`). The worker subprocess does not
receive variables that look like credentials.

There are no provider retries. OpenAI tool conversations use the Responses API.
Other providers in the price card are called as OpenAI-compatible endpoints.

### Role pins

Several tasks pin the grader or the user simulator. Local and hosted packs
record the same pins. A run fails before any router call if a pinned role is
bound to a different model or effort.

| Benchmarks | Role | Model | Settings |
| --- | --- | --- | --- |
| SimpleQA, SimpleQA Verified, Arena-Hard v2.0, HLE | `grader` | `openai/gpt-5.4-mini` | low effort, 16,384 output tokens |
| Tau2 airline, banking, retail, telecom | `user` | `openai/gpt-5.4-mini` | low effort, 4,096 output tokens |

The price card must list that model at the pinned effort. Reasoning models
ignore temperature.

Two local runs of the same pack and price card compare with each other. They
do not compare with hosted competition runs: the live pack is a different
question set, hosted spend uses provider bills, and hosted timing is measured
on the service. Run only bundles you trust. Local execution is not a sandbox
except for tasks that opt into Docker.

### Code tasks

`code_tests` runs candidate programs in a digest-pinned image. The catalog
default is the official `python:3.12-slim` index; pass
`--arg image='"name@sha256:..."'` to replace it. The image needs `python3`
3.11+ and the standard library. A `local` sandbox is refused.

Contributor test instructions are in the [repository development guide](../../README.md#development).

### Grader versions

New MATH, HumanEval, MBPP and SimpleQA tool tasks record their benchmark-owned
`grader_version` in the prepared recipe and task version. Each benchmark can
advance independently. MATH uses the shared `math_answer_scorer` and bounded
Math-Verify worker. General answer-line rules recognize explicit requests for
comma-separated roots, requested units, and count nouns; other lists retain
their ordering. No item-specific grading hints are used. Reviewed HumanEval
and MBPP data edits verify hashes of their original fields and stop preparation
if the pinned source data changes.

The `simpleqa_numeric_scorer` uses a Yugos-specific tool rubric with required
answer extraction and rationale. Its results are not directly comparable with
the upstream tool scorer. Numeric checks use the reference's last significant
digit; unsupported formats retain the judge verdict and record
`numeric_check: judge_fallback`. The explicit `original` scorer retains the
paper contract.

Frozen recipes without a grader version use the original native graders.
Restoration preserves their samples and task version. To adopt updated prompts,
tests or grading, prepare a new pack and recalibrate it; retain prior results.
