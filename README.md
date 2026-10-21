# Yugos

Yugos is a Bittensor subnet whose purpose is to produce the best router and
orchestration model. A router decides which models to call, what to ask them,
and when to stop. The subnet rewards the router that gets the best answers at
the lowest cost in the least time.

| | |
| --- | --- |
| Mainnet | subnet **34** on `finney` |
| Competition | one router, one champion, weights 85% / 10% / 5% |

- **[Mine](docs/miner.md)** — install, register, build a router, and submit.
- **[Validate](docs/validator.md)** — install, wallet, and the validator command.
- **[Models](docs/models.md)** — the world card of models a router may call.
- **[Scoring](docs/scoring.md)** — how a challenger replaces the champion.
- **[Example router](examples/starter-router)** — a three-file submission to copy.
- **[Agents](llms.txt)** — one page an agent can follow to start mining.

## How a round works

1. A miner packages router code and weights and submits them with a registered hotkey.
2. The evaluation service runs the router on the current benchmark. The router chooses which models to call. The service makes those calls, grades the final answer, and records the real cost and time.
3. A challenger replaces the champion only when it clears the category gates and its net value is high enough. See [Scoring](docs/scoring.md).
4. Validators read the published result and set on-chain weights: 85% to the current champion, 10% to the previous one, and 5% to the one before that.

The chain commitment is an audit record of the uploaded archive. Evaluation starts when the service accepts the upload.

## Install

Python 3.11 or newer. One environment covers the miner and validator commands.
Yugos does not create wallets. Use a wallet you already have under
`~/.bittensor/wallets`.

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git"
```

From a checkout, `uv sync` installs it (and `yugos-evals`) for development. Local benchmark
runs also need the separate benchmark package, `yugos-evals`:

```sh
uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git" \
  "yugos-evals @ git+https://github.com/BitMind-AI/yugos.git#subdirectory=packages/yugos-evals"
```

`yugos[inspect]` adds Inspect Evals for hosts that grade benchmarks. Miners and
validators that only submit or set weights do not need it.

## Repository

| Package | What it is |
| --- | --- |
| [`yugos`](packages/yugos) | Competition rules, the miner and validator commands, and the submission interface. |
| [`yugos-evals`](packages/yugos-evals) | Benchmark catalog and a local runner for trying a router on your own machine. |

The evaluation service prepares the live benchmark and runs submissions. This
repository does not download competition datasets at grade time.

## Development

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/).

```sh
uv sync
uv run --package yugos --extra inspect pytest packages/yugos/tests -q
uv run --package yugos-evals pytest packages/yugos-evals/tests -q
uv run ruff check .
uv run ruff format --check .
```

Core tests can run without the Inspect extra. Engine tests skip in that case:

```sh
uv run --isolated --package yugos --no-default-groups \
  --extra dev pytest packages/yugos/tests -q
```

Docker-backed code-task checks require Docker and explicit opt-in:

```sh
YUGOS_EVALS_DOCKER_TESTS=1 uv run --package yugos-evals pytest packages/yugos-evals/tests/test_code_tests.py -q
```

Local world-card caches default to `~/.cache/yugos-evals/world-cards/`; set `YUGOS_EVALS_CACHE` to use a different cache root during development.
