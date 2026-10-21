# Mine on Yugos

You submit a **router**: code and weights that choose which models to call, how
to prompt them, and when to stop. The subnet runs those calls, grades the final
answer, and scores the result on quality, cost, and time. Training and routing
policy belong to your submission. Copy
[`examples/starter-router`](../examples/starter-router). It calls the cheap
OpenRouter model `openai/gpt-5-nano` from the [world card](models.md). A local
run needs `OPENROUTER_API_KEY`.

Mainnet is subnet **34** (`finney`).

## Setup

Python 3.11 or newer.

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git"
```

Local runs also need the benchmark package, `yugos-evals`:

```sh
uv pip install "yugos-evals @ git+https://github.com/BitMind-AI/yugos.git#subdirectory=packages/yugos-evals"
```

## Register

Create a wallet, then register its hotkey. Registration spends a small amount
of TAO from the coldkey. Testnet TAO does not pay a mainnet fee. Stay on
testnet until one submit succeeds.

```sh
btcli wallet create --wallet miner
btcli subnet burn-cost --network test 169
btcli subnet register --network test --wallet miner --hotkey default --netuid 169
```

Mainnet is a separate registration. Fund the coldkey with mainnet TAO first.

```sh
btcli subnet burn-cost 34
btcli subnet register --wallet miner --hotkey default --netuid 34
```

The hotkey name `default` is the one you pass to `yugos submit`. See the
[Bittensor mining guide](https://guides.learnbittensor.org/miners/miners-btcli-guide)
for wallet recovery and staking.

## Providers and models

Your router yields a `CallAction` naming a **provider, exact model ID, and reasoning effort**. The hosted service makes the provider request and accounts for its cost. You do **not** need your own provider accounts or API keys for hosted evaluation, and must not put credentials in your bundle. Local runs use your own keys.

These provider integrations are available in Yugos. The catalog and the competition's allowed choices determine which models you can actually submit:

| Provider | `provider_id` in `models.json` | API key for local runs only |
| --- | --- | --- |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google` | `GOOGLE_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Venice | `venice` | `VENICE_API_KEY` |
| SayGM | `gm` | `GM_API_KEY` |
| Engy | `engy` | `ENGY_API_KEY` |
| xAI | `spacexai` | `XAI_API_KEY` |
| Nous Research | `nous` | `NOUS_API_KEY` |
| Chutes | `chutes` | `CHUTES_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |

### Find the current catalog

The service publishes a **world card**: a JSON document containing models, efforts, reviewed token limits, and prices. Download the current catalog:

```sh
curl --fail --show-error \
  https://upload.yugos.ai/world/cards/active \
  --output world-card.json
cat world-card.json | jq
```

No wallet or payment is needed to read a card.

The response has `sha256` and `card` fields. Under `card.providers`, look for:

| Field | What to use it for |
| --- | --- |
| `provider_id` | Exact provider identifier for a call. |
| `models[].model_id` | Exact model ID; keep any namespace or date suffix. |
| `models[].effort_levels` | Allowed effort choices. JSON `null` is different from the string `"none"`. |
| `token_prices[model_id]` | USD per million tokens, including any cache rates and context bands. |
| `model_limits[model_id]` | Reviewed input/output token ceilings and review expiry. |

Only select models with prices and reviewed limits. More output, reasoning, and repeated calls can increase cost and elapsed time. The active card can change; a hosted run keeps the card selected for that run. Save the card if you want to reproduce a local comparison.

Your declared choices must be supported by the competition. Unsupported declarations fail preparation; there is no automatic provider or model substitution.

## What to build

Put the submission in its own directory:

```text
my-router/
  model.py
  model.safetensors
  models.json
  router.json          # optional settings and tokenizers
  encoder/             # optional nested assets
```

`models.json` is a JSON array of every provider, model, and effort the router
may call. Different efforts are different choices. Duplicates are rejected.

```json
[
  {"provider_id": "provider-a", "model_id": "model-a", "effort": "low"},
  {"provider_id": "provider-b", "model_id": "model-b", "effort": null}
]
```

The service only runs choices it supports. A call outside this list scores that
question zero. There is no fallback model.

`model.py` defines `load_model(weights_path)`. It returns an object with
`solve(input)`, a generator. The worker loads the model once and starts a fresh
generator for each question.

```python
from yugos.inference.actions import CallAction


class Router:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    def solve(self, input):
        choice = self.choose(input)  # one of the models.json entries
        yield CallAction(model=choice, input=input)


def load_model(weights_path):
    return Router(weights_path)
```

`input` is a `ModelInput`: Inspect chat messages, tools, and generation
settings. The generator can read a response and call again:

```python
reply = yield CallAction(model=choice, input=next_input)
# reply.text, reply.finish_reason, reply.tool_calls
```

Calls for one question are sequential. Finish by ending the generator or using
a bare `return`. The service grades the last provider response. Returning an
answer yourself, or stopping before any call, fails that question. The
submission has no network access during evaluation, so ship every tokenizer
and weight file the loader needs and open them relative to `weights_path`.

The archive must stay within 24 GiB and 4,096 files. `yugos pack` skips hidden
files, caches, other ZIPs, and `*.receipt.json`. It rejects symlinks. It writes
`manifest.json` itself.

## Test locally first

Local runs use your router and public tasks on your machine. They are not
competition results. Set an API key for every provider in `models.json`. The
hosted service does not supply those keys. The runner reads the environment
and does not load a `.env` file.

```sh
yugos pack . --output router.zip
yugos-evals pack build --benchmark aime2025 --limit 20 --out aime.pack.tar
yugos-evals run --pack aime.pack.tar --bundle router.zip --log-dir runs/a
```

`--max-cost` defaults to 100 and caps your own spend. Details are in the
[yugos-evals guide](../packages/yugos-evals/README.md#run-a-router-locally).

## Package

Packaging is local. It does not use a wallet, the network, or the service.

```sh
yugos pack ./my-router
```

The ZIP is written to `my-router/submission-<UTC timestamp>.zip` and the
command prints the path. `--name` overrides the submission name (the directory
name by default). `--output` chooses the destination. Existing files are not
overwritten.

## Submit

The default network is mainnet, subnet 34, at `https://upload.yugos.ai`.
`--network test` is subnet 169. That network has no built-in service, so pass
`--endpoint` or set `YUGOS_ENDPOINT`.

```sh
yugos submit ./submission.zip --network test --endpoint "$YUGOS_ENDPOINT" --wallet miner --hotkey default
yugos status ./submission.receipt.json --network test --endpoint "$YUGOS_ENDPOINT" --wallet miner --hotkey default
```

After one successful testnet submit, pack again or reuse the ZIP and submit
on mainnet with a new receipt. Do not reuse the testnet receipt.

```sh
yugos submit ./submission.zip --wallet miner --hotkey default
```

`submit` uploads the ZIP, confirms it, and publishes its commitment on chain. **Confirmation can reserve credits and start a paid evaluation.**

A terminal shows upload progress. Set `NO_COLOR=1` to disable color. The
command writes `submission.receipt.json` next to the ZIP (or `--receipt PATH`).
The receipt holds the upload session and no wallet secrets. Keep it private:
it also stores the link to your funding page. Rerun the same command with the
same ZIP to resume after a crash. A completed attempt is not evaluated again.

```sh
yugos status ./submission.receipt.json --wallet miner --hotkey default
```

`yugos status ./submission.zip` recovers the status link when the receipt is
missing. Status does not start evaluation.

To drop an unfinished upload:

```sh
yugos cancel-upload ./submission.receipt.json --wallet miner --hotkey default
```

That aborts storage only. A funded evaluation keeps running. Incomplete upload
sessions expire after 24 hours.

## Credits

`yugos submit` opens a funding page when payment is required. The page shows the reservation, your balance, and the run. The minimum payment is $0.50; extra funds stay on your hotkey's balance. You can close the browser after payment and resume with the same receipt.

The receipt's `status_url` returns you to that page. Quotes last one hour.
Rerun `submit` with the same receipt and `--renew-funding` to review a new
quote. Funds already received stay available. The private link lasts seven
days. `yugos status` can issue a new link.

Each hotkey has its own balance and can spend only that balance.

```sh
yugos credits claim --wallet miner --hotkey default
yugos credits balance --wallet miner --hotkey default
yugos credits history --wallet miner --hotkey default
yugos credits refund --all --wallet miner --hotkey default \
  --receipt ./refund-001.json
```

`claim` is signed by the coldkey and links historical credits to this billing
flow. It does not move funds. Use `--amount-usd 5` instead of `--all` for a
partial refund. Reuse the same refund receipt if the result is unclear.
Refunds go back to the original payment methods. An active run blocks
withdrawals until its bill is final.

## How scoring works

Your router is compared with the current champion on the same questions, in
three categories: chat and QA, math and code, and agentic work. Net value is
extra correct answers after extra spend and extra time. Hard questions count
more. You also have to stay close to the quality bar and finish inside the
time limit.

If you win, validators pay the current champion 85%, the previous champion 10%,
and the one before that 5%. Details and the formula are in
[Scoring](scoring.md).

## Try it locally

[`yugos-evals`](../packages/yugos-evals/README.md) can run your ZIP against a
development pack with your own API keys. Local runs are for debugging. They
are not competition scores: the live pack, prices, and hardware differ.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| Hotkey is not registered | Register on subnet 34, then rerun `submit`. |
| Request signature expired | Check the machine clock, keep the receipt, and rerun the same command. |
| Upload interrupted | Rerun `submit` on the same ZIP. The receipt resumes completed parts. |
| Receipt does not match this submission | Use the wallet and network that created it. |
| More funds needed | Open the status link and pay the shortfall, or add credits. |
| Funding is held by an earlier run | Wait until that bill is final. Keep the ZIP and receipt. |
| Funding page expired | Rerun `submit` with `--renew-funding` and the same receipt. |
| You closed the CLI during payment | Open `status_url` from the receipt, or run `yugos status`. |
| Refund looked like it failed | Retry with the same `--receipt`. Do not start a second refund file. |
| `cancel-upload` says the upload is complete | Storage is done. Use the status link for the run and refunds. |
| A question scored zero | The router called a model that was not in `models.json`, returned its own answer, or made no call. |

The receipt is the resume point. Keep the ZIP and the receipt together until
the run has a final bill.
