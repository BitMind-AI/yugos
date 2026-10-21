# Providers and models

A router may call only models the hosted service can price. The live list is
the active world card. Fetch it. Do not copy model ids from a provider's own
docs and assume they are admitted.

No account is required.

```sh
curl -s https://upload.yugos.ai/world/cards/active \
  | jq -r '.card.providers[] | .provider_id as $p | .token_prices | keys[] | [$p, .] | @tsv'
```

Each line is `provider_id` then `model_id`. Those two strings are what
`models.json` and each `solve` call must use. Prices are USD per million
tokens under `token_prices`. The card changes as providers change their
prices. The response includes `sha256` for the card you just read.

`card.roles` lists grader and simulator models. Those are operator calls.
They are not extra models your router should declare.

## What you may declare

`models.json` is a JSON array. One object per distinct choice:

```json
[
  {"provider_id": "openai", "model_id": "gpt-5.6-sol", "effort": null}
]
```

`effort` may be `null`. A non-null effort is a different choice and must be
repeated in `models.json` if `solve` can yield it. Duplicate choices are
rejected.

The known provider ids are `openai`, `anthropic`, `google`, `openrouter`,
`venice`, `gm`, `engy`, `spacexai`, `nous`, `deepseek`, and `chutes`. The
card is the authority. A provider with no `token_prices` entry is not
available, even if it appears in this paragraph.

For a local run, set that provider's key yourself. The hosted service does
not give you these keys.

| Provider id | Environment variable |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `venice` | `VENICE_API_KEY` |
| `gm` | `GM_API_KEY` |
| `engy` | `ENGY_API_KEY` |
| `spacexai` | `XAI_API_KEY` |
| `nous` | `NOUS_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `chutes` | `CHUTES_API_KEY` |
