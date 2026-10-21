# Starter router

Example miner submission. Nothing in the Yugos package imports this directory.

Every question calls OpenRouter `openai/gpt-5-nano`, a cheap model on the
active world card. A local run needs `OPENROUTER_API_KEY`. The hosted service
uses its own key.

- `model.py` loads `model.safetensors` and yields that one call from `solve`.
- `models.json` lists the same model.
- `model.safetensors` is the required weights file. This starter does not read it.

Pack it from this directory:

```sh
export OPENROUTER_API_KEY=...
yugos pack . --output ../starter-router.zip
yugos-evals pack build --benchmark aime2025 --limit 5 --out aime.pack.tar
yugos-evals run --pack aime.pack.tar --bundle ../starter-router.zip --log-dir runs/starter
```

The ZIP is the submission. `manifest.json` is generated inside it. See
[Mine on Yugos](../../docs/miner.md).
