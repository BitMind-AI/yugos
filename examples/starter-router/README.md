# Starter router

Fixed-model baseline for checking packaging and local evaluation. Nothing in the
Yugos package imports this directory. It does not train a router; replace its
constant choice with learned inference before competitive submission.

Every model turn selects OpenRouter `openai/gpt-5-nano`. Verify that the choice
in `models.json` is supported by your world card. A local run needs
`OPENROUTER_API_KEY` and NumPy (`uv pip install numpy`).

- `model.py` validates `model.safetensors` and returns one `ModelChoice` from `route`.
- `models.json` lists the same model.
- `model.safetensors` supplies the required asset layout; it does not drive this
  baseline's fixed choice.

Pack it from this directory:

```sh
export OPENROUTER_API_KEY=...
yugos pack . --output ../starter-router.zip
yugos-evals pack build --benchmark aime2025 --limit 5 --out aime.pack.tar
yugos-evals run --pack aime.pack.tar --bundle ../starter-router.zip --log-dir runs/starter
```

The ZIP is the submission. `manifest.json` is generated inside it. See
[Mine on Yugos](../../docs/miner.md).
