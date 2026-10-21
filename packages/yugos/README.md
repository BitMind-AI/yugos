# yugos

Subnet core for Yugos: competition scoring, the submission format, and the
`yugos` (miner) and `yugos-validator` commands.

```sh
uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git"
```

Operating instructions are in the repository docs:

- [Mine](https://github.com/BitMind-AI/yugos/blob/main/docs/miner.md)
- [Validate](https://github.com/BitMind-AI/yugos/blob/main/docs/validator.md)
- [Scoring](https://github.com/BitMind-AI/yugos/blob/main/docs/scoring.md)

`yugos[inspect]` adds Inspect Evals for hosts that run benchmarks. Miners and
validators do not need that extra.
