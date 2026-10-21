# Working agreements

## Scope and ownership

- Treat README.md as the ownership and behavior specification. Establish a consumer
  and an owner before adding a capability.
- Port one complete supported path at a time. Do not copy whole source trees,
  create placeholder packages for future work, or add runtime dependencies on
  the previous implementation.
- Keep one implementation of each grading, metering, and scoring rule. Ycore
  and BRR consume these rules instead of reproducing them.
- The benchmark catalog, public source loaders, tasks Inspect Evals lacks and
  local runs belong to the `yugos-evals` package. Private selection, holdouts,
  acquisition for competition packs, dependency/image preparation, freezing and
  publication belong to Ycore. Core `yugos` consumes frozen packs; do not add
  source loaders, caches or acquisition to it, and never import `yugos_evals` from it.
- Delegate supported benchmark execution to Inspect/Inspect Evals. Do not wrap
  legacy Yugos benchmark runtimes or reproduce upstream prompts, agents, tools
  or scorers. Add a portable extension only for an established missing capability.
- Inspect execution uses prepared samples and installed, pinned task builders
  that construct native components without source acquisition. Packs retain
  data and plain settings, never component graphs. Do not add serialization
  adapters or a legacy execution fallback.
- This repository owns the complete subnet application: evaluation, validator
  operation, miner submission tools, and shared Bittensor integration. Include
  the adapters and operating instructions needed to run those applications.
- Keep wallets, chain access, provider clients, artifact storage, and runtime
  adapters at the application boundary. Evaluation and scoring must remain
  usable without Bittensor or a particular hosting service.
- Yugos owns portable worker behavior; the configured execution service owns
  isolation, transport, resources, and teardown. Ycore may provide a hosted
  implementation, but running a validator must not require its source checkout.
- Preserve grading and scoring semantics during a structural port. Report any
  discovered behavior defect separately instead of silently changing the rules.
- Do not introduce a local execution fallback for uploaded or generated code.
- Keep dependencies pointing from applications into the core. Core evaluation
  must not import a CLI, neuron, or hosting application.
- Use brief, plain-English docstrings for public operations and non-obvious
  helpers. Describe their responsibility; skip boilerplate for obvious accessors
  and constructors. Keep inline comments sparse.

## Layout

- The repository is a uv workspace. The subnet core is the `yugos` package.
  Its source and tests stay in `packages/yugos`; the root `pyproject.toml` is
  what `uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git"` builds.
  Benchmarks and local testing are the `yugos-evals` package in
  `packages/yugos-evals`.

## Verification

- Add tests for behavior, invariants, interface guarantees, and observed failure
  modes. Identify the plausible defect each new test would catch.
- Reuse relevant behavioral tests from the source implementation and compare
  equivalent inputs and outcomes during migration.
- Do not assert copied tuning values, implementation structure, or registry
  sizes just to freeze them. Remove redundant tests.
- Verify the installed package independently of the source checkout before
  claiming a port is self-contained.
- Local verification does not authorize paid experiments, image rebuilds,
  deployment, or migration of a consumer.

## Repository hygiene

- Keep documentation focused on architecture, data formats, and operating
  instructions. Put change-specific explanation and validation in the PR.
- Retain critical or non-reproducible evidence under the ignored `var/`
  directory. Verify remote retention when it is part of the workflow.
- Preserve historical experiment artifacts and dependency pins.
