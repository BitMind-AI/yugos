# Run a Yugos validator

The validator reads the subnet's latest competition result, checks that it is
intact and fresh, maps champion hotkeys to current UIDs, and sets weights.
It does not download miner code, call providers, or regrade answers.

| Network | Flag | Subnet | Service |
| --- | --- | --- | --- |
| Mainnet | omitted (`finney`) | **34** | `https://upload.yugos.ai` |
| Testnet | `--network test` | **169** | pass `--endpoint` |

Weights are 85% for the current champion, 10% for the previous champion, and
5% for the one before that. A missing older seat is added to the current
champion. If the current champion is not registered, the validator sets no new
weights.

## Requirements

- Python 3.11 or newer, and the `yugos` package.
- A Bittensor wallet whose hotkey is registered on the subnet and has a
  **validator permit**, or is the subnet owner hotkey.
- Outbound access to the chain and to the results endpoint over HTTPS.
- A persistent disk for the state file. The state file is locked; two
  processes cannot share it.
- The results endpoint. Mainnet results are served at
  `https://upload.yugos.ai`.

No provider API keys. No benchmark data on the validator machine.

## Install

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install "yugos @ git+https://github.com/BitMind-AI/yugos.git"
```

Yugos does not create wallets. Place an existing wallet at
`~/.bittensor/wallets`, or pass `--wallet-path`.

## Wallet and hotkey

Register and stake with the Bittensor CLI until the hotkey has a validator
permit on the subnet. The process signs the snapshot request with Bittensor's
signed-request headers and signs weight extrinsics with that hotkey. The
coldkey is not required at runtime if the hotkey file is available.

Confirm the hotkey and the subnet's weights version before the first publish:

```python
import bittensor as bt

subtensor = bt.Subtensor("finney")
print(subtensor.read("subnet_hyperparameters", netuid=34))
```

Pass the subnet's `weights_version` as
`--version-key`. The flag defaults to `0`, which the chain rejects when the
subnet requires another version.

## Mainnet (finney, netuid 34)

Preview one cycle without setting weights:

```sh
yugos-validator \
  --wallet validator \
  --hotkey default \
  --version-key WEIGHTS_VERSION \
  --once \
  --dry-run
```

Drop `--once --dry-run` to publish on an interval. Testnet uses `--network test`
and `--endpoint`; its default state file is separate.

```sh
yugos-validator \
  --network test \
  --endpoint "$YUGOS_ENDPOINT" \
  --wallet validator \
  --hotkey default \
  --once \
  --dry-run
```

## Flags

| Flag | Default | Role |
| --- | --- | --- |
| `--network` | `finney` | `finney` (subnet 34) or `test` (subnet 169). |
| `--wallet` | required | Existing wallet name. |
| `--hotkey` | `default` | Hotkey name inside the wallet. |
| `--wallet-path` | `~/.bittensor/wallets` | Wallet directory. |
| `--endpoint` | network default | Results service. HTTPS, no credentials or query string. |
| `--state` | automatic | Optional override for the durable JSON state file. |
| `--version-key` | `0` | Chain weights version. Set this to the subnet's `weights_version`. |
| `--blocks-between-weights` | `100` | Minimum blocks between successful publishes; the chain rate limit also applies. |
| `--max-snapshot-age-seconds` | `86400` | Reject results older than this. |
| `--once` | off | Exit after one cycle. |
| `--dry-run` | off | Log the weight vector and skip the extrinsic. |

`--cohort` is accepted and ignored so older launch scripts keep working.

By default, state lives in `$XDG_STATE_HOME/yugos/validators/<scope-digest>.json`,
or `~/.local/state/yugos/validators/<scope-digest>.json` when `XDG_STATE_HOME` is
unset, empty, or relative. The filename is derived from the network, subnet,
normalized endpoint and hotkey address, so restarts reuse progress and different
validators use separate files. The resolved path is logged at startup. Keep that
directory on persistent storage, or set `--state` to choose another location.

SIGINT and SIGTERM stop waiting or retrying after the in-flight request or bounded
SDK wait returns, then exit.

## Process supervision

Run one validator process per state file under a supervisor that starts it on
boot and restarts it if it exits.

### PM2

With Node.js and PM2 installed, use
[`validator.config.js`](../validator.config.js) from a checkout with Yugos
installed in `.venv`. Edit its `args` for your wallet,
hotkey and `--version-key`. Testnet also needs `--network test` and `--endpoint`.
Change `script` if your virtualenv is elsewhere. The config launches one process,
retries exits after 15 seconds, and allows 180 seconds for a cycle to finish on
stop. [PM2's application settings](https://pm2.keymetrics.io/docs/usage/application-declaration/)
describe the supervisor options.

```sh
yugos validator start
pm2 logs yugos-validator
pm2 save
pm2 startup
```

`yugos v start` is an alias. Run from the checkout root, or pass
`--config /path/to/validator.config.js`. The command wraps `pm2 start` and
preserves its output and exit status.

Run the command printed by `pm2 startup` to enable startup on boot. After editing
the config, use `pm2 restart validator.config.js` and `pm2 save`. Use
`pm2 stop yugos-validator` before running a manual cycle with the same state.

### systemd

Example unit
(`/etc/systemd/system/yugos-validator.service`):

```ini
[Unit]
Description=Yugos validator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=yugos
Group=yugos
Environment=PATH=/opt/yugos/.venv/bin:/usr/bin
ExecStart=/opt/yugos/.venv/bin/yugos-validator \
  --wallet validator \
  --hotkey default \
  --wallet-path /var/lib/yugos/wallets \
  --state /var/lib/yugos/validator-finney.json \
  --version-key WEIGHTS_VERSION
Restart=on-failure
RestartSec=15
KillSignal=SIGTERM
TimeoutStopSec=180

[Install]
WantedBy=multi-user.target
```

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now yugos-validator
```

Put the wallet and resolved state file on persistent storage. A host restart
must come back to the same paths. The state file records the network, netuid,
endpoint, and hotkey. Changing any of those selects a new default file; when
overriding `--state`, choose a separate file for the new identity.

## Logs and monitoring

Logs go to stderr at INFO. Each cycle logs the current chain block and the
next eligible weight block, followed by its outcome:

```text
2026-10-16 12:00:00,000 INFO chain block: 1000; next weight block: 1000
2026-10-16 12:00:00,000 INFO published weights for 2 miners from snapshot <id>
2026-10-16 12:00:00,000 INFO waiting for chain block 1100
2026-10-16 12:20:00,000 INFO chain block: 1100; next weight block: 1100
2026-10-16 12:20:00,000 INFO no eligible registered miners; weights unchanged
2026-10-16 12:20:00,000 INFO waiting for chain block 1200
```

With systemd: `journalctl -u yugos-validator -f`.

The next publish is due after the larger of `--blocks-between-weights` and the
subnet's `weights_rate_limit`, measured from the latest on-chain or saved
successful update. The first publish can run immediately. Between attempts, the
validator waits for the target block using the SDK's block subscription.

Every weight submission requires a fresh snapshot fetched and validated during
that attempt. Snapshot transport failures, unavailable publications, rate limits,
and server errors get up to ten attempts, with delays increasing from five to
60 seconds. Authentication errors and invalid snapshots fail immediately.
Dry runs use the same snapshot retries. `--once` also performs those retries,
then exits after success or exhausted retries.

After an unsuccessful attempt or a result with no eligible weights, the validator
waits another weight interval before trying again. Previous on-chain weights
stay in place. The validator does not publish when
the result is missing, stale, older than the last accepted result, paused,
empty, or has no registered champion.

The state file stores the last accepted snapshot and the last successful
publish block. Use it to see progress after a restart. Compare the logged
weight UIDs with the chain if you need to confirm inclusion. Commit-reveal
subnets can include a transaction before the weights are revealed; check the
chain again after the reveal delay.

## Upgrades

1. Read the release notes for the new `yugos` ref.
2. `sudo systemctl stop yugos-validator`.
3. Install that ref into the same virtualenv.
4. `sudo systemctl start yugos-validator` with the same `--state` path.

Within competition major version `yugos-overall-v2`, a new benchmark, policy
file, or scoring tweak is picked up from the results service. Install a new
Yugos release when that major version changes. Keep the state file. A publish
that fails is not recorded as success, so the next cycle retries.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `hotkey has no validator permit` | Stake until the hotkey has a permit, or use the subnet owner hotkey. |
| `hotkey is not registered` | Register the hotkey on netuid 34. |
| `state belongs to another validator or service` | The state file is bound to network, netuid, endpoint, and hotkey. Use the matching flags or a new state file. |
| `another process is using <state>` | Stop the other process. A stale lock clears when that process exits. |
| `competition confirmation is stale` | The service has not published a fresh result inside `--max-snapshot-age-seconds`. Weights stay as they are. |
| `competition result rolled back` | The new result is older than one already accepted. Do not delete state to force it through. |
| `subnet weight limits do not permit winner-take-all allocation` | The subnet hyperparameters cannot represent this weight vector. Fix the subnet configuration before publishing. |
| Chain rejected the weights version | Set `--version-key` from `weights_version` in the subnet hyperparameters. |
| Service errors every cycle | The process keeps running and retries. Existing weights remain until a valid result is published. |
| Champion left the subnet | Publication stops. It resumes when a registered champion is in the result. |
