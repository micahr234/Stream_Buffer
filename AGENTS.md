# Agent guidance – Non_Stationary_Learning

Context for AI agents working in this repo.

## Project overview

**Purpose**: This project tests whether **context-conditioned reinforcement learning** can solve **non-stationary** sequential decision problems — i.e. environments whose dynamics, rewards, or structure change over time so that a fixed policy fails.

**Context-conditioned RL**: The agent conditions on history (or context) so that behavior can adapt without changing network weights at test time. This can be viewed as **meta-RL** with two loops:
- **Slow outer loop**: gradient descent over the transformer (and Q-head) parameters, so the network learns how to use context.
- **Fast inner loop**: “adaptation” happens in the **latent representation** inside the transformer — each forward pass consumes a history window and produces context-dependent Q-values, with no inner gradient steps.

**Test suite**: We use the **NS-Gym** suite ([ns_gym](https://nsgym.io)): Gymnasium-based environments with configurable non-stationarity (e.g. continuous or periodic parameter changes) to evaluate whether the context-conditioned Q-learning approach can track and exploit changing dynamics.

## Key locations

- **`configs/`** – YAML configs. All use the train path: set `train_interval` (0 = test rollouts only) or define test_envs/load_train_dataset to infer.
- **`src/`** – Core application code: run (unified entrypoint), train, config loading, stream store and dataset handling, env wrappers, models, and eval/test runners.
- **`scripts/`** – CLI entrypoint (`run`) used by `run.sh`; installed via pyproject.
- **`tests/`** – Unit and integration tests.

## Unified pathway

One entry point (`run`). `num_steps` drives the main loop. Each iteration can train (`train_interval`), eval (`eval_interval`), and/or run a test rollout (`test_interval`) — set any interval to 0 to disable it. Set `save_dataset.name` to save rollout data to Hugging Face.

## Workflow

- **`install.sh`** – Run once at the start (e.g. `source install.sh`) to set up Python (uv), git, and VSCode configs.
- **`run.sh`** – Pass a config path to run (e.g. `./run.sh configs/train.yaml`). All configs use the train path; set `train_interval: 0` with `test_envs` for test rollouts only (envs is not allowed); set `save_dataset.name` to save rollout data.
- **`connect.sh`** – Reattach to an existing tmux session if you lost the terminal connection.

### Adding new packages

Never use `pip install` directly. Instead:
1. Add the package to `pyproject.toml` under `[project].dependencies` (or `[project.optional-dependencies].dev` for dev-only tools like linters and test frameworks).
2. Re-run `bash install.sh` to rebuild the venv from scratch with the updated dependencies.
3. Use the venv at `./.venv` as normal.

## Data and dataset schema

Datasets are stored and retrieved via Hugging Face Datasets. **Do not change column order** — inconsistent ordering causes subtle bugs.
- **Canonical column order**: env_name, env_number, step_id, action, observation, reward, done. Preserve this order when writing or streaming.
- **Per-step token order**: action, obs (L tokens), reward, done. The first action in a rollout is always random (no prior context).
- **Saved rollout datasets** always include env_name and env_number — these are required constructor arguments on StreamStore.


## Running

- **Environment**: The project uses **Python 3** (see `pyproject.toml` for the required version). Use the project venv at `./.venv` (activate with `source .venv/bin/activate`).
- **Via run**: `./run.sh configs/<config>.yaml` — e.g. `configs/train.yaml` for training (with optional eval and test rollout), or `configs/test_random_small.yaml` for test rollouts only (`train_interval: 0`). Use `save_dataset.name` to save rollout data.
- **Other scripts and tests** (anything that does not use the `run` CLI): Activate the venv first (`source .venv/bin/activate`), then run with Python 3 — e.g. `pytest tests/`, `python3 -m src.some_module`, or `python3 path/to/script.py`. Same venv ensures consistent dependencies.

## StreamStore

- **Critical paths**: `append` (online rollout) and `__getitem__` (sampling for training). Keep these fast and simple. `from_dataset` / `to_dataset` are one-shot bulk operations — do not add complexity to optimise them.
- **Field names must come from `FIELD_TO_TYPE`**: never hardcode strings like `"observation"`, `"reward"`, `"done"`, `"action"` in code. Derive them via a reverse lookup on `FIELD_TO_TYPE` (e.g. `{v: k for k, v in field_to_type.items()}`). `CORE_FIELDS` was removed because it duplicated this information.

## Conventions

- Verify your work at least five times before considering it done.
- After any change, ask: do variable names still make sense?
- After any change, ask: could a refactor or restructure clarify the code?
- If you detect that code has been removed or changed, do not automatically revert it. Assume the change was intentional.
- As you learns other way in which this repo works or user preferences please suggest adding them to this document.
