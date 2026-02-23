# Non-Stationary Learning

Tests whether **context-conditioned reinforcement learning** can solve non-stationary sequential decision problems — environments whose dynamics, rewards, or structure change over time so that a fixed policy fails.

The agent conditions on history via a transformer, adapting through its latent representation at inference time rather than through gradient steps. Evaluated on [NS-Gym](https://nsgym.io): Gymnasium environments with configurable non-stationarity.

## Setup

```bash
source install.sh
```

## Running

All runs go through `run.sh` with a config file:

```bash
./run.sh configs/<config>.yaml
```

### Collect rollout data

Set `train_interval: 0` and define `test_envs`. Set `save_dataset.name` to push the dataset to Hugging Face.

```bash
./run.sh configs/test_random_small.yaml
```

### Train

Set `load_train_dataset.name` to a Hugging Face dataset repo. `train_interval: 1` is the default when a train dataset is provided.

```bash
./run.sh configs/train.yaml
```

Optionally enable eval (`eval_interval > 0`) and online test rollouts (`test_interval > 0`) within the same run.

## Config reference

| Key | Description |
|---|---|
| `q_training.train_interval` | Train every N steps. `0` = rollout-only (no training). |
| `q_training.eval_interval` | Eval every N steps. `0` = disabled. Requires `load_eval_dataset`. |
| `q_training.test_interval` | Online test rollout every N train steps. `0` = disabled. |
| `test_envs` | Dict of environments to roll out in. Each entry specifies `id`, `num_envs`, `num_steps`, `policy`, and optionally `non_stationary_params`. |
| `save_dataset.name` | Hugging Face repo id to push rollout data to. `null` = skip. |

## Dataset schema

Columns are always written in this order:

```
env_name, env_number, step_id, action, observation, reward, done
```

## Non-stationary environments

Add `non_stationary_params` to any env config to use [NS-Gym](https://nsgym.io) schedulers:

```yaml
test_envs:
  cartpole_ns:
    id: CartPole-v1
    num_envs: 20
    num_steps: 1000
    policy: random
    non_stationary_params:
      gravity:
        scheduler: "periodic"
        update_function: "random_walk"
        scheduler_kwargs:
          period: 10
        update_kwargs:
          sigma: 0.5
          mu: 0.0
```

See the [NS-Gym docs](https://nsgym.io) and [paper](https://openreview.net/pdf?id=YOXZuRy40U) for available schedulers and update functions.

## Tests

```bash
source .venv/bin/activate
pytest tests/
```
