"""Tests for config.py: load_config and EnvConfig parsing.

Uses temporary YAML files so tests are self-contained and do not depend on
any real HuggingFace datasets or environments.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_config, Config, EnvConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# Single env block ready to paste into test_envs:
_ENV1 = textwrap.dedent("""\
    test_envs:
      env1:
        id: CartPole-v1
        seed: 1
        num_envs: 1
        num_steps: 50
        render: false
        non_stationary_params:
          gravity:
            scheduler: periodic
            update_function: random_walk
            scheduler_kwargs:
              period: 10
            update_kwargs:
              sigma: 0.5
              mu: 0.0
              seed: 0
""")


def _single_env_yaml(
    key: str = "env1",
    seed: int = 1,
    num_envs: int = 1,
    num_steps: int = 50,
    split: str | None = None,
    render: bool = False,
    include_num_steps: bool = True,
) -> str:
    # Extra lines are indented to match the 12-space depth used for other env fields
    # (before textwrap.dedent strips the 8-space common prefix they become 4-space).
    split_line = f"\n            split: {split}" if split is not None else ""
    num_steps_line = f"\n            num_steps: {num_steps}" if include_num_steps else ""
    render_str = "true" if render else "false"
    return textwrap.dedent(f"""\
        test_envs:
          {key}:
            id: CartPole-v1
            seed: {seed}
            num_envs: {num_envs}
            render: {render_str}{num_steps_line}{split_line}
            non_stationary_params:
              gravity:
                scheduler: periodic
                update_function: random_walk
                scheduler_kwargs:
                  period: 10
                update_kwargs:
                  sigma: 0.5
                  mu: 0.0
                  seed: 0
    """)


# ---------------------------------------------------------------------------
# 1. Rollout-only configs (train_interval=0)
# ---------------------------------------------------------------------------


def test_rollout_only_minimal(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("cartpole", seed=1, num_steps=50))
    cfg = load_config(path)
    assert cfg.train_interval == 0
    assert "cartpole" in cfg.test_envs
    ec = cfg.test_envs["cartpole"]
    assert ec.env_id == "CartPole-v1"
    assert ec.num_steps == 50
    assert ec.num_envs == 1


def test_rollout_only_infers_train_interval_zero(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=2, num_envs=2, num_steps=100))
    cfg = load_config(path)
    assert cfg.train_interval == 0


def test_rollout_only_test_interval_defaults_to_one(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=3, num_steps=10))
    cfg = load_config(path)
    assert cfg.test_interval == 1


def test_rollout_only_missing_test_envs_raises(tmp_path):
    path = _write_yaml(tmp_path, "loop:\n  train_interval: 0\n")
    with pytest.raises(ValueError, match="test_envs"):
        load_config(path)


def test_rollout_only_missing_num_steps_raises(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=4, include_num_steps=False))
    with pytest.raises(ValueError, match="num_steps"):
        load_config(path)


# ---------------------------------------------------------------------------
# 2. Train configs (train_interval>0)
# ---------------------------------------------------------------------------


def test_train_infers_train_interval_one(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        load_train_dataset:
          name: user/dataset
          split: train
        q_network:
          head_dim: 4
          base_model_id: meta-llama/Llama-3.2-1B
        loop:
          num_steps: 100
    """))
    cfg = load_config(path)
    assert cfg.train_interval == 1


def test_train_missing_dataset_raises(tmp_path):
    path = _write_yaml(tmp_path, "loop:\n  train_interval: 1\n")
    with pytest.raises(ValueError, match="load_train_dataset"):
        load_config(path)


def test_train_explicit_train_interval(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        load_train_dataset:
          name: user/dataset
          split: train
        loop:
          train_interval: 5
          num_steps: 1000
    """))
    cfg = load_config(path)
    assert cfg.train_interval == 5


def test_train_eval_interval_defaults_zero(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        load_train_dataset:
          name: user/dataset
          split: train
        loop:
          num_steps: 100
    """))
    cfg = load_config(path)
    assert cfg.eval_interval == 0


def test_train_test_interval_with_test_envs(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        load_train_dataset:
          name: user/dataset
          split: train
        loop:
          num_steps: 100
          test_interval: 10
        test_envs:
          env1:
            id: CartPole-v1
            seed: 5
            num_envs: 1
            num_steps: 50
            render: false
            non_stationary_params:
              gravity:
                scheduler: periodic
                update_function: random_walk
                scheduler_kwargs:
                  period: 10
                update_kwargs:
                  sigma: 0.5
                  mu: 0.0
                  seed: 0
    """))
    cfg = load_config(path)
    assert cfg.test_interval == 10
    assert "env1" in cfg.test_envs


def test_train_test_interval_without_test_envs_raises(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        load_train_dataset:
          name: user/dataset
          split: train
        loop:
          num_steps: 100
          test_interval: 10
    """))
    with pytest.raises(ValueError):
        load_config(path)


# ---------------------------------------------------------------------------
# 3. EnvConfig parsing
# ---------------------------------------------------------------------------


def test_env_config_default_split_is_train(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=6, num_steps=10))
    cfg = load_config(path)
    assert cfg.test_envs["env1"].split == "train"


def test_env_config_explicit_split_eval(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=7, num_steps=10, split="eval"))
    cfg = load_config(path)
    assert cfg.test_envs["env1"].split == "eval"


def test_env_config_invalid_split_raises(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        test_envs:
          env1:
            id: CartPole-v1
            seed: 8
            num_envs: 1
            num_steps: 10
            split: something_else
            render: false
            non_stationary_params:
              gravity:
                scheduler: periodic
                update_function: random_walk
                scheduler_kwargs:
                  period: 10
                update_kwargs:
                  sigma: 0.5
                  mu: 0.0
                  seed: 0
    """))
    with pytest.raises(ValueError, match="split"):
        load_config(path)


def test_env_config_render_with_multi_env_raises(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=9, num_envs=3, num_steps=10, render=True))
    with pytest.raises(ValueError, match="render"):
        load_config(path)


def test_env_config_seed_none_assigns_unique_seeds(tmp_path):
    path = _write_yaml(tmp_path, textwrap.dedent("""\
        test_envs:
          env1:
            id: CartPole-v1
            seed: null
            num_envs: 1
            num_steps: 10
            render: false
            non_stationary_params:
              gravity:
                scheduler: periodic
                update_function: random_walk
                scheduler_kwargs:
                  period: 10
                update_kwargs:
                  sigma: 0.5
                  mu: 0.0
                  seed: 0
          env2:
            id: CartPole-v1
            seed: null
            num_envs: 1
            num_steps: 10
            render: false
            non_stationary_params:
              gravity:
                scheduler: periodic
                update_function: random_walk
                scheduler_kwargs:
                  period: 10
                update_kwargs:
                  sigma: 0.5
                  mu: 0.0
                  seed: 0
    """))
    cfg = load_config(path)
    seed1 = cfg.test_envs["env1"].seed
    seed2 = cfg.test_envs["env2"].seed
    assert isinstance(seed1, int)
    assert isinstance(seed2, int)
    assert seed1 != seed2, "null seeds must be unique across envs"


# ---------------------------------------------------------------------------
# 4. Deprecated keys raise errors
# ---------------------------------------------------------------------------


def test_deprecated_envs_key_raises(tmp_path):
    path = _write_yaml(tmp_path, "envs:\n  foo:\n    id: CartPole-v1\n")
    with pytest.raises(ValueError, match="deprecated"):
        load_config(path)


def test_deprecated_env_key_raises(tmp_path):
    path = _write_yaml(tmp_path, "env:\n  id: CartPole-v1\n")
    with pytest.raises(ValueError, match="deprecated"):
        load_config(path)


# ---------------------------------------------------------------------------
# 5. Config field values
# ---------------------------------------------------------------------------


def test_config_defaults(tmp_path):
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=10, num_steps=5))
    cfg = load_config(path)
    assert cfg.gamma > 0
    assert isinstance(cfg.seed, int)
    assert cfg.max_steps >= 1


def test_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/definitely/does/not/exist.yaml")


def test_config_empty_path_raises():
    with pytest.raises(ValueError, match="required"):
        load_config("   ")


def test_config_no_envs_field(tmp_path):
    """Config should NOT have an 'envs' attribute (it was removed as stale)."""
    path = _write_yaml(tmp_path, _single_env_yaml("env1", seed=11, num_steps=5))
    cfg = load_config(path)
    assert not hasattr(cfg, "envs"), "Stale 'envs' field must not exist on Config"
