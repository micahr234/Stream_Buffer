"""Configuration loading utilities for Differential Q-Learning.

Loads a single YAML config file. Essential keys depend on train_interval:
- train_interval>0: load_train_dataset.name required
- train_interval=0: test_envs with num_steps per env required

train_interval is inferred from load_train_dataset (->1) or test_envs (->0) if not set.
All configs use the train path; train_interval=0 runs only test rollouts.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class EnvConfig:
    env_id: str
    seed: int
    num_envs: int
    max_episode_steps: int | None
    kwargs: dict | None
    render: bool
    non_stationary_params: dict | None
    # Deploy only: steps to collect for this env; policy (e.g. random_actions) per env; split (train/eval) per env
    num_steps: int | None
    random_actions: bool | None
    split: str | None  # "train" or "eval"; default "train"


@dataclass
class Config:
    name: str
    max_steps: int
    batch_size: int
    reward_offset: float | None
    gamma: float
    gamma_done: float
    q_mean_loss_weight: float | None
    centering_factor: float
    q_training_enabled: bool
    train_interval: int  # 0 = rollout only; >0 = train every N steps (1 = every step)
    q_start_step: int
    q_num_steps: int
    max_epochs: int
    head_dim: int | None  # Q head output dim; max action index in dataset must be < head_dim
    sequence_length: int  # sequence length budget; we use as many context steps as fit
    base_model_id: str
    num_hidden_layers: int | None
    load_pretrained_backbone: bool  # initialize transformer from base_model_id weights
    torch_compile: bool  # torch.compile the transformer backbone for faster inference
    lr: float
    weight_decay: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    epsilon_start_step: int
    polyak_tau: float
    wandb_project: str
    wandb_run_name: str | None
    load_dataset: bool  # True when train_dataset_name is set (backward compat)
    save_dataset: bool
    train_dataset_name: str | None  # load_train_dataset.name (required for train)
    train_dataset_split: str  # load_train_dataset.split
    eval_dataset_name: str | None  # load_eval_dataset.name (optional)
    eval_dataset_split: str  # load_eval_dataset.split
    save_dataset_name: str | None  # from save_dataset.name (for rollout/train save)
    num_steps: int | None
    random_actions: bool | None
    q_save_model_name: str | None
    q_load_model_name: str | None
    run_id: str
    seed: int  # for training RNG; rollout uses per-env seeds
    env_id: str | None  # optional; for model save metadata when training (e.g. env name)
    # Validation (train mode): run eval every eval_interval steps; 0 = disabled
    eval_interval: int
    eval_batch_size: int  # number of batches to run for eval loss
    # Periodic online test rollout (train mode)
    test_interval: int  # run test rollout every N train steps; 0 = disabled
    test_start_length: int  # tokens for initial/re-prefill context (0 = use sequence_length)
    test_max_cache_length: int  # max KV-cache tokens before re-prefill (0 = no caching)
    test_env: EnvConfig | None  # deprecated; use test_envs
    test_envs: dict[str, EnvConfig]  # test environments; each must set num_steps


def _get(cfg, key: str, default):
    """Get nested key from OmegaConf/dict; return default if any part missing."""
    try:
        val = OmegaConf.select(cfg, key, default=None)
        return val if val is not None else default
    except Exception:
        return default


def _nonempty_name(val) -> str | None:
    """Return value as non-empty string or None (empty/whitespace → do not load/save)."""
    if val is None:
        return None
    s = (val if isinstance(val, str) else str(val)).strip()
    return s if s else None


def _parse_env_config(
    env_key: str,
    env_cfg: Any,
    used_seeds: set[int],
) -> EnvConfig:
    """Parse a single env config from YAML into EnvConfig."""
    env_key_str = str(env_key)
    if not OmegaConf.is_dict(env_cfg):
        raise ValueError(f"envs.{env_key_str} must be a dict of env options.")

    ec = dict(env_cfg)
    resolved_env_id = str(ec.get("id", env_key_str))
    kwargs_val = ec.get("kwargs", None)
    kwargs_norm = dict(kwargs_val) if kwargs_val is not None else None
    render = bool(ec.get("render", False))
    num_envs = int(ec.get("num_envs", 1))
    if render and num_envs != 1:
        raise ValueError(
            f"envs.{env_key_str}.render=true requires envs.{env_key_str}.num_envs: 1"
        )

    non_stationary_val = ec.get("non_stationary_params", None)
    non_stationary_norm = (
        dict(non_stationary_val) if non_stationary_val is not None else None
    )

    # Rollout-only fields (optional at parse-time; validated when train_interval=0)
    num_steps_env = ec.get("num_steps", None)
    if num_steps_env is not None:
        num_steps_env = int(num_steps_env)
    policy = ec.get("policy", None)
    random_actions_env = ec.get("random_actions", None)
    if random_actions_env is None and policy is not None:
        random_actions_env = str(policy).lower() == "random"
    if random_actions_env is None:
        random_actions_env = True
    else:
        random_actions_env = bool(random_actions_env)

    split_env = ec.get("split", None)
    if split_env is not None:
        split_env = str(split_env).lower()
        if split_env not in ("train", "eval"):
            raise ValueError(
                f"envs.{env_key_str}.split must be 'train' or 'eval', got {split_env!r}."
            )
    else:
        split_env = "train"

    seed_val = ec.get("seed", None)
    if seed_val is None or str(seed_val).lower() == "none":
        # Assign a unique seed so all envs with seed: None get different seeds.
        base = random.randint(0, 2**31 - 1)
        while base in used_seeds:
            base = random.randint(0, 2**31 - 1)
        used_seeds.add(base)
        seed_val = base
    else:
        seed_val = int(seed_val)

    return EnvConfig(
        env_id=resolved_env_id,
        seed=int(seed_val),
        num_envs=num_envs,
        max_episode_steps=ec.get("max_episode_steps", None),
        kwargs=kwargs_norm,
        render=render,
        non_stationary_params=non_stationary_norm,
        num_steps=num_steps_env,
        random_actions=random_actions_env,
        split=split_env,
    )


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file.

    Essential keys depend on train_interval:
    - train_interval=0: test_envs with num_steps per env required
    - train_interval>0: load_train_dataset.name required

    train_interval defaults: 1 if load_train_dataset set, else 0 if test_envs set.

    Raises:
        ValueError: If config_path is empty or an essential key is missing.
        FileNotFoundError: If the config file does not exist.
    """
    if not (config_path and config_path.strip()):
        raise ValueError("Config file path is required. Usage: run <config.yaml>")

    config_path = config_path.strip()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml_cfg = OmegaConf.load(config_path)

    # --- train_interval (0 = test rollouts only; >0 = train every N steps) ---
    _train_interval_raw = _get(yaml_cfg, "loop.train_interval", None)
    _has_load_train = _nonempty_name(_get(yaml_cfg, "load_train_dataset.name", None)) is not None
    _has_test_envs = OmegaConf.is_dict(yaml_cfg.get("test_envs"))
    _has_test_env = OmegaConf.is_dict(yaml_cfg.get("test_env"))

    if yaml_cfg.get("envs") is not None or yaml_cfg.get("env") is not None:
        raise ValueError("Config uses deprecated 'envs' or 'env'. Use test_envs only.")

    if _train_interval_raw is not None:
        _train_interval = int(_train_interval_raw)
    else:
        if _has_load_train:
            _train_interval = 1
        elif _has_test_envs or _has_test_env:
            _train_interval = 0
        else:
            raise ValueError(
                "Config must define load_train_dataset.name (training) or test_envs (test rollouts only). "
                "Or set loop.train_interval explicitly."
            )

    used_seeds: set[int] = set()

    # test_envs: from test_envs or test_env (single)
    test_envs_raw = yaml_cfg.get("test_envs")
    test_env_raw = yaml_cfg.get("test_env")
    test_envs_parsed: dict[str, EnvConfig] = {}
    test_env_cfg: EnvConfig | None = None
    if test_envs_raw is not None and OmegaConf.is_dict(test_envs_raw):
        for ek, ev in dict(test_envs_raw).items():
            test_envs_parsed[str(ek)] = _parse_env_config(str(ek), ev, used_seeds)
    elif test_env_raw is not None and OmegaConf.is_dict(test_env_raw):
        test_env_cfg = _parse_env_config("test_env", test_env_raw, used_seeds)
        test_envs_parsed = {"test_env": test_env_cfg} if test_env_cfg else {}

    if _train_interval == 0:
        if not test_envs_parsed:
            raise ValueError(
                "Config must define test_envs when train_interval=0."
            )
        for ek, ev in test_envs_parsed.items():
            if ev.num_steps is None:
                raise ValueError(
                    f"test_envs.{ek}.num_steps is required when train_interval=0. Set num_steps per env."
                )

    # --- Optional: name ---
    name = _get(yaml_cfg, "name", None) or _get(yaml_cfg, "wandb.run_name", None) or "differential_q_learning"

    # --- Optional: q_training (enabled when train_interval > 0) ---
    q_training_section = yaml_cfg.get("loop")
    q_training_enabled = _train_interval > 0

    # In-code defaults for optional parameters
    cfg_dict = {
        "name": name,
        "q_training_enabled": q_training_enabled,
        "train_interval": _train_interval,
        "q_start_step": _get(yaml_cfg, "loop.start_step", 0) if q_training_enabled else 0,
        "q_num_steps": int(_get(yaml_cfg, "loop.num_steps", 25000)),
        "max_epochs": int(_get(yaml_cfg, "loop.max_epochs", 0)) if q_training_enabled else 0,
        "batch_size": _get(yaml_cfg, "loop.batch_size", 128) if q_training_enabled else 128,
        "reward_offset": None,
        "gamma": float(_get(yaml_cfg, "loop.gamma", 0.99)),
        "gamma_done": float(_get(yaml_cfg, "loop.gamma_done", 0.0)),
        "q_mean_loss_weight": None,
        "centering_factor": _get(yaml_cfg, "loop.centering_factor", 0.1) if q_training_enabled else 0.1,
        "head_dim": int(_get(yaml_cfg, "q_network.head_dim", 0)) or None,
        "sequence_length": int(_get(yaml_cfg, "loop.sequence_length", 512)),
        "base_model_id": str(_get(yaml_cfg, "q_network.base_model_id", "meta-llama/Llama-3.2-1B")),
        "num_hidden_layers": (
            int(_get(yaml_cfg, "q_network.num_hidden_layers", 0))
            if _get(yaml_cfg, "q_network.num_hidden_layers", None) is not None
            else None
        ),
        "load_pretrained_backbone": bool(_get(yaml_cfg, "q_network.load_pretrained_backbone", False)),
        "torch_compile": bool(_get(yaml_cfg, "q_network.torch_compile", False)),
        "lr": _get(yaml_cfg, "loop.lr", 0.0001) if q_training_enabled else 0.0001,
        "weight_decay": _get(yaml_cfg, "loop.weight_decay", 0.0) if q_training_enabled else 0.0,
        "q_save_model_name": _nonempty_name(_get(yaml_cfg, "loop.save_model_name", None)) if q_training_section else None,
        "q_load_model_name": _nonempty_name(_get(yaml_cfg, "loop.load_model_name", None)) if q_training_section else None,
        "eval_interval": int(_get(yaml_cfg, "loop.eval_interval", 0)) if q_training_enabled else 0,
        "eval_batch_size": int(_get(yaml_cfg, "loop.eval_batch_size", 32)) if q_training_enabled else 32,
        "test_interval": (
            int(_get(yaml_cfg, "loop.test_interval", 1)) if _train_interval == 0
            else int(_get(yaml_cfg, "loop.test_interval", 0)) if q_training_enabled else 0
        ),
        "test_start_length": int(_get(yaml_cfg, "loop.test_start_length", 512)) if (_train_interval == 0 or q_training_enabled) else 0,
        "test_max_cache_length": int(_get(yaml_cfg, "loop.test_max_cache_length", 0)) if (_train_interval == 0 or q_training_enabled) else 0,
        "test_env": test_env_cfg if q_training_enabled and test_env_cfg else None,
        "test_envs": test_envs_parsed,
        "epsilon_start": _get(yaml_cfg, "exploration.epsilon_start", 1.0),
        "epsilon_min": _get(yaml_cfg, "exploration.epsilon_min", 0.02),
        "epsilon_decay": _get(yaml_cfg, "exploration.epsilon_decay", 0.9995),
        "epsilon_start_step": _get(yaml_cfg, "exploration.epsilon_start_step", 0),
        "polyak_tau": _get(yaml_cfg, "loop.polyak_tau", 0.0) if q_training_section else 0.0,
        "wandb_project": _get(yaml_cfg, "wandb.project", "DiffQ"),
        "wandb_run_name": _get(yaml_cfg, "wandb.run_name", None),
        "train_dataset_name": _nonempty_name(_get(yaml_cfg, "load_train_dataset.name", None)),
        "train_dataset_split": str(_get(yaml_cfg, "load_train_dataset.split", "train")).strip() or "train",
        "eval_dataset_name": _nonempty_name(_get(yaml_cfg, "load_eval_dataset.name", None)),
        "eval_dataset_split": str(_get(yaml_cfg, "load_eval_dataset.split", "eval")).strip() or "eval",
        "save_dataset_name": _nonempty_name(_get(yaml_cfg, "save_dataset.name", None)),
        "load_dataset": _nonempty_name(_get(yaml_cfg, "load_train_dataset.name", None)) is not None,
        "save_dataset": _nonempty_name(_get(yaml_cfg, "save_dataset.name", None)) is not None,
        "num_steps": None,
        "random_actions": None,
        "run_id": str(random.randint(10000000, 99999999)),
        "seed": int(_get(yaml_cfg, "seed", 42)),
        "env_id": _get(yaml_cfg, "env_id", None),
    }


    cfg_dict["max_steps"] = cfg_dict["q_start_step"] + cfg_dict["q_num_steps"]

    if _train_interval > 0 and not cfg_dict.get("train_dataset_name"):
        raise ValueError(
            "Config must define load_train_dataset.name when train_interval>0. "
            "Example: load_train_dataset: { name: 'user/dataset', split: 'train' }"
        )

    if cfg_dict["test_interval"] > 0:
        if not test_envs_parsed:
            raise ValueError(
                "Config must define test_envs (or test_env) when test_interval > 0."
            )
        for ek, ev in test_envs_parsed.items():
            if ev.num_steps is None:
                raise ValueError(
                    f"test_envs.{ek}.num_steps is required. Set num_steps per env."
                )
            if ev.non_stationary_params is None:
                raise ValueError(
                    f"test_envs.{ek}.non_stationary_params is required when loop.test_interval > 0."
                )

    if q_training_enabled and cfg_dict["max_epochs"] < 0:
        raise ValueError("loop.max_epochs must be >= 0.")

    return Config(**cfg_dict)
