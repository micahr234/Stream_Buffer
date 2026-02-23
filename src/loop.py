"""Main entry point: train (optional), eval (optional), test rollouts, save."""

import dataclasses
import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

from config import Config, EnvConfig, load_config
from stream_store import StreamStore
from runner_eval import RunnerEval
from definitions import FIELD_TO_TYPE
from models import OfflineDQNTransformer
from runner_test import RunnerTest
from runner_train import RunnerTrain
from auth import setup_huggingface, setup_wandb


@dataclass
class _TestRunContext:
    """One test env: runner, config, and stores for saving rollout data."""

    env_key: str
    runner: RunnerTest
    env_cfg: EnvConfig
    stores: list[StreamStore]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"✅ CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("⚠️  CUDA is not available. Using CPU.")
    return torch.device("cpu")


def _setup_train_runner(cfg: Config) -> RunnerTrain | None:
    """Load train dataset and create RunnerTrain. None when train_interval=0."""
    if cfg.train_interval <= 0:
        return None
    if cfg.head_dim is None:
        raise ValueError("q_network.head_dim is required for training.")
    print(f"📥 Loading train dataset ({cfg.train_dataset_name}, split={cfg.train_dataset_split})...")
    store = StreamStore(field_to_type=FIELD_TO_TYPE, env_name="", env_number=0)
    store.load_dataset(
        dataset_name=cfg.train_dataset_name,
        dataset_split=cfg.train_dataset_split,
    )
    runner = RunnerTrain(
        store=store,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
    )
    print(
        f"✅ Train dataset ready with {len(store)} tokens, "
        f"batch_size={cfg.batch_size}, sequence_length={cfg.sequence_length}."
    )
    return runner


def _setup_eval_runner(cfg: Config) -> RunnerEval | None:
    """Load eval dataset and create RunnerEval. None when eval_interval=0."""
    if cfg.eval_interval <= 0:
        return None
    if not cfg.eval_dataset_name:
        raise ValueError("load_eval_dataset.name is required when eval_interval > 0.")
    print(f"📥 Loading eval dataset ({cfg.eval_dataset_name}, split={cfg.eval_dataset_split})...")
    store = StreamStore(field_to_type=FIELD_TO_TYPE, env_name="", env_number=0)
    store.load_dataset(
        dataset_name=cfg.eval_dataset_name,
        dataset_split=cfg.eval_dataset_split,
    )
    runner = RunnerEval(
        store=store,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.eval_batch_size,
    )
    print(
        f"✅ Eval dataset ready with {len(store)} tokens, "
        f"batch_size={cfg.eval_batch_size}, sequence_length={cfg.sequence_length}."
    )
    return runner


def _setup_test_runners(cfg: Config, device: torch.device) -> list[_TestRunContext]:
    """Create one RunnerTest per test_env. Empty when test_interval=0 or no test_envs."""
    if cfg.test_interval <= 0 or not cfg.test_envs:
        return []

    contexts: list[_TestRunContext] = []
    for env_key, env_cfg in cfg.test_envs.items():
        stores = [
            StreamStore(
                field_to_type=FIELD_TO_TYPE,
                env_name=env_key,
                env_number=env_idx,
            )
            for env_idx in range(int(env_cfg.num_envs))
        ]
        runner = RunnerTest(
            test_env=env_cfg,
            sequence_length=cfg.sequence_length,
            start_length=cfg.test_start_length,
            max_cache_length=cfg.test_max_cache_length,
            device=device,
            stores=stores,
        )
        contexts.append(_TestRunContext(env_key=env_key, runner=runner, env_cfg=env_cfg, stores=stores))

    env_names = ", ".join(cfg.test_envs.keys())
    step_counts = [ec.num_steps for ec in cfg.test_envs.values()]
    print(
        f"🧪 Test rollouts enabled every {cfg.test_interval} steps "
        f"({env_names}); step counts: {step_counts}."
    )
    return contexts


def _setup_q_networks(
    cfg: Config,
    device: torch.device,
) -> tuple[OfflineDQNTransformer | None, OfflineDQNTransformer | None, optim.Optimizer | None]:
    """Build or load Q-networks and optimizer. All None when train_interval=0.

    When polyak_tau==1.0 the target network is always identical to the online
    network, so q_target is set to None and the online network is used directly
    for bootstrapping — avoiding the cost of a second forward pass.
    """
    if cfg.train_interval <= 0:
        return None, None, None

    use_target = cfg.polyak_tau < 1.0

    if cfg.q_load_model_name:
        if "/" not in cfg.q_load_model_name:
            raise ValueError(
                f"loop.load_model_name must be a Hugging Face repo_id (e.g. 'user/repo'), "
                f"got: {cfg.q_load_model_name!r}"
            )
        print(f"📥 Loading pretrained Q-network from {cfg.q_load_model_name}...")
        q_online = OfflineDQNTransformer.from_pretrained(
            cfg.q_load_model_name,
            map_location=str(device),
        ).to(device)
        if use_target:
            q_target = OfflineDQNTransformer(
                head_dim=int(q_online.head_dim),
                base_model_id=str(q_online.base_model_id),
                num_hidden_layers=(
                    int(q_online.num_hidden_layers) if q_online.num_hidden_layers is not None else None
                ),
                field_to_type=q_online._field_to_type,
            ).to(device)
            q_target.load_state_dict(q_online.state_dict())
        else:
            q_target = None
        print("   Loaded.")
    else:
        action = "Loading pretrained backbone" if cfg.load_pretrained_backbone else "Building Q-network from scratch"
        print(f"📥 {action}...")
        q_online = OfflineDQNTransformer(
            head_dim=cfg.head_dim,
            base_model_id=cfg.base_model_id,
            num_hidden_layers=cfg.num_hidden_layers,
            field_to_type=FIELD_TO_TYPE,
            load_pretrained_backbone=cfg.load_pretrained_backbone,
        ).to(device)
        if use_target:
            q_target = OfflineDQNTransformer(
                head_dim=cfg.head_dim,
                base_model_id=cfg.base_model_id,
                num_hidden_layers=cfg.num_hidden_layers,
                field_to_type=FIELD_TO_TYPE,
            ).to(device)
            q_target.load_state_dict(q_online.state_dict())
        else:
            q_target = None
        print("   Built.")

    if cfg.torch_compile:
        torch.set_float32_matmul_precision("high")
        print("⚡ Compiling transformer backbone with torch.compile...")
        q_online.model = torch.compile(q_online.model)
        if q_target is not None:
            q_target.model = torch.compile(q_target.model)

    optimizer = optim.AdamW(
        q_online.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    return q_online, q_target, optimizer


def _size_category(num_rows: int) -> str:
    """Return Hugging Face size_categories tag for dataset card coherence."""
    if num_rows < 1_000:
        return "n<1K"
    if num_rows < 10_000:
        return "1K<n<10K"
    if num_rows < 100_000:
        return "10K<n<100K"
    if num_rows < 1_000_000:
        return "100K<n<1M"
    if num_rows < 10_000_000:
        return "1M<n<10M"
    if num_rows < 100_000_000:
        return "10M<n<100M"
    if num_rows < 1_000_000_000:
        return "100M<n<1B"
    return "n>1B"


def _save_test_rollout_dataset(test_contexts: list[_TestRunContext], repo_id: str) -> None:
    """Concatenate rollout data by split (train/eval) and push to Hugging Face."""
    by_split: dict[str, list[Dataset]] = {"train": [], "eval": []}
    for ctx in test_contexts:
        per_store_datasets = [store.to_dataset() for store in ctx.stores]
        ds = concatenate_datasets(per_store_datasets)
        split_key = (ctx.env_cfg.split or "train").lower()
        if split_key not in by_split:
            by_split[split_key] = []
        by_split[split_key].append(ds)

    splits_to_push = {k: concatenate_datasets(v) for k, v in by_split.items() if v}
    if not splits_to_push:
        return

    if len(splits_to_push) == 1:
        (name, single) = next(iter(splits_to_push.items()))
        dataset_dict = DatasetDict([(name, single)])
    else:
        dataset_dict = DatasetDict(list(splits_to_push.items()))

    total_rows = sum(len(ds) for ds in dataset_dict.values())
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=False,
        commit_message="New rollout data",
    )

    readme = f"""---
tags:
- reinforcement-learning
- tabular
size_categories:
- {_size_category(total_rows)}
---

# Rollout dataset

RL rollout data (action, observation, reward, done, step_id, env_name, env_number). Total rows: {total_rows:,}.
"""
    HfApi().upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card with coherent size metadata",
    )

    parts = ", ".join(f"{k}: {len(v)}" for k, v in splits_to_push.items())
    print(f"Saved: {repo_id} ({parts})")


def main(config_path: str = "") -> None:
    cfg = load_config(config_path)
    print(f"📋 Loaded config from: {config_path}")
    print(f"   Name: {cfg.name}")

    print("🔥 Authenticating with Weights & Biases...")
    setup_wandb()
    print("🤗 Authenticating with Hugging Face...")
    setup_huggingface()

    _set_seed(cfg.seed)
    device = _get_device()
    print(f"🔧 Using device: {device}")

    train_runner = _setup_train_runner(cfg)
    eval_runner = _setup_eval_runner(cfg)
    test_contexts = _setup_test_runners(cfg, device)
    q_online, q_target, q_optimizer = _setup_q_networks(cfg, device)

    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=dataclasses.asdict(cfg),
    )
    print(f"📝 W&B initialized (project={cfg.wandb_project}, run={wandb_run.name}).")

    max_steps = cfg.max_steps
    if train_runner is not None and cfg.max_epochs > 0:
        max_steps = min(max_steps, cfg.max_epochs * train_runner.total_batches)
    print(
        f"🚀 Starting run for {max_steps} steps "
        f"(train_interval={cfg.train_interval}, eval_interval={cfg.eval_interval}, "
        f"test_interval={cfg.test_interval}, max_epochs={cfg.max_epochs})."
    )

    step_idx = 0
    train_metrics: dict[str, float] = {"loss": 1.0}
    pbar = tqdm(total=max_steps, desc="Steps", unit="step")

    while step_idx < max_steps:
        status_tags: list[str] = []

        if (
            cfg.train_interval > 0
            and train_runner is not None
            and q_online is not None
            and q_optimizer is not None
            and step_idx % cfg.train_interval == 0
        ):
            status_tags.append("train")
            batch = train_runner.next_batch()
            train_metrics = train_runner.train(
                online_q_network=q_online,
                target_q_network=q_target,
                optimizer=q_optimizer,
                centering_factor=cfg.centering_factor,
                gamma=cfg.gamma,
                gamma_done=cfg.gamma_done,
                polyak_tau=cfg.polyak_tau,
                device=device,
                stream=batch,
            )
            wandb.log({
                "q_train/loss": train_metrics["loss"],
                "q_train/q_value_mean": train_metrics["q_value_mean"],
                "q_train/step": step_idx,
            })

        if (
            eval_runner is not None
            and q_online is not None
            and cfg.eval_interval > 0
            and step_idx % cfg.eval_interval == 0
        ):
            status_tags.append("eval")
            batch = eval_runner.next_batch()
            eval_metrics = eval_runner.eval(
                online_q_network=q_online,
                target_q_network=q_target,
                centering_factor=cfg.centering_factor,
                gamma=cfg.gamma,
                gamma_done=cfg.gamma_done,
                device=device,
                stream=batch,
            )
            loss_ratio = (
                eval_metrics["loss"] / train_metrics["loss"]
                if train_metrics.get("loss")
                else 0.0
            )
            wandb.log({
                "q_eval/loss": eval_metrics["loss"],
                "q_eval/loss_ratio": loss_ratio,
                "q_eval/q_value_mean": eval_metrics["q_value_mean"],
                "q_eval/step": step_idx,
            })

        if test_contexts and cfg.test_interval > 0 and step_idx % cfg.test_interval == 0:
            status_tags.append("test")
            for ctx in test_contexts:
                avg_reward = ctx.runner.run(
                    online_q_network=q_online,
                )
                wandb.log({
                    f"test/{ctx.env_key}/avg_reward": avg_reward,
                    f"test/{ctx.env_key}/step": step_idx,
                })

        pbar.set_postfix_str("|".join(status_tags) or "-")
        step_idx += 1
        pbar.update(1)

    pbar.close()

    if test_contexts and cfg.save_dataset and cfg.save_dataset_name:
        _save_test_rollout_dataset(test_contexts, cfg.save_dataset_name)

    if cfg.q_save_model_name and q_online is not None:
        if "/" not in cfg.q_save_model_name:
            raise ValueError(
                f"loop.save_model_name must be a Hugging Face repo_id (e.g. 'user/repo'), "
                f"got: {cfg.q_save_model_name!r}"
            )
        print(f"💾 Saving Q-network to {cfg.q_save_model_name}...")
        q_online.push_to_hub(cfg.q_save_model_name)
        print("   Saved.")

    wandb_run.finish()
    print("✅ Run complete.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
