"""Authentication helpers for third-party services (Hugging Face, Weights & Biases)."""

import os
from huggingface_hub import login as hf_login
import wandb


def setup_huggingface() -> None:
    """Authenticate with Hugging Face using env vars.

    Expects one of: HF_TOKEN, RUNPOD_HF_TOKEN
    """
    hf_token_envs = ["HF_TOKEN", "RUNPOD_HF_TOKEN"]
    hf_token = None
    for env_var in hf_token_envs:
        hf_token = os.getenv(env_var)
        if hf_token:
            break
    if not hf_token:
        raise ValueError(f"None of the following environment variables are set: {hf_token_envs}")

    os.environ["HF_TOKEN"] = hf_token
    hf_login(token=hf_token)


def setup_wandb() -> None:
    """Authenticate with Weights & Biases using env vars.

    Expects one of: WANDB_TOKEN, RUNPOD_WANDB_TOKEN
    """
    wandb_token_envs = ["WANDB_TOKEN", "RUNPOD_WANDB_TOKEN"]
    wandb_token = None
    for env_var in wandb_token_envs:
        wandb_token = os.getenv(env_var)
        if wandb_token:
            break
    if not wandb_token:
        raise ValueError(f"None of the following environment variables are set: {wandb_token_envs}")

    wandb.login(key=wandb_token, verify=True)


