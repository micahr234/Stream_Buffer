from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import EnvConfig
from stream_store import StreamStore
from tensordict import TensorDict
from environment import NSVectorEnvRunner
from models import OfflineDQNTransformer


class RunnerTest:
    """Persistent test rollout runner that continues env state across intervals."""

    def __init__(
        self,
        test_env: EnvConfig,
        sequence_length: int,
        start_length: int,
        max_cache_length: int,
        device: torch.device,
        stores: list[StreamStore],
    ):
        if test_env.non_stationary_params is None:
            raise ValueError("test_env.non_stationary_params is required.")
        if test_env.num_steps is None or test_env.num_steps <= 0:
            raise ValueError("test_env.num_steps must be > 0.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0.")
        if max_cache_length < 0:
            raise ValueError("max_cache_length must be >= 0.")

        self.num_steps = test_env.num_steps
        self.sequence_length = sequence_length
        self.start_length = start_length if start_length > 0 else sequence_length
        self.max_cache_length = max_cache_length
        self.use_cache = max_cache_length > 0
        self.device = device
        self.num_envs = int(test_env.num_envs)
        self.stores = stores
        self._step: int = 0  # monotonically increasing step index passed to append
        self.env_runner = NSVectorEnvRunner(
            env_id=test_env.env_id,
            non_stationary_params=test_env.non_stationary_params,
            seed=test_env.seed,
            num_envs=self.num_envs,
            max_steps_per_episode=test_env.max_episode_steps,
            env_kwargs=test_env.kwargs,
            render=test_env.render,
        )
        self.action_dim = self.env_runner.action_dim
        if len(stores) != self.num_envs:
            raise ValueError(
                f"Expected one streamstore per env ({self.num_envs}), got {len(stores)}."
            )

    def _save_experience(self, experience: dict[str, np.ndarray], step: int) -> None:
        field_names = list(experience.keys())
        field_values = list(experience.values())
        for env_idx in range(self.num_envs):
            tmp_field_values = [v[env_idx, ...] for v in field_values]
            self.stores[env_idx].append(
                field_names=field_names,
                field_values=tmp_field_values,
                step=step,
            )

    def _get_context(self, cached_positions: torch.Tensor | None = None) -> TensorDict:
        """Return batched token streams for all envs.

        When *cached_positions* is ``None`` (prefill), returns up to
        ``sequence_length`` most-recent tokens.  Otherwise returns only the
        tokens appended after the given positions (incremental decode).
        """
        per_env_streams: list[TensorDict] = []
        for env_idx in range(self.num_envs):
            end_idx = len(self.stores[env_idx])
            if cached_positions is None:
                start_idx = max(0, end_idx - self.start_length)
            else:
                start_idx = cached_positions[env_idx].item()
            raw = self.stores[env_idx][torch.arange(start_idx, end_idx)]
            stream = TensorDict(
                {k: torch.from_numpy(v) for k, v in raw.items()},
                batch_size=raw["types"].shape,
            )
            per_env_streams.append(stream)
        return TensorDict.maybe_dense_stack(per_env_streams, dim=0).to(device=self.device)

    @staticmethod
    def _cache_seq_len(past_key_values: Any) -> int:
        """Return the current number of tokens stored in the KV cache."""
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[2]

    def run(
        self,
        online_q_network: OfflineDQNTransformer | None = None,
    ) -> float:
        """Run num_steps steps with optional KV-cached incremental decoding.

        If *online_q_network* is None, actions are sampled uniformly at random.
        """
        if online_q_network is not None:
            online_q_network.eval()

        with torch.no_grad():
            rewards = []
            past_key_values: Any = None
            cached_positions: torch.Tensor | None = None

            for _ in range(self.num_steps):

                next_actions = None
                if not self.env_runner.init:
                    if online_q_network is not None:
                        if self.use_cache:
                            store_lens = torch.tensor([len(s) for s in self.stores], dtype=torch.long)
                            prev = cached_positions if cached_positions is not None else torch.zeros_like(store_lens)
                            max_new_tokens = (store_lens - prev).max().item()
                            if self._cache_seq_len(past_key_values) + max_new_tokens > self.max_cache_length:
                                past_key_values = None
                                cached_positions = None

                        streams = self._get_context(cached_positions)
                        q_values, q_values_mask, past_key_values = online_q_network(
                            stream=streams,
                            past_key_values=past_key_values,
                            use_cache=self.use_cache,
                        )
                        if self.use_cache:
                            cached_positions = torch.tensor([len(store) for store in self.stores], dtype=torch.long)

                        last_valid_q_mask = q_values_mask[:, -1]
                        if torch.any(~last_valid_q_mask):
                            raise ValueError("Invalid q values in last step. Input stream is not valid.")
                        next_actions = q_values[:, -1, :self.action_dim].argmax(dim=-1).cpu().numpy()

                self.env_runner.step(next_actions)

                self._save_experience(
                    {"action": self.env_runner.actions,
                    "observation": self.env_runner.obs,
                    "reward": self.env_runner.rewards,
                    "done": self.env_runner.dones},
                    step=self._step,
                )
                rewards.append(torch.from_numpy(self.env_runner.rewards))
                self._step += 1

        return torch.stack(rewards, dim=-1).mean().item()
