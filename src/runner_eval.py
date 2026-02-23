from __future__ import annotations

import torch
import torch.nn.functional as F

from runner_batch import RunnerBatch
from tensordict import TensorDict
from models import OfflineDQNTransformer


class RunnerEval(RunnerBatch):
    """Batch runner for offline DQN evaluation (no grad, no weight update)."""

    @torch.no_grad()
    def eval(
        self,
        online_q_network: OfflineDQNTransformer,
        target_q_network: OfflineDQNTransformer | None,
        centering_factor: float,
        gamma: float,
        gamma_done: float,
        device: torch.device,
        stream: TensorDict,
    ) -> dict[str, float]:
        """Compute eval loss for one batch (no grad).

        Mirrors the train step exactly: the target network provides max Q(s')
        for the Bellman target, and the online network provides Q(s, a).
        When *target_q_network* is None (polyak_tau==1.0), the single online
        pass is reused for both roles. This makes the eval loss directly
        comparable to the train loss in both modes.
        """
        stream = stream.to(device)
        
        online_q_network.eval()
        q_online_values, q_values_mask, _ = online_q_network(stream=stream)
        actions, rewards, dones = online_q_network.extract_stream_tensors(stream)

        if target_q_network is not None:
            target_q_network.eval()
            q_target_values, _, _ = target_q_network(stream=stream)
        else:
            q_target_values = q_online_values.detach()

        next_q_values_valid = ~(q_target_values[..., 1:, 0].isnan())
        mask = q_values_mask[..., :-1] & next_q_values_valid

        if torch.all(~mask):
            raise ValueError("Not enough valid q values in data.")

        next_max_q = q_target_values[..., 1:, :].amax(dim=-1)
        # dones[..., 1:] at done_{s-1} gives the token at position done_{s-1}+1
        # (the action_s token), which via rightward backfill carries d_s — the
        # done flag for the transition whose Bellman target we are computing.
        d = dones[..., 1:]
        discount = gamma * (1.0 - d) + gamma_done * d
        target = rewards[..., :-1] + discount * next_max_q
        valid_target = target[mask]
        valid_target_normalized = valid_target - valid_target.mean() * centering_factor

        q_sa = q_online_values[..., :-1, :].gather(-1, actions[..., :-1].unsqueeze(-1)).squeeze(-1)
        valid_q_sa = q_sa[mask]
        q_loss = F.mse_loss(valid_q_sa, valid_target_normalized)

        return {
            "loss": q_loss.item(),
            "q_value_mean": valid_q_sa.mean().item(),
        }
