from __future__ import annotations

import torch
import torch.nn.functional as F

from runner_batch import RunnerBatch
from tensordict import TensorDict
from models import OfflineDQNTransformer, polyak_update


class RunnerTrain(RunnerBatch):
    """Batch runner for offline DQN training.

    Windows are shuffled at the start of each epoch so consecutive batches
    contain uncorrelated transitions — essential for stable offline RL training.
    """

    @torch.no_grad()
    def train(
        self,
        online_q_network: OfflineDQNTransformer,
        target_q_network: OfflineDQNTransformer | None,
        optimizer: torch.optim.Optimizer,
        centering_factor: float,
        gamma: float,
        gamma_done: float,
        polyak_tau: float,
        device: torch.device,
        stream: TensorDict,
    ) -> dict[str, float]:
        """Run one offline DQN optimization step.

        When *target_q_network* is None (polyak_tau==1.0), the online network is
        used directly for bootstrapping with a single forward pass.
        """
        stream = stream.to(device)
        
        online_q_network.train()
        with torch.enable_grad():
            q_online_values, q_values_mask, _ = online_q_network(stream=stream)
        actions, rewards, dones = online_q_network.extract_stream_tensors(stream)
        q_values_mask = q_values_mask.detach()

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

        with torch.enable_grad():
            q_sa = q_online_values[..., :-1, :].gather(-1, actions[..., :-1].unsqueeze(-1)).squeeze(-1)
            valid_q_sa = q_sa[mask]
            q_loss = F.mse_loss(valid_q_sa, valid_target_normalized)
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

        if target_q_network is not None:
            polyak_update(target_q_network, online_q_network, polyak_tau)

        return {
            "loss": q_loss.item(),
            "q_value_mean": valid_q_sa.mean().item(),
        }
