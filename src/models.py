import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from tensordict import TensorDict
from transformers import AutoConfig, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel


class FourierFeatures(nn.Module):
    """Fourier feature embedding for scalar inputs."""

    def __init__(self, num_frequencies: int = 8, base: float = 2.0):
        super().__init__()
        self.freqs: torch.Tensor
        exp_range = torch.arange(num_frequencies, dtype=torch.float32) - num_frequencies / 2
        freqs = base ** exp_range
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        angles = x * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def backfill_values(x: torch.Tensor, x_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Backfill invalid entries (x_mask=False) with the next valid value to the right along `dim`.
    Trailing invalids with no valid value to the right are left unchanged.
    """
    if x.numel() == 0:
        return x

    dim = dim % x.dim()
    moved = dim != x.dim() - 1
    if moved:
        x = x.movedim(dim, -1)
        x_mask = x_mask.movedim(dim, -1)

    L = x.size(-1)
    idx = torch.arange(L, device=x.device).view(*([1] * (x.dim() - 1)), L)

    sentinel = L
    next_idx = torch.flip(
        torch.cummin(
            torch.flip(torch.where(x_mask, idx, sentinel), dims=[-1]),
            dim=-1,
        )[0],
        dims=[-1],
    )

    has_next = next_idx != sentinel
    gathered = x.gather(-1, next_idx.clamp_max(L - 1).expand_as(x))
    out = torch.where(has_next, gathered, x)

    return out.movedim(-1, dim) if moved else out


class OfflineDQNTransformer(nn.Module, PyTorchModelHubMixin, library_name="DQN Transformer"):
    """Offline DQN with transformer over trajectory sequences.

    The transformer does not take discrete tokens in general: observations and rewards
    are continuous and embedded with Fourier features. Actions use a dedicated
    action embedding. Type embeddings cover action, obs, reward, done. Sequence per
    step: action, obs (L tokens), reward, done. The first action in a rollout is
    always random (no prior context); Q-values are emitted at done-token positions.
    """

    def __init__(
        self,
        head_dim: int,
        base_model_id: str,
        field_to_type: dict[str, int],
        num_hidden_layers: int | None = None,
        num_fourier_features: int | None = None,
        fourier_base: float = 1.1,
        load_pretrained_backbone: bool = False,
    ):
        super().__init__()
        self._field_to_type = field_to_type
        self._obs_type = self._field_to_type["observation"]
        self._reward_type = self._field_to_type["reward"]
        self._done_type = self._field_to_type["done"]
        self._action_type = self._field_to_type["action"]

        self.head_dim = head_dim
        self.base_model_id = base_model_id
        self.num_hidden_layers = num_hidden_layers

        config = AutoConfig.from_pretrained(base_model_id)
        if not isinstance(config, LlamaConfig):
            raise TypeError(f"Expected LlamaConfig, got {type(config).__name__}")
        if num_hidden_layers is not None:
            config.num_hidden_layers = int(num_hidden_layers)
        config.output_hidden_states = True
        config.use_cache = False

        hidden_size = config.hidden_size
        if hidden_size is None:
            raise ValueError("LlamaConfig.hidden_size must be set (cannot be None).")
        self.hidden_dim = int(hidden_size)

        self.num_fourier_features = num_fourier_features if num_fourier_features is not None else self.hidden_dim // 2
        if self.num_fourier_features % 2 != 0:
            raise ValueError("num_fourier_features must be even.")

        num_types = max(self._field_to_type.values()) + 1
        fourier_out_dim = 2 * self.num_fourier_features
        self.obs_fourier = FourierFeatures(num_frequencies=self.num_fourier_features, base=fourier_base)
        self.obs_fourier_proj = nn.Linear(fourier_out_dim, self.hidden_dim)
        self.reward_fourier = FourierFeatures(num_frequencies=self.num_fourier_features, base=fourier_base)
        self.reward_fourier_proj = nn.Linear(fourier_out_dim, self.hidden_dim)
        self.type_embed = nn.Embedding(num_types, self.hidden_dim)
        self.action_embed = nn.Embedding(head_dim, self.hidden_dim)
        self.done_embed = nn.Embedding(2, self.hidden_dim)
        if load_pretrained_backbone:
            self.model = LlamaModel.from_pretrained(base_model_id, config=config)
        else:
            self.model = LlamaModel(config)
        self.q_head = nn.Linear(self.hidden_dim, head_dim)

    def extract_stream_tensors(
        self,
        stream: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract rewards, actions, and dones aligned to the full token sequence.

        Pure data extraction — no learned parameters involved. Call once per
        batch rather than once per network when using a target network.

        Returns:
            rewards: [B, T_step] reward at each position (backfilled from reward tokens)
            actions: [B, T_step] action index at each position (backfilled from action tokens)
            dones:   [B, T_step] done flag at each position (backfilled from done tokens)
        """
        device = self.model.device
        stream = stream.to(device)

        types = stream["types"]
        values = stream["values"]

        batch_shape = types.shape[:-1]
        seq_len = types.shape[-1]

        action_mask = types == self._action_type
        reward_mask = types == self._reward_type
        done_mask = types == self._done_type

        action_values = values[action_mask].to(torch.int64)
        reward_values = values[reward_mask].to(torch.float32)
        done_values = values[done_mask].to(torch.int64)

        with torch.no_grad():
            actions = torch.zeros((*batch_shape, seq_len), dtype=torch.long, device=device)
            actions_mask = torch.zeros((*batch_shape, seq_len), dtype=torch.bool, device=device)
            actions[action_mask] = action_values
            actions_mask[action_mask] = True
            actions = backfill_values(actions, actions_mask, dim=-1)

            rewards = torch.full((*batch_shape, seq_len), torch.nan, dtype=torch.float32, device=device)
            rewards_mask = torch.zeros((*batch_shape, seq_len), dtype=torch.bool, device=device)
            rewards[reward_mask] = reward_values
            rewards_mask[reward_mask] = True
            rewards = backfill_values(rewards, rewards_mask, dim=-1)

            dones = torch.zeros((*batch_shape, seq_len), dtype=torch.float32, device=device)
            dones[done_mask] = done_values.float()
            dones = backfill_values(dones, done_mask, dim=-1)

        return actions, rewards, dones

    def forward(
        self,
        stream: TensorDict,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None]:
        """
        Args:
            stream: dict with
                - types: [B, T_step]
                - values_discrete: [B, T_step]
                - values_continuous: [B, T_step]
            past_key_values: cached key/value states from a previous forward pass.
            use_cache: if True, return updated past_key_values for incremental decoding.
        Returns:
            q_values:     [B, T_step, head_dim] Q values (NaN outside done positions, backfilled)
            q_values_mask:[B, T_step] True at positions that have a valid (done-token) Q value
            past_key_values: updated KV cache, or None when use_cache=False
        """
        device = self.model.device
        stream = stream.to(device)

        types = stream["types"]
        values = stream["values"]

        batch_shape = types.shape[:-1]
        seq_len = types.shape[-1]

        action_mask = types == self._action_type
        obs_mask = types == self._obs_type
        reward_mask = types == self._reward_type
        done_mask = types == self._done_type

        action_type_embed = self.type_embed.weight[self._action_type].view(1, -1)
        obs_type_embed = self.type_embed.weight[self._obs_type].view(1, -1)
        reward_type_embed = self.type_embed.weight[self._reward_type].view(1, -1)
        done_type_embed = self.type_embed.weight[self._done_type].view(1, -1)

        action_values = values[action_mask].to(torch.int64)
        obs_values = values[obs_mask].to(torch.float32)
        reward_values = values[reward_mask].to(torch.float32)
        done_values = values[done_mask].to(torch.int64)

        action_embeddings = self.action_embed(action_values) + action_type_embed
        obs_fourier = self.obs_fourier_proj(self.obs_fourier(obs_values)) + obs_type_embed
        reward_fourier = self.reward_fourier_proj(self.reward_fourier(reward_values)) + reward_type_embed
        done_embeddings = self.done_embed(done_values) + done_type_embed

        embeddings = torch.full((*batch_shape, seq_len, self.hidden_dim), torch.nan, device=device)
        embeddings[action_mask] = action_embeddings
        embeddings[obs_mask] = obs_fourier
        embeddings[reward_mask] = reward_fourier
        embeddings[done_mask] = done_embeddings

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden = outputs.last_hidden_state
        new_past_key_values = outputs.past_key_values if use_cache else None

        valid_hidden = hidden[done_mask]
        valid_q_values = self.q_head(valid_hidden)

        q_values = torch.full((*batch_shape, seq_len, self.head_dim), torch.nan, dtype=torch.float32, device=device)
        q_values_mask = done_mask
        q_values[done_mask] = valid_q_values
        q_values = backfill_values(q_values, q_values_mask.unsqueeze(-1), dim=-2)

        return q_values, q_values_mask, new_past_key_values

def polyak_update(target_net: nn.Module, online_net: nn.Module, tau: float) -> None:
    """
    Polyak averaging (soft update) of target parameters.

    Updates target network parameters using exponential moving average:
        θ_target ← τ θ_online + (1 - τ) θ_target

    Args:
        target_net: Target network to update
        online_net: Online network providing source parameters
        tau: Mixing coefficient (0 < tau <= 1). Smaller values mean slower updates.
    """
    with torch.no_grad():
        for p_t, p in zip(target_net.parameters(), online_net.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)
