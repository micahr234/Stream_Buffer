"""Tests for models.py: FourierFeatures, backfill_values, OfflineDQNTransformer.

OfflineDQNTransformer tests build a tiny LlamaConfig in-process so no
HuggingFace download is required.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from definitions import FIELD_TO_TYPE, TokenType
from models import FourierFeatures, OfflineDQNTransformer, backfill_values, polyak_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_llama_model_id(tmp_path) -> str:
    """Save a tiny LlamaConfig to a temp directory and return the path.

    This avoids any HuggingFace Hub network calls.
    """
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        vocab_size=1000,
    )
    model_dir = str(tmp_path / "tiny_llama")
    cfg.save_pretrained(model_dir)
    return model_dir


def _make_transformer(tmp_path, head_dim: int = 4) -> OfflineDQNTransformer:
    model_id = _tiny_llama_model_id(tmp_path)
    return OfflineDQNTransformer(
        head_dim=head_dim,
        base_model_id=model_id,
        field_to_type=FIELD_TO_TYPE,
    )


def _make_stream(
    batch: int,
    num_steps: int,
    obs_dim: int,
    head_dim: int,
    *,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, torch.Tensor], int]:
    """Build a synthetic token stream TensorDict-like dict.

    Token layout per step: action + obs (obs_dim tokens) + reward + done.
    Returns (stream_dict, seq_len).
    """
    from tensordict import TensorDict

    tokens_per_step = obs_dim + 3  # action + obs + reward + done
    seq_len = num_steps * tokens_per_step

    types = torch.zeros(batch, seq_len, dtype=torch.long)
    values = torch.zeros(batch, seq_len, dtype=torch.float64)

    for step in range(num_steps):
        base = step * tokens_per_step
        # action (first in step; random on step 0, model-based thereafter)
        types[:, base] = int(TokenType.ACTION)
        values[:, base] = torch.randint(0, head_dim, (batch,)).float()
        # obs tokens
        for d in range(obs_dim):
            types[:, base + 1 + d] = int(TokenType.OBS)
            values[:, base + 1 + d] = torch.randn(batch)
        # reward
        types[:, base + 1 + obs_dim] = int(TokenType.REWARD)
        values[:, base + 1 + obs_dim] = torch.randn(batch)
        # done (0 except last step)
        types[:, base + 1 + obs_dim + 1] = int(TokenType.DONE)
        values[:, base + 1 + obs_dim + 1] = float(step == num_steps - 1)

    stream = TensorDict(
        {"types": types, "values": values},
        batch_size=(batch, seq_len),
    )
    return stream, seq_len


# ---------------------------------------------------------------------------
# 1. FourierFeatures
# ---------------------------------------------------------------------------


def test_fourier_features_output_shape():
    num_freq = 8
    ff = FourierFeatures(num_frequencies=num_freq)
    x = torch.randn(3, 5)
    out = ff(x)
    assert out.shape == (3, 5, 2 * num_freq), f"Expected (3, 5, 16), got {out.shape}"


def test_fourier_features_scalar_input():
    ff = FourierFeatures(num_frequencies=4)
    x = torch.tensor([1.0, 2.0])
    out = ff(x)
    assert out.shape == (2, 8)


def test_fourier_features_sin_cos_split():
    """First half should be sin, second half cos — check a known value."""
    num_freq = 1
    base = 2.0
    ff = FourierFeatures(num_frequencies=num_freq, base=base)
    # freq = base^(0 - 0.5) = 2^(-0.5)
    freq = base ** (torch.tensor([0]) - num_freq / 2)
    x = torch.tensor([1.0])
    out = ff(x)  # shape (1, 2)
    expected_sin = math.sin(1.0 * freq.item())
    expected_cos = math.cos(1.0 * freq.item())
    assert abs(out[0, 0].item() - expected_sin) < 1e-5
    assert abs(out[0, 1].item() - expected_cos) < 1e-5


def test_fourier_features_different_bases():
    ff1 = FourierFeatures(num_frequencies=4, base=1.1)
    ff2 = FourierFeatures(num_frequencies=4, base=2.0)
    x = torch.ones(2)
    assert not torch.allclose(ff1(x), ff2(x)), "Different bases should produce different outputs"


def test_fourier_features_no_grad_through_buffer():
    """freqs buffer should be non-trainable (not in parameters)."""
    ff = FourierFeatures(num_frequencies=4)
    param_names = [n for n, _ in ff.named_parameters()]
    assert "freqs" not in param_names


# ---------------------------------------------------------------------------
# 2. backfill_values
# ---------------------------------------------------------------------------


def test_backfill_values_basic():
    """Invalid entries (mask=False) get the next valid value to the right."""
    x = torch.tensor([0.0, 10.0, 0.0, 20.0, 0.0])
    mask = torch.tensor([False, True, False, True, False])
    out = backfill_values(x, mask)
    # pos 0 → next valid right = pos 1 → 10
    # pos 1 → valid → 10
    # pos 2 → next valid right = pos 3 → 20
    # pos 3 → valid → 20
    # pos 4 → no valid right → unchanged (0)
    expected = torch.tensor([10.0, 10.0, 20.0, 20.0, 0.0])
    assert torch.allclose(out, expected), f"Got {out}"


def test_backfill_values_all_valid():
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.ones(3, dtype=torch.bool)
    out = backfill_values(x, mask)
    assert torch.allclose(out, x)


def test_backfill_values_all_invalid():
    x = torch.tensor([7.0, 8.0, 9.0])
    mask = torch.zeros(3, dtype=torch.bool)
    out = backfill_values(x, mask)
    # All invalid, no valid right → all unchanged
    assert torch.allclose(out, x)


def test_backfill_values_empty():
    x = torch.tensor([], dtype=torch.float32)
    mask = torch.tensor([], dtype=torch.bool)
    out = backfill_values(x, mask)
    assert out.numel() == 0


def test_backfill_values_batched():
    """2-D input: each row is filled independently."""
    x = torch.tensor([
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 9.0],
    ])
    mask = torch.tensor([
        [False, True, False],
        [False, False, True],
    ])
    out = backfill_values(x, mask)
    # Row 0: pos 0 → 5; pos 2 → no valid right → 0 unchanged
    assert out[0, 0].item() == 5.0
    assert out[0, 1].item() == 5.0
    assert out[0, 2].item() == 0.0
    # Row 1: pos 0 → 9; pos 1 → 9; pos 2 → valid
    assert out[1, 0].item() == 9.0
    assert out[1, 1].item() == 9.0
    assert out[1, 2].item() == 9.0


def test_backfill_values_non_default_dim():
    """Fill along dim=0 (columns)."""
    x = torch.tensor([
        [0.0, 0.0],
        [3.0, 4.0],
        [0.0, 0.0],
    ])
    mask = torch.tensor([
        [False, False],
        [True, True],
        [False, False],
    ])
    out = backfill_values(x, mask, dim=0)
    # Column 0: row 0 → next valid below = row 1 → 3; row 2 → no valid below → 0
    assert out[0, 0].item() == 3.0
    assert out[2, 0].item() == 0.0


# ---------------------------------------------------------------------------
# 3. OfflineDQNTransformer
# ---------------------------------------------------------------------------


def test_offline_dqn_forward_output_shapes(tmp_path):
    head_dim = 4
    batch = 2
    num_steps = 3
    obs_dim = 2

    model = _make_transformer(tmp_path, head_dim=head_dim)
    stream, seq_len = _make_stream(batch, num_steps, obs_dim, head_dim)

    q_values, q_values_mask, pkv = model(stream)
    actions, rewards, dones = model.extract_stream_tensors(stream)

    assert q_values.shape == (batch, seq_len, head_dim), f"q_values shape wrong: {q_values.shape}"
    assert rewards.shape == (batch, seq_len), f"rewards shape wrong: {rewards.shape}"
    assert actions.shape == (batch, seq_len), f"actions shape wrong: {actions.shape}"
    assert dones.shape == (batch, seq_len), f"dones shape wrong: {dones.shape}"
    assert q_values_mask.shape == (batch, seq_len), f"q_values_mask shape wrong: {q_values_mask.shape}"
    assert pkv is None  # use_cache=False by default


def test_offline_dqn_q_values_valid_only_at_done_positions(tmp_path):
    """Q-values should be non-NaN only at done token positions (before backfill)."""
    head_dim = 4
    batch = 1
    num_steps = 4
    obs_dim = 3

    model = _make_transformer(tmp_path, head_dim=head_dim)

    from tensordict import TensorDict

    tokens_per_step = obs_dim + 3
    seq_len = num_steps * tokens_per_step

    types = torch.zeros(batch, seq_len, dtype=torch.long)
    for step in range(num_steps):
        base = step * tokens_per_step
        # New ordering: action, obs..., reward, done
        types[0, base] = int(TokenType.ACTION)
        for d in range(obs_dim):
            types[0, base + 1 + d] = int(TokenType.OBS)
        types[0, base + 1 + obs_dim] = int(TokenType.REWARD)
        types[0, base + 1 + obs_dim + 1] = int(TokenType.DONE)

    values = torch.zeros(batch, seq_len, dtype=torch.float64)
    stream = TensorDict({"types": types, "values": values}, batch_size=(batch, seq_len))

    q_values, q_values_mask, _ = model(stream)

    # q_values_mask should be True at done positions
    done_positions = (types[0] == int(TokenType.DONE))
    assert q_values_mask[0, done_positions].all(), "Q-values mask must be True at all done positions"

    # Before backfill, Q-values at non-done positions would have been NaN,
    # but after backfill they carry the next valid Q — so we just check the
    # mask is False at obs/reward/action positions.
    non_done = ~done_positions
    assert not q_values_mask[0, non_done].any(), "Q-values mask must be False at non-done positions"


def test_offline_dqn_no_cache_by_default(tmp_path):
    model = _make_transformer(tmp_path)
    stream, _ = _make_stream(1, 2, 2, 4)
    _, _, pkv = model(stream)
    assert pkv is None


def test_offline_dqn_with_cache_returns_past_key_values(tmp_path):
    model = _make_transformer(tmp_path)
    stream, _ = _make_stream(1, 2, 2, 4)
    _, _, pkv = model(stream, use_cache=True)
    assert pkv is not None


def test_polyak_update_tau_one_copies_weights():
    """tau=1 should make target exactly equal to online."""
    target = torch.nn.Linear(4, 4)
    online = torch.nn.Linear(4, 4)
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(0.0)
        for p in online.parameters():
            p.fill_(1.0)
    polyak_update(target, online, tau=1.0)
    for p_t, p in zip(target.parameters(), online.parameters()):
        assert torch.allclose(p_t, p), "tau=1 should copy online → target exactly"


def test_polyak_update_tau_zero_leaves_target_unchanged():
    """tau=0 should leave target completely unchanged."""
    target = torch.nn.Linear(4, 4)
    online = torch.nn.Linear(4, 4)
    target_before = [p.clone() for p in target.parameters()]
    with torch.no_grad():
        for p in online.parameters():
            p.fill_(99.0)
    polyak_update(target, online, tau=0.0)
    for p_t, p_before in zip(target.parameters(), target_before):
        assert torch.allclose(p_t, p_before), "tau=0 should not change target"


def test_polyak_update_interpolates():
    """tau=0.5 should produce the midpoint between target and online weights."""
    target = torch.nn.Linear(2, 2, bias=False)
    online = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(0.0)
        for p in online.parameters():
            p.fill_(2.0)
    polyak_update(target, online, tau=0.5)
    for p_t in target.parameters():
        assert torch.allclose(p_t, torch.full_like(p_t, 1.0)), "tau=0.5 should yield midpoint"


def test_odd_num_fourier_features_raises(tmp_path):
    """Odd num_fourier_features must be rejected (requires a valid base_model_id)."""
    model_id = _tiny_llama_model_id(tmp_path)
    with pytest.raises(ValueError, match="even"):
        OfflineDQNTransformer(
            head_dim=4,
            base_model_id=model_id,
            field_to_type=FIELD_TO_TYPE,
            num_fourier_features=3,  # odd → should raise
        )
