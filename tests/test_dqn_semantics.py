"""DQN token-order and semantic correctness tests.

The OfflineDQNTransformer consumes token streams with a fixed per-step layout:

    [action, obs_0, ..., obs_{D-1}, reward, done]

Per-step, there are D + 3 tokens (tokens_per_step = D + 3).  The first action
in a rollout is always random (no prior context).

Backfill propagates values **rightward** (toward higher indices).  That means:

  • action position → carries the current step's reward (REWARD is to its right)
  • obs positions   → carry the current step's reward (REWARD is to their right)
  • reward position → valid; carries the current step's reward
  • done position   → carries the **next** step's reward (REWARD_{s+1} is to its right)
  • action position → carries the current step's done flag (DONE is to its right)
  • obs/reward/done positions → carry the **next** step's action (ACTION_{s+1} is to their right)
  • done position   → carries the current step's done flag (valid; no backfill needed)

DQN Bellman target is computed at done_{s-1} (the DONE token of step s-1):

    target[done_{s-1}] = rewards[done_{s-1}] + discount * max_a Q(done_s, a)

where:
  • rewards[done_{s-1}]   = r_s   (backfilled from REWARD token of step s, which is to the right)
  • discount              = γ·(1-d_s) + γ_done·d_s
                            d_s comes from dones[done_{s-1}+1] (position of action_s,
                            which via backfill carries d_s from DONE_s to its right)
  • max_a Q(done_s, a)    = q_values[done_{s-1}+1]  (backfilled from next DONE token)
  • actions[done_{s-1}]   = a_s  (backfilled from ACTION_s, the next action to the right)

These tests verify all of the above properties hold in the actual model output.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from definitions import FIELD_TO_TYPE, TokenType
from models import OfflineDQNTransformer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

OBS_DIM = 1   # keep small for analytical tractability
HEAD_DIM = 4


def _tiny_llama_model_id(tmp_path) -> str:
    """Write a tiny LlamaConfig to tmp_path; avoids any HuggingFace network call."""
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


def _make_model(tmp_path) -> OfflineDQNTransformer:
    return OfflineDQNTransformer(
        head_dim=HEAD_DIM,
        base_model_id=_tiny_llama_model_id(tmp_path),
        field_to_type=FIELD_TO_TYPE,
    )


def _make_stream(
    num_steps: int,
    *,
    obs_dim: int = OBS_DIM,
    head_dim: int = HEAD_DIM,
    rewards: list[float] | None = None,
    dones: list[bool] | None = None,
    actions: list[int] | None = None,
):
    """Build a batch=1 TensorDict stream with fully controlled per-step values.

    Per-step layout: [action, obs_0, ..., obs_{D-1}, reward, done]
    Returns (stream, tokens_per_step, seq_len).
    """
    from tensordict import TensorDict

    if rewards is None:
        rewards = [float(s) for s in range(num_steps)]
    if dones is None:
        dones = [s == num_steps - 1 for s in range(num_steps)]
    if actions is None:
        actions = [s % head_dim for s in range(num_steps)]

    tps = obs_dim + 3  # tokens per step: action + obs... + reward + done
    seq_len = num_steps * tps

    types = torch.zeros(1, seq_len, dtype=torch.long)
    values = torch.zeros(1, seq_len, dtype=torch.float64)

    for s in range(num_steps):
        base = s * tps
        types[0, base] = int(TokenType.ACTION)
        values[0, base] = float(actions[s])
        for d in range(obs_dim):
            types[0, base + 1 + d] = int(TokenType.OBS)
            values[0, base + 1 + d] = float(s * obs_dim + d)
        types[0, base + 1 + obs_dim] = int(TokenType.REWARD)
        values[0, base + 1 + obs_dim] = rewards[s]
        types[0, base + 1 + obs_dim + 1] = int(TokenType.DONE)
        values[0, base + 1 + obs_dim + 1] = float(dones[s])

    stream = TensorDict({"types": types, "values": values}, batch_size=(1, seq_len))
    return stream, tps, seq_len


# ---------------------------------------------------------------------------
# 1. Token type parsing: correct values appear at typed positions
# ---------------------------------------------------------------------------


def test_reward_values_at_reward_positions(tmp_path):
    """rewards output at REWARD token positions must equal the input reward values.

    Verifies that the model correctly identifies REWARD tokens by position and
    does not confuse them with action, obs, or done tokens.
    """
    reward_vals = [1.0, 2.5, -0.5, 3.0]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(4, rewards=reward_vals)

    with torch.no_grad():
        _, rewards_out, _ = model.extract_stream_tensors(stream)

    for s, expected in enumerate(reward_vals):
        pos = s * tps + 1 + OBS_DIM  # REWARD is at offset 1+obs_dim (after action + obs tokens)
        got = rewards_out[0, pos].item()
        assert math.isclose(got, expected, rel_tol=1e-4), (
            f"Step {s}: rewards[{pos}] = {got!r}, expected {expected!r}"
        )


def test_done_values_at_done_positions(tmp_path):
    """dones output at DONE token positions must match the input done flags.

    Verifies that DONE tokens (last token of each step) are correctly
    distinguished from REWARD tokens (second-to-last token of each step).
    """
    done_flags = [False, True, False, True]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(4, dones=done_flags)

    with torch.no_grad():
        _, _, dones_out = model.extract_stream_tensors(stream)

    for s, expected in enumerate(done_flags):
        pos = s * tps + 1 + OBS_DIM + 1  # DONE is last: offset 1+obs_dim+1
        got = bool(dones_out[0, pos].item())
        assert got == expected, (
            f"Step {s}: dones[{pos}] = {got!r}, expected {expected!r}"
        )


def test_action_values_at_action_positions(tmp_path):
    """actions output at ACTION token positions must equal the input actions.

    Verifies that ACTION tokens (the first token of each step) are correctly
    identified, not mistaken for obs, reward, or done tokens.
    """
    action_vals = [0, 2, 1, 3]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(4, actions=action_vals)

    with torch.no_grad():
        actions_out, _, _ = model.extract_stream_tensors(stream)

    for s, expected in enumerate(action_vals):
        pos = s * tps  # ACTION is the first token in each step
        got = int(actions_out[0, pos].item())
        assert got == expected, (
            f"Step {s}: actions[{pos}] = {got!r}, expected {expected!r}"
        )


def test_q_values_mask_exactly_at_done_positions(tmp_path):
    """q_values_mask must be True at DONE token positions and False everywhere else.

    Only done tokens produce Q-values; obs, reward, and action tokens do not.
    This is the mask used to select valid Bellman target positions.
    """
    num_steps = 3
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(num_steps)
    types = stream["types"][0]

    with torch.no_grad():
        _, q_values_mask, _ = model(stream)

    done_positions = types == int(TokenType.DONE)
    assert q_values_mask[0, done_positions].all(), (
        "q_values_mask must be True at all DONE positions"
    )
    assert not q_values_mask[0, ~done_positions].any(), (
        "q_values_mask must be False at all non-DONE positions"
    )


# ---------------------------------------------------------------------------
# 2. Backfill semantics: values propagate rightward to adjacent tokens
# ---------------------------------------------------------------------------


def test_obs_tokens_carry_current_step_reward(tmp_path):
    """OBS tokens precede REWARD within each step, so rightward backfill gives
    them the reward of the *same* step.
    """
    reward_vals = [1.0, 2.0, 3.0]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, rewards=reward_vals)

    with torch.no_grad():
        _, rewards_out, _ = model.extract_stream_tensors(stream)

    for s, expected in enumerate(reward_vals):
        for d in range(OBS_DIM):
            pos = s * tps + 1 + d  # obs starts at offset 1 (after action)
            got = rewards_out[0, pos].item()
            assert math.isclose(got, expected, rel_tol=1e-4), (
                f"Step {s} obs[{d}] at pos {pos}: rewards = {got!r}, "
                f"expected current-step reward {expected!r}"
            )


def test_done_token_carries_next_step_reward(tmp_path):
    """DONE token comes *after* REWARD within a step.

    Rightward backfill gives the DONE token the REWARD of the **next** step.
    This is the reward used in the DQN Bellman target at done_{s-1}:
    Q(context_{s-1}, a_s) ← r_s + γ * max_a Q(context_s, a).
    Note: rewards[done_{s-1}] = r_s (the reward immediately to the right).
    """
    reward_vals = [1.0, 2.0, 3.0]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, rewards=reward_vals)

    with torch.no_grad():
        _, rewards_out, _ = model.extract_stream_tensors(stream)

    for s in range(len(reward_vals) - 1):  # last step has no next reward
        done_offset = 1 + OBS_DIM + 1  # done is last token of step
        pos = s * tps + done_offset
        expected = reward_vals[s + 1]
        got = rewards_out[0, pos].item()
        assert math.isclose(got, expected, rel_tol=1e-4), (
            f"Step {s} done token at pos {pos}: rewards = {got!r}, "
            f"expected next-step reward {expected!r}"
        )


def test_action_token_carries_current_step_reward(tmp_path):
    """ACTION token comes *before* obs/reward/done within a step.

    Rightward backfill gives the ACTION token the REWARD of the *same* step
    (reward_s is the first reward to the right of action_s).
    """
    reward_vals = [1.0, 2.0, 3.0]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, rewards=reward_vals)

    with torch.no_grad():
        _, rewards_out, _ = model.extract_stream_tensors(stream)

    for s, expected in enumerate(reward_vals):
        pos = s * tps  # action is the first token of each step
        got = rewards_out[0, pos].item()
        assert math.isclose(got, expected, rel_tol=1e-4), (
            f"Step {s} action token at pos {pos}: rewards = {got!r}, "
            f"expected current-step reward {expected!r}"
        )


def test_obs_reward_done_tokens_carry_next_step_action(tmp_path):
    """ACTION is the *first* token in each step.  Rightward backfill propagates
    the next ACTION (a_{s+1}) to all later positions within step s (obs, reward,
    done all see a_{s+1} as the next valid action to their right).
    The action token itself carries the current-step action (a_s).
    """
    action_vals = [0, 2, 1]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, actions=action_vals)

    with torch.no_grad():
        actions_out, _, _ = model.extract_stream_tensors(stream)

    # Action token itself carries current-step action.
    for s, expected in enumerate(action_vals):
        pos = s * tps
        got = int(actions_out[0, pos].item())
        assert got == expected, (
            f"Step {s} action pos {pos}: actions = {got!r}, expected {expected!r}"
        )

    # obs, reward, done tokens of step s carry a_{s+1} (next step's action).
    for s in range(len(action_vals) - 1):
        expected_next = action_vals[s + 1]
        for offset in range(1, tps):  # skip action at offset 0
            pos = s * tps + offset
            got = int(actions_out[0, pos].item())
            assert got == expected_next, (
                f"Step {s} offset {offset} at pos {pos}: actions = {got!r}, "
                f"expected next-step action {expected_next!r}"
            )


def test_actions_at_done_positions_are_next_step_actions(tmp_path):
    """At the DONE position of step s, the backfilled actions carry a_{s+1}.

    In the new per-step layout [action, obs..., reward, done], DONE is the last
    token of step s, so the next ACTION to its right is the first token of step
    s+1.  Rightward backfill therefore gives the done position the action of the
    *next* step.
    """
    action_vals = [1, 3, 0]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, actions=action_vals)

    with torch.no_grad():
        actions_out, _, _ = model.extract_stream_tensors(stream)

    for s in range(len(action_vals) - 1):  # last step has no next action
        done_pos = s * tps + 1 + OBS_DIM + 1  # done is last token of step s
        expected_next = action_vals[s + 1]
        got = int(actions_out[0, done_pos].item())
        assert got == expected_next, (
            f"Step {s}: actions at done pos {done_pos} = {got!r}, "
            f"expected next-step action {expected_next!r}"
        )


def test_actions_at_prev_done_carry_current_step_action(tmp_path):
    """At done_{s-1} (the Bellman-loss anchor), backfilled actions carry a_s.

    The training loss at done_{s-1} gathers Q(context_{s-1}, a_s).  This test
    verifies the gather index is correct: actions[done_{s-1}] = a_s because
    action_s is the first token to the right of done_{s-1}.
    """
    action_vals = [1, 3, 0]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, actions=action_vals)

    with torch.no_grad():
        actions_out, _, _ = model.extract_stream_tensors(stream)

    for s in range(1, len(action_vals)):
        prev_done_pos = (s - 1) * tps + 1 + OBS_DIM + 1
        expected = action_vals[s]
        got = int(actions_out[0, prev_done_pos].item())
        assert got == expected, (
            f"At done_{{s-1={s-1}}} pos {prev_done_pos}: actions = {got!r}, "
            f"expected a_s={expected!r}"
        )


def test_q_values_at_position_after_done_are_non_nan(tmp_path):
    """After rightward backfill, q_values at position t+1 (where t is a non-last
    DONE token) must be non-NaN.

    In the new ordering the token at done_s+1 is action_{s+1} (first token of
    step s+1).  Backfill propagates Q from done_{s+1} leftward to action_{s+1}.
    The DQN next-step Q lookup is: next_max_q = q_values[t+1].
    """
    num_steps = 3
    model = _make_model(tmp_path)
    stream, tps, seq_len = _make_stream(num_steps)

    with torch.no_grad():
        q_values, _, _ = model(stream)

    # For every non-last done position, position t+1 must have a valid Q-value.
    for s in range(num_steps - 1):
        done_pos = s * tps + 1 + OBS_DIM + 1  # done is last token of step s
        next_pos = done_pos + 1                # action_{s+1} position
        assert not q_values[0, next_pos, 0].isnan(), (
            f"q_values at pos {next_pos} (after done pos {done_pos} of step {s}) "
            f"must be non-NaN after backfill"
        )

    # The position immediately after the *last* done token has no valid Q to its
    # right, so it remains NaN.
    last_done_pos = (num_steps - 1) * tps + 1 + OBS_DIM + 1
    last_next_pos = last_done_pos + 1
    if last_next_pos < seq_len:
        assert q_values[0, last_next_pos, 0].isnan(), (
            f"q_values at pos {last_next_pos} (after last done) should remain NaN"
        )


# ---------------------------------------------------------------------------
# 3. Token order is load-bearing: permuting types changes outputs
# ---------------------------------------------------------------------------


def test_swapping_reward_and_done_types_changes_rewards_output(tmp_path):
    """If REWARD and DONE token types are swapped (values kept the same), the
    rewards output must differ from the correctly-ordered stream.

    This proves that the model's reward parsing depends on position/type, not
    just the raw value stored at that position.

    Note: reward values are restricted to {0.0, 1.0} so they remain valid
    indices for the done_embed layer when the types are swapped (done_embed
    has exactly 2 entries).  The done values are chosen to differ from the
    reward values at step 0 so the swap is detectable.
    """
    from tensordict import TensorDict

    num_steps = 3
    # reward ∈ {0, 1} so they are valid done-embedding indices after the swap.
    # done values also ∈ {0, 1} (bool → float).  At step 0, reward≠done so
    # the rewards output at the reward position must change after the swap.
    reward_vals = [0.0, 0.0, 1.0]      # step 0 reward = 0.0
    done_vals = [True, True, False]     # step 0 done   = 1.0  (≠ 0.0)
    model = _make_model(tmp_path)

    stream_correct, tps, seq_len = _make_stream(
        num_steps, rewards=reward_vals, dones=done_vals
    )

    # Swap DONE ↔ REWARD types only; keep values identical.
    types_swapped = stream_correct["types"].clone()
    for s in range(num_steps):
        base = s * tps
        reward_pos = base + 1 + OBS_DIM      # reward at offset 1+obs_dim
        done_pos = base + 1 + OBS_DIM + 1    # done at offset 1+obs_dim+1
        types_swapped[0, reward_pos] = int(TokenType.DONE)
        types_swapped[0, done_pos] = int(TokenType.REWARD)

    stream_swapped = TensorDict(
        {"types": types_swapped, "values": stream_correct["values"].clone()},
        batch_size=(1, seq_len),
    )

    with torch.no_grad():
        _, rewards_correct, _ = model.extract_stream_tensors(stream_correct)
        _, rewards_swapped, _ = model.extract_stream_tensors(stream_swapped)

    # With the correct order, rewards at the REWARD position equal reward_vals[s].
    # With types swapped, that position is now a DONE token — the reward_mask no
    # longer selects it, so rewards_swapped at that position is backfilled from
    # the misplaced REWARD token at the old done position, giving a different value.
    # At step 0: rewards_correct[reward_pos_0]=0.0, rewards_swapped[reward_pos_0]=1.0.
    any_differ = False
    for s in range(num_steps):
        pos = s * tps + 1 + OBS_DIM  # reward position in new ordering
        if not math.isclose(
            rewards_correct[0, pos].item(),
            rewards_swapped[0, pos].item(),
            rel_tol=1e-4,
        ):
            any_differ = True
            break
    assert any_differ, (
        "Swapping REWARD and DONE types should change the rewards output at "
        "one or more positions"
    )


def test_swapping_obs_and_reward_types_changes_outputs(tmp_path):
    """Replacing OBS tokens with REWARD tokens (and vice versa) changes rewards output.

    Confirms that the model does not treat obs and reward tokens interchangeably.
    """
    from tensordict import TensorDict

    num_steps = 2
    reward_vals = [5.0, 10.0]
    model = _make_model(tmp_path)

    stream_correct, tps, seq_len = _make_stream(num_steps, rewards=reward_vals)

    # Swap first OBS token ↔ REWARD token types (keep values the same).
    types_swapped = stream_correct["types"].clone()
    for s in range(num_steps):
        base = s * tps
        # First obs position (base+1) ↔ reward position (base+1+OBS_DIM)
        types_swapped[0, base + 1] = int(TokenType.REWARD)
        types_swapped[0, base + 1 + OBS_DIM] = int(TokenType.OBS)

    stream_swapped = TensorDict(
        {"types": types_swapped, "values": stream_correct["values"].clone()},
        batch_size=(1, seq_len),
    )

    with torch.no_grad():
        _, rewards_correct, _ = model.extract_stream_tensors(stream_correct)
        _, rewards_swapped, _ = model.extract_stream_tensors(stream_swapped)

    # rewards should differ because reward_mask now selects different positions.
    assert not torch.allclose(rewards_correct, rewards_swapped, atol=1e-4), (
        "Swapping OBS and REWARD types should change rewards output"
    )


# ---------------------------------------------------------------------------
# 4. DQN Bellman target structure
# ---------------------------------------------------------------------------


def test_dqn_mask_includes_non_last_done_positions(tmp_path):
    """The Bellman mask (q_values_mask[:-1] & next_q_valid) must be True at
    every non-last DONE position.

    These are the transitions for which a valid target r + γ * max_Q' exists.
    """
    num_steps = 3
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(num_steps)

    with torch.no_grad():
        q_values, q_values_mask, _ = model(stream)

    next_q_values_valid = ~(q_values[..., 1:, 0].isnan())
    mask = q_values_mask[..., :-1] & next_q_values_valid

    for s in range(num_steps - 1):
        done_pos = s * tps + 1 + OBS_DIM + 1  # done is last token of step s
        assert mask[0, done_pos], (
            f"Done pos {done_pos} (step {s}) must be in the Bellman mask"
        )


def test_dqn_mask_excludes_last_done_position(tmp_path):
    """The last DONE position must be excluded from the Bellman mask.

    In the new per-step ordering [action, obs..., reward, done], DONE is the
    very last token of each step.  The final DONE is therefore at seq_len-1,
    which is always outside mask = q_values_mask[:-1] by construction.
    Additionally, even if it were included, no valid next Q-value exists.
    """
    num_steps = 3
    model = _make_model(tmp_path)
    stream, tps, seq_len = _make_stream(num_steps)

    with torch.no_grad():
        q_values, q_values_mask, _ = model(stream)

    next_q_values_valid = ~(q_values[..., 1:, 0].isnan())
    mask = q_values_mask[..., :-1] & next_q_values_valid

    # The last DONE is at seq_len-1, which is beyond mask's length (seq_len-1).
    last_done_pos = (num_steps - 1) * tps + 1 + OBS_DIM + 1
    assert last_done_pos == seq_len - 1, (
        f"Expected last done at seq_len-1={seq_len - 1}, got {last_done_pos}"
    )
    assert last_done_pos >= mask.shape[-1], (
        f"Last done pos {last_done_pos} should be outside mask (size {mask.shape[-1]})"
    )
    # Confirm the mask has no True entries beyond what the inner steps provide.
    assert not mask[0, last_done_pos - 1:].any(), (
        "No mask entries should be True at or after the last done position"
    )


def test_dqn_non_terminal_discount_is_gamma(tmp_path):
    """When done=False the discount is γ·(1 − 0) = γ.

    Verifies that the dones output correctly returns 0.0 at a non-terminal
    DONE token so the standard discount applies.
    """
    gamma = 0.99
    gamma_done = 0.0
    done_flags = [False, True]  # step 0 non-terminal
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(2, dones=done_flags)

    with torch.no_grad():
        _, _, dones_out = model.extract_stream_tensors(stream)

    done_pos_0 = 0 * tps + 1 + OBS_DIM + 1  # done is last token of step 0
    d = dones_out[0, done_pos_0].item()
    discount = gamma * (1.0 - d) + gamma_done * d

    assert math.isclose(d, 0.0, abs_tol=1e-6), (
        f"Step 0 done should be 0.0 (non-terminal), got {d!r}"
    )
    assert math.isclose(discount, gamma, rel_tol=1e-6), (
        f"Non-terminal discount should equal gamma={gamma}, got {discount!r}"
    )


def test_dqn_terminal_discount_is_gamma_done(tmp_path):
    """When done=True the discount is γ_done·1 = γ_done.

    Verifies that the dones output correctly returns 1.0 at a terminal
    DONE token so the episode-end discount (γ_done, typically 0) applies.
    """
    gamma = 0.99
    gamma_done = 0.1  # distinct from gamma so the assertion is meaningful
    done_flags = [True, False]  # step 0 terminal
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(2, dones=done_flags)

    with torch.no_grad():
        _, _, dones_out = model.extract_stream_tensors(stream)

    done_pos_0 = 0 * tps + 1 + OBS_DIM + 1  # done is last token of step 0
    d = dones_out[0, done_pos_0].item()
    discount = gamma * (1.0 - d) + gamma_done * d

    assert math.isclose(d, 1.0, abs_tol=1e-6), (
        f"Step 0 done should be 1.0 (terminal), got {d!r}"
    )
    assert math.isclose(discount, gamma_done, rel_tol=1e-6), (
        f"Terminal discount should equal gamma_done={gamma_done}, got {discount!r}"
    )


def test_dqn_bellman_reward_at_done_position_is_next_step_reward(tmp_path):
    """At the DONE position of step s, rewards[done_s] equals r_{s+1} (the next
    step's reward) — not r_s.

    This is a consequence of the per-step layout [action, obs..., reward, done]:
    DONE comes *after* REWARD, so rightward backfill propagates the next step's
    REWARD token to the current step's DONE token.

    For the DQN training loss (computed at done_{s-1}):
        Q(context_{s-1}, a_s) ← rewards[done_{s-1}] + γ * max_a Q(done_s, a)
    where rewards[done_{s-1}] = r_s (reward of step s, not s-1).
    """
    reward_vals = [0.5, 1.5, 2.5]
    done_vals = [False, False, True]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, rewards=reward_vals, dones=done_vals)

    with torch.no_grad():
        _, rewards_out, _ = model.extract_stream_tensors(stream)

    for s in range(len(reward_vals) - 1):  # exclude last (no next reward)
        done_pos = s * tps + 1 + OBS_DIM + 1  # done is last token of step s
        expected = reward_vals[s + 1]          # next step's reward
        got = rewards_out[0, done_pos].item()
        assert math.isclose(got, expected, rel_tol=1e-4), (
            f"Bellman reward at done pos of step {s} (pos {done_pos}): "
            f"got {got!r}, expected next-step reward {expected!r}"
        )


def test_dqn_q_gather_uses_current_step_action(tmp_path):
    """The DQN loss gathers Q(context_{s-1}, a_s) using actions[done_{s-1}].

    The Bellman-loss anchor is done_{s-1}, not done_s.  This test verifies the
    full chain: at done_{s-1}, the backfilled action is a_s (action_s is the
    next ACTION token to the right), and Q-values there are finite for the
    gather to work.
    """
    action_vals = [2, 0, 3]
    model = _make_model(tmp_path)
    stream, tps, _ = _make_stream(3, actions=action_vals)

    with torch.no_grad():
        q_values, q_values_mask, _ = model(stream)
        actions_out, _, _ = model.extract_stream_tensors(stream)

    for s in range(1, len(action_vals)):  # s=1..N-1: done_{s-1} is defined
        prev_done_pos = (s - 1) * tps + 1 + OBS_DIM + 1
        expected_action = action_vals[s]
        # Verify actions backfill: done_{s-1} carries a_s.
        got_action = int(actions_out[0, prev_done_pos].item())
        assert got_action == expected_action, (
            f"s={s}: actions[done_{{s-1}}={prev_done_pos}] = {got_action!r}, "
            f"expected a_s={expected_action!r}"
        )
        # Verify gather is possible: Q-values at done_{s-1} are finite.
        assert not q_values[0, prev_done_pos, expected_action].isnan(), (
            f"s={s}: Q-value at done_{{s-1}}={prev_done_pos}, "
            f"action={expected_action} must be finite for gather to work"
        )
