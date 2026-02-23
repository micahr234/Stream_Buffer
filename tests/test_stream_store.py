"""Comprehensive StreamStore test suite.

Encodes all desired behaviours:
  - append individual fields with caller-supplied step index
  - 1-D, 2-D and N-D __getitem__ returning a TensorDict
  - from_dataset → to_dataset roundtrip (core fields + extra)
  - Variable-length observation support (no hardcoded obs_dim)
  - step field enables type-driven reconstruction without shape assumptions
  - np.memmap storage: numpy-only internals, torch only at __getitem__ boundary
  - Dynamic capacity (buffer grows as needed)
"""

from __future__ import annotations

import math
import sys
import os
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import Dataset, Features, Sequence as HFSequence, Value
from definitions import FIELD_TO_TYPE, TokenType
from stream_store import StreamStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4


def _make_store(capacity: int = 4096) -> StreamStore:
    return StreamStore(field_to_type=FIELD_TO_TYPE, env_name="test_env", env_number=0, capacity=capacity)


def _core_columns(ds: Dataset) -> Dataset:
    """Keep only columns known to FIELD_TO_TYPE, in canonical token order."""
    canonical = list(FIELD_TO_TYPE.keys())
    return Dataset.from_dict({c: ds[c] for c in canonical if c in ds.column_names})


def _make_dataset(num_rows: int = 20, obs_dim: int = OBS_DIM) -> Dataset:
    """Dataset with core columns plus numeric and string extras."""
    observations = [[float(i * obs_dim + d) for d in range(obs_dim)] for i in range(num_rows)]
    rewards = [float(i) * 0.1 for i in range(num_rows)]
    dones = [i % 7 == 6 for i in range(num_rows)]
    actions = [i % 3 for i in range(num_rows)]
    step_ids = list(range(num_rows))
    episode_ids = [i // 7 for i in range(num_rows)]
    relative_times = [i % 7 for i in range(num_rows)]
    env_ids = [f"env({i % 2})" for i in range(num_rows)]

    return Dataset.from_dict(
        {
            "observation": observations,
            "reward": rewards,
            "done": dones,
            "action": actions,
            "step_id": step_ids,
            "episode_id": episode_ids,
            "relative_time": relative_times,
            "env_id": env_ids,
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
                "step_id": Value("int64"),
                "episode_id": Value("int64"),
                "relative_time": Value("int64"),
                "env_id": Value("string"),
            }
        ),
    )


# ---------------------------------------------------------------------------
# 1. append: step tagging and token counts
# ---------------------------------------------------------------------------


def test_append_token_count():
    obs_dim = 4
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [0, obs, 0.0, False], step=0)
    assert len(store) == obs_dim + 3  # action + 4 obs + reward + done = 7


def test_append_accumulates():
    obs_dim = 4
    N = 10
    store = _make_store()
    for i in range(N):
        store.append(["action", "observation", "reward", "done"],
                     [i % 3, np.ones(obs_dim) * i, float(i), False], step=i)
    assert len(store) == N * (obs_dim + 3)


def test_append_step_stored_in_buffer():
    """Every token from a call gets the step value passed by the caller."""
    store = _make_store()
    obs = np.array([1.0, 2.0], dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [1, obs, 0.5, False], step=42)
    # Read the live buffer directly to verify step values.
    raw = store._buf[:]
    assert (raw["step"] == 42).all()


def test_append_allows_variable_obs_dim():
    """Different steps may have different obs_dim — no error should be raised."""
    store = _make_store(capacity=500)
    store.append(["action", "observation", "reward", "done"],
                 [0, np.zeros(3), 0.0, False], step=0)
    store.append(["action", "observation", "reward", "done"],
                 [0, np.zeros(5), 0.0, False], step=1)
    assert len(store) == (3 + 3) + (5 + 3)


def test_split_append_same_step():
    """Split appends (action first, then obs/reward/done) with same step work correctly."""
    obs_dim = 3
    store = _make_store()
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    store.append(["action"], [2], step=7)
    assert len(store) == 1

    store.append(["observation", "reward", "done"], [obs, 0.5, False], step=7)
    assert len(store) == obs_dim + 3

    # Both appends share step=7.
    raw = store._buf[:]
    assert (raw["step"] == 7).all()


def test_append_unknown_field_raises():
    store = _make_store()
    with pytest.raises(ValueError, match="Unknown field"):
        store.append(["bad_field"], [1.0], step=0)


def test_append_mismatched_lengths_raises():
    store = _make_store()
    with pytest.raises(ValueError):
        store.append(["observation", "reward"], [np.zeros(4)], step=0)


def test_append_grows_beyond_initial_capacity():
    """NumpyMemmapBuffer grows dynamically; initial capacity is just a chunk-size hint."""
    store = _make_store(capacity=3)
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [1, obs, 0.5, False], step=0)
    assert len(store) == 6


# ---------------------------------------------------------------------------
# 2. __getitem__: 1-D, 2-D and N-D indices return TensorDict
# ---------------------------------------------------------------------------


def test_getitem_1d_types():
    obs_dim = 2
    store = _make_store()
    obs = np.array([1.0, 2.0], dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [1, obs, 0.5, True], step=0)
    result = store[torch.tensor([1, 2])]
    assert result["types"].tolist() == [int(TokenType.OBS), int(TokenType.OBS)]


def test_getitem_reward_token():
    obs_dim = 2
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [0, obs, 9.9, False], step=0)
    # New ordering: action(0), obs(1..obs_dim), reward(obs_dim+1), done(obs_dim+2)
    result = store[torch.tensor([obs_dim + 1])]
    assert result["types"].tolist() == [int(TokenType.REWARD)]
    assert result["values"][0].item() == pytest.approx(9.9, rel=1e-4)


def test_getitem_done_token():
    obs_dim = 2
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [0, obs, 0.0, True], step=0)
    # New ordering: action(0), obs(1..obs_dim), reward(obs_dim+1), done(obs_dim+2)
    result = store[torch.tensor([obs_dim + 2])]
    assert result["types"].tolist() == [int(TokenType.DONE)]
    assert bool(result["values"][0].item()) is True


def test_getitem_action_token():
    obs_dim = 2
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(["action", "observation", "reward", "done"], [7, obs, 0.0, False], step=0)
    # New ordering: action is first (position 0)
    result = store[torch.tensor([0])]
    assert result["types"].tolist() == [int(TokenType.ACTION)]
    assert result["values"][0].item() == pytest.approx(7.0)


def test_getitem_3d_shape():
    obs_dim = 2
    N, tps = 4, obs_dim + 3
    store = _make_store()
    for i in range(N):
        store.append(["action", "observation", "reward", "done"],
                     [i, np.ones(obs_dim) * i, float(i), False], step=i)
    idx = torch.arange(N * tps).reshape(2, 2, tps)
    result = store[idx]
    assert result["types"].shape == (2, 2, tps)


def test_getitem_2d_shape():
    obs_dim = 3
    N, tps = 5, obs_dim + 3
    store = _make_store()
    for i in range(N):
        store.append(["action", "observation", "reward", "done"],
                     [i, np.ones(obs_dim) * i, float(i), False], step=i)
    result = store[torch.arange(N * tps).reshape(N, tps)]
    assert result["types"].shape == (N, tps)


def test_getitem_2d_types():
    obs_dim = 3
    N, tps = 4, obs_dim + 3
    store = _make_store()
    for i in range(N):
        store.append(["action", "observation", "reward", "done"],
                     [i, np.ones(obs_dim) * i, float(i), False], step=i)
    result = store[torch.arange(N * tps).reshape(N, tps)]
    types = result["types"]
    # New ordering per step: action(0), obs(1..obs_dim), reward(obs_dim+1), done(obs_dim+2)
    assert (types[:, 0] == int(TokenType.ACTION)).all()
    assert (types[:, 1:obs_dim + 1] == int(TokenType.OBS)).all()
    assert (types[:, obs_dim + 1] == int(TokenType.REWARD)).all()
    assert (types[:, obs_dim + 2] == int(TokenType.DONE)).all()


def test_getitem_context_window():
    obs_dim = 2
    tps = obs_dim + 3
    N = 5
    store = _make_store()
    for i in range(N):
        obs = np.array([float(i), float(i + 1)], dtype=np.float32)
        store.append(["action", "observation", "reward", "done"],
                     [0, obs, float(i), False], step=i)
    K = 2 * tps
    ctx = store[torch.arange(len(store) - K, len(store))]
    assert ctx["types"].shape == (K,)
    # New ordering: first token of window is action of step N-2 (value = 0.0)
    assert ctx["values"][0].item() == pytest.approx(0.0, abs=1e-5)


def test_getitem_returns_tensors():
    store = _make_store()
    store.append(["action", "observation", "reward", "done"],
                 [0, np.zeros(2), 0.0, False], step=0)
    result = store[torch.tensor([0])]
    assert isinstance(result["types"], np.ndarray)
    assert isinstance(result["values"], np.ndarray)
    assert result["values"].dtype == np.float64


def test_getitem_bit_reinterpretation():
    """All values are stored as float64 raw bits (int64) and recovered via view.

    Checks four properties:
      1. Float payload is recovered exactly (no rounding — float64 roundtrip).
      2. Integer payload (action, done) is recovered exactly when cast back to int.
      3. Large integer (2^20) fits without precision loss (float64 is exact up to 2^53).
      4. numpy's .view(np.int64) on the returned float64 array yields the same bit
         pattern that was originally stored — proving the chain is lossless end-to-end.
    """
    obs_val = np.pi          # high-precision float (not representable in float32)
    reward  = -273.15
    done    = True
    action  = 2 ** 20        # large int, well within float64's exact integer range

    store = _make_store()
    store.append(
        ["action", "observation", "reward", "done"],
        [action, np.array([obs_val]), reward, done],
        step=0,
    )

    # 4 tokens: action, obs, reward, done
    vals = store[np.arange(4)]["values"]
    assert vals.dtype == np.float64

    # 1. Action integer recovered exactly (now first)
    assert int(vals[0].item()) == action

    # 2. Float obs preserved exactly (float64 roundtrip, not float32)
    assert vals[1].item() == obs_val

    # 3. Float reward preserved exactly
    assert vals[2].item() == reward

    # 4. Done (bool→1.0) casts back to True without loss
    assert bool(vals[3].item()) is done

    # numpy-level: view the float64 array as int64 — should match the bits
    # that were originally stored (float64(value).view(int64)).
    for token_idx, value in [(0, float(action)), (1, obs_val), (2, reward), (3, float(done))]:
        expected_bits = int(np.float64(value).view(np.int64))
        stored_bits  = vals[token_idx : token_idx + 1].view(np.int64)[0].item()
        assert stored_bits == expected_bits, (
            f"Token {token_idx}: expected bits {expected_bits}, got {stored_bits}"
        )


# ---------------------------------------------------------------------------
# 3. from_dataset → to_dataset roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_core_columns():
    original = _make_dataset(num_rows=30)
    store = _make_store(capacity=4096)
    store.from_dataset(_core_columns(original))
    recovered = store.to_dataset()

    assert len(recovered) == len(original)
    for i in range(len(original)):
        for a, b in zip(original[i]["observation"], recovered[i]["observation"]):
            assert math.isclose(a, b, rel_tol=1e-5), f"Row {i} obs mismatch"
        assert math.isclose(original[i]["reward"], recovered[i]["reward"], rel_tol=1e-5)
        assert original[i]["done"] == recovered[i]["done"]
        assert original[i]["action"] == recovered[i]["action"]


def test_roundtrip_metadata_ignored():
    """Non-core numeric columns (episode_id, relative_time, etc.) are ignored on load."""
    original = _make_dataset(num_rows=30)
    store = _make_store(capacity=4096)
    store.from_dataset(_core_columns(original))
    recovered = store.to_dataset()

    for col in ("episode_id", "relative_time"):
        assert col not in recovered.column_names, f"Metadata column {col} should be ignored"


def test_string_columns_excluded():
    original = _make_dataset(num_rows=20)
    store = _make_store(capacity=4096)
    store.from_dataset(_core_columns(original))
    recovered = store.to_dataset()
    assert "env_id" not in recovered.column_names


def test_from_dataset_token_count():
    num_rows = 20
    obs_dim = OBS_DIM
    original = _make_dataset(num_rows=num_rows, obs_dim=obs_dim)
    store = _make_store(capacity=4096)
    store.from_dataset(_core_columns(original))
    assert len(store) == num_rows * (obs_dim + 3)


def test_from_dataset_step_values():
    """After from_dataset, token step values must equal the row index."""
    num_rows = 5
    obs_dim = 2
    ds = Dataset.from_dict(
        {
            "observation": [[float(i)] * obs_dim for i in range(num_rows)],
            "reward": [float(i) for i in range(num_rows)],
            "done": [False] * num_rows,
            "action": [0] * num_rows,
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
            }
        ),
    )
    store = _make_store(capacity=512)
    store.from_dataset(ds)
    raw = store._buf[:]
    tps = obs_dim + 3
    for i in range(num_rows):
        s, e = i * tps, (i + 1) * tps
        assert (raw["step"][s:e] == i).all(), f"Row {i}: step values wrong"


def test_roundtrip_no_metadata():
    num_rows = 15
    obs_dim = 3
    observations = [[float(i * obs_dim + d) for d in range(obs_dim)] for i in range(num_rows)]
    ds = Dataset.from_dict(
        {
            "observation": observations,
            "reward": [float(i) * 0.2 for i in range(num_rows)],
            "done": [i % 5 == 4 for i in range(num_rows)],
            "action": [i % 4 for i in range(num_rows)],
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
            }
        ),
    )
    store = _make_store(capacity=1024)
    store.from_dataset(ds)
    recovered = store.to_dataset()
    assert len(recovered) == num_rows
    for i in range(num_rows):
        for a, b in zip(ds[i]["observation"], recovered[i]["observation"]):
            assert math.isclose(a, b, rel_tol=1e-5)
        assert math.isclose(ds[i]["reward"], recovered[i]["reward"], rel_tol=1e-5)
        assert ds[i]["done"] == recovered[i]["done"]
        assert ds[i]["action"] == recovered[i]["action"]


# ---------------------------------------------------------------------------
# 4. to_dataset with extra columns
# ---------------------------------------------------------------------------


def test_to_dataset_extra_string_column():
    num_rows = 10
    ds = Dataset.from_dict(
        {
            "observation": [[float(i)] for i in range(num_rows)],
            "reward": [float(i) for i in range(num_rows)],
            "done": [False] * num_rows,
            "action": [0] * num_rows,
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
            }
        ),
    )
    store = _make_store(capacity=512)
    store.from_dataset(ds)
    env_ids = [f"env({i})" for i in range(num_rows)]
    recovered = store.to_dataset(extra={"env_id": env_ids})
    assert "env_id" in recovered.column_names
    for i in range(num_rows):
        assert recovered[i]["env_id"] == env_ids[i]


def test_to_dataset_extra_wrong_length_raises():
    num_rows = 5
    ds = Dataset.from_dict(
        {
            "observation": [[0.0] for _ in range(num_rows)],
            "reward": [0.0] * num_rows,
            "done": [False] * num_rows,
            "action": [0] * num_rows,
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
            }
        ),
    )
    store = _make_store()
    store.from_dataset(ds)
    with pytest.raises(ValueError, match="extra column"):
        store.to_dataset(extra={"env_id": ["x"] * (num_rows + 1)})


# ---------------------------------------------------------------------------
# 5. to_dataset after deploy-style appends (no metadata in store)
# ---------------------------------------------------------------------------


def test_to_dataset_after_appends():
    obs_dim = 3
    N = 8
    store = _make_store()
    obs_vals, rew_vals, done_vals, act_vals = [], [], [], []
    for i in range(N):
        obs = np.arange(obs_dim, dtype=np.float32) + i * obs_dim
        rew, done, act = float(i) * 0.1, i == N - 1, i % 3
        obs_vals.append(obs.tolist())
        rew_vals.append(rew)
        done_vals.append(done)
        act_vals.append(act)
        store.append(["action", "observation", "reward", "done"],
                     [act, obs, rew, done], step=i)

    ds = store.to_dataset()
    assert len(ds) == N
    for i in range(N):
        for a, b in zip(obs_vals[i], ds[i]["observation"]):
            assert math.isclose(a, b, rel_tol=1e-5)
        assert math.isclose(rew_vals[i], ds[i]["reward"], rel_tol=1e-5)
        assert done_vals[i] == ds[i]["done"]
        assert act_vals[i] == ds[i]["action"]


def test_to_dataset_split_appends():
    obs_dim = 2
    N = 5
    store = _make_store()
    obs_vals, act_vals = [], []
    for i in range(N):
        obs = np.array([float(i), float(i + 1)], dtype=np.float32)
        obs_vals.append(obs.tolist())
        store.append(["action"], [i % 4], step=i)
        store.append(["observation", "reward", "done"], [obs, float(i), False], step=i)
        act_vals.append(i % 4)

    ds = store.to_dataset()
    assert len(ds) == N
    for i in range(N):
        for a, b in zip(obs_vals[i], ds[i]["observation"]):
            assert math.isclose(a, b, rel_tol=1e-5)
        assert ds[i]["action"] == act_vals[i]


def test_to_dataset_variable_obs_dim():
    """to_dataset correctly reconstructs steps with different obs lengths."""
    store = _make_store(capacity=500)
    obs3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    obs5 = np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    store.append(["action", "observation", "reward", "done"], [0, obs3, 0.1, False], step=0)
    store.append(["action", "observation", "reward", "done"], [1, obs5, 0.2, True], step=1)

    ds = store.to_dataset()
    assert len(ds) == 2
    assert len(ds[0]["observation"]) == 3
    assert len(ds[1]["observation"]) == 5
    for a, b in zip(obs3.tolist(), ds[0]["observation"]):
        assert math.isclose(a, b, rel_tol=1e-5)
    for a, b in zip(obs5.tolist(), ds[1]["observation"]):
        assert math.isclose(a, b, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 6. Guard conditions
# ---------------------------------------------------------------------------


def test_empty_store_raises_on_to_dataset():
    store = _make_store()
    with pytest.raises(ValueError):
        store.to_dataset()


def test_no_complete_steps_raises_on_to_dataset():
    store = _make_store()
    store.append(["observation", "reward", "done"],
                 [np.zeros(3), 0.5, False], step=0)
    with pytest.raises(ValueError, match="complete steps"):
        store.to_dataset()


def test_from_dataset_grows_beyond_initial_capacity():
    ds = _make_dataset(num_rows=100, obs_dim=OBS_DIM)
    store = _make_store(capacity=10)
    store.from_dataset(_core_columns(ds))
    assert len(store) == 100 * (OBS_DIM + 3)


def test_zero_capacity_raises():
    with pytest.raises(ValueError):
        StreamStore(field_to_type=FIELD_TO_TYPE, env_name="test_env", env_number=0, capacity=0)


def test_clear_resets_state():
    store = _make_store()
    store.append(["action", "observation", "reward", "done"],
                 [1, np.array([1.0, 2.0]), 0.5, False], step=0)
    assert len(store) > 0
    store._clear()
    assert len(store) == 0


# ---------------------------------------------------------------------------
# 7. Column order preserved
# ---------------------------------------------------------------------------


def test_column_order_preserved():
    original = _make_dataset(num_rows=10)
    store = _make_store(capacity=1024)
    store.from_dataset(_core_columns(original))
    recovered = store.to_dataset()
    assert recovered.column_names[:5] == ["env_name", "env_number", "step_id", "action", "observation"]


# ---------------------------------------------------------------------------
# 8. Performance benchmarks
# ---------------------------------------------------------------------------


def test_from_dataset_performance():
    N = 50_000
    obs_dim = 4
    ds = Dataset.from_dict(
        {
            "observation": np.random.randn(N, obs_dim).astype(np.float32).tolist(),
            "reward": np.random.randn(N).astype(np.float32).tolist(),
            "done": [bool(i % 100 == 99) for i in range(N)],
            "action": np.random.randint(0, 3, N).tolist(),
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
            }
        ),
    )
    store = StreamStore(field_to_type=FIELD_TO_TYPE, env_name="test_env", env_number=0, capacity=N * 10)
    t0 = time.time()
    store.from_dataset(ds)
    assert time.time() - t0 < 15.0, f"from_dataset too slow: {time.time() - t0:.2f}s"


def test_to_dataset_performance():
    N = 50_000
    obs_dim = 4
    ds = Dataset.from_dict(
        {
            "observation": np.random.randn(N, obs_dim).astype(np.float32).tolist(),
            "reward": np.random.randn(N).astype(np.float32).tolist(),
            "done": [bool(i % 100 == 99) for i in range(N)],
            "action": np.random.randint(0, 3, N).tolist(),
        },
        features=Features(
            {
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
                "action": Value("int64"),
            }
        ),
    )
    store = StreamStore(field_to_type=FIELD_TO_TYPE, env_name="test_env", env_number=0, capacity=N * (obs_dim + 3) + 1000)
    store.from_dataset(ds)
    t0 = time.time()
    _ = store.to_dataset()
    assert time.time() - t0 < 10.0, f"to_dataset too slow: {time.time() - t0:.2f}s"
