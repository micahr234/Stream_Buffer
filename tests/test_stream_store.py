"""Comprehensive StreamStore test suite.

Encodes all desired behaviours:
  - append individual fields with caller-supplied step index
  - 1-D, 2-D and N-D __getitem__ returning a TensorDict
  - from_dataset → to_dataset roundtrip (core fields + extra)
  - Variable-length field support (no hardcoded field length)
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
from stream_store import StreamStore

# Fields used throughout these tests — type IDs are auto-assigned 0, 1, 2, 3.
FIELDS = ["action", "observation", "reward", "done"]

# Integer type IDs for assertion use (matches auto-assignment order above).
_ACTION_TYPE      = 0
_OBS_TYPE         = 1
_REWARD_TYPE      = 2
_DONE_TYPE        = 3

# HuggingFace feature types matching the test fields.
_STORE_FEATURES = {
    "action":      Value("int64"),
    "observation": HFSequence(Value("float32")),
    "reward":      Value("float32"),
    "done":        Value("bool"),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4


def _make_store(capacity: int = 4096) -> StreamStore:
    store = StreamStore(capacity=capacity)
    store.define_fields(FIELDS, field_features=_STORE_FEATURES)
    return store


def _core_columns(ds: Dataset) -> Dataset:
    """Keep only columns known to FIELDS, in canonical token order."""
    return Dataset.from_dict({c: ds[c] for c in FIELDS if c in ds.column_names})


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
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[0, obs, 0.0, False])
    assert len(store) == obs_dim + 3  # action + 4 obs + reward + done = 7


def test_append_accumulates():
    obs_dim = 4
    N = 10
    fields = ["action", "observation", "reward", "done"]
    store = _make_store()
    for i in range(N):
        store.append(steps=[i] * 4, names=fields,
                     values=[i % 3, np.ones(obs_dim) * i, float(i), False])
    assert len(store) == N * (obs_dim + 3)


def test_append_multi_step_inserts_in_order():
    """Batched append writes tokens in step order."""
    obs_dim = 2
    N = 3
    fields = ["action", "observation", "reward", "done"]
    store = _make_store()
    for i in range(N):
        store.append(steps=[i] * 4, names=fields,
                     values=[i, np.ones(obs_dim) * i, float(i), False])
    raw = store._buf[:]
    tps = obs_dim + 3
    for i in range(N):
        assert (raw["step"][i * tps:(i + 1) * tps] == i).all()


def test_append_step_stored_in_buffer():
    """Every token from a call gets the step value passed by the caller."""
    store = _make_store()
    obs = np.array([1.0, 2.0], dtype=np.float32)
    store.append(steps=[42, 42, 42, 42], names=["action", "observation", "reward", "done"],
                 values=[1, obs, 0.5, False])
    raw = store._buf[:]
    assert (raw["step"] == 42).all()


def test_append_allows_variable_obs_dim():
    """Different steps may have different obs_dim — no error should be raised."""
    store = _make_store(capacity=500)
    fields = ["action", "observation", "reward", "done"]
    store.append(steps=[0] * 4, names=fields, values=[0, np.zeros(3), 0.0, False])
    store.append(steps=[1] * 4, names=fields, values=[0, np.zeros(5), 0.0, False])
    assert len(store) == (3 + 3) + (5 + 3)


def test_split_append_same_step():
    """Split appends (action first, then obs/reward/done) with same step work correctly."""
    obs_dim = 3
    store = _make_store()
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    store.append(steps=[7], names=["action"], values=[2])
    assert len(store) == 1

    store.append(steps=[7, 7, 7], names=["observation", "reward", "done"],
                 values=[obs, 0.5, False])
    assert len(store) == obs_dim + 3

    raw = store._buf[:]
    assert (raw["step"] == 7).all()


def test_append_unknown_field_raises():
    store = _make_store()
    with pytest.raises(ValueError, match="Unknown field"):
        store.append(steps=[0], names=["bad_field"], values=[1.0])


def test_append_unequal_lengths_raises():
    store = _make_store()
    with pytest.raises(ValueError):
        store.append(steps=[0, 1], names=["action"], values=[0])


def test_append_out_of_order_raises():
    store = _make_store()
    with pytest.raises(ValueError, match="order"):
        store.append(steps=[0, 0], names=["reward", "action"], values=[0.0, 1])


def test_append_reversed_order_raises():
    store = _make_store()
    with pytest.raises(ValueError, match="order"):
        store.append(steps=[0, 0, 0], names=["done", "observation", "reward"],
                     values=[False, np.zeros(2), 0.0])


def test_append_skipping_fields_preserves_order():
    """Missing fields are fine as long as present fields stay in canonical order."""
    store = _make_store()
    store.append(steps=[0, 0, 0], names=["action", "reward", "done"], values=[1, 0.5, False])
    assert len(store) == 3


def test_split_append_order_enforced():
    """Each split append call must itself be in order."""
    store = _make_store()
    store.append(steps=[0], names=["action"], values=[0])
    store.append(steps=[0, 0, 0], names=["observation", "reward", "done"],
                 values=[np.zeros(2), 0.5, False])
    with pytest.raises(ValueError, match="order"):
        store.append(steps=[1, 1], names=["done", "reward"], values=[False, 0.0])


def test_cross_call_order_enforced():
    """Order state persists across append calls — the buffer's last token defines what comes next."""
    store = _make_store()
    # Write action + observation for step 0.
    store.append(steps=[0, 0], names=["action", "observation"], values=[0, np.zeros(2)])
    # Trying to re-insert action for the same step must fail.
    with pytest.raises(ValueError, match="order"):
        store.append(steps=[0], names=["action"], values=[1])


def test_append_grows_beyond_initial_capacity():
    """NumpyMemmapBuffer grows dynamically; initial capacity is just a chunk-size hint."""
    store = _make_store(capacity=3)
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[1, obs, 0.5, False])
    assert len(store) == 6


# ---------------------------------------------------------------------------
# 2. __getitem__: 1-D, 2-D and N-D indices return TensorDict
# ---------------------------------------------------------------------------


def test_getitem_1d_types():
    obs_dim = 2
    store = _make_store()
    obs = np.array([1.0, 2.0], dtype=np.float32)
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[1, obs, 0.5, True])
    result = store[torch.tensor([1, 2])]
    assert result["types"].tolist() == [_OBS_TYPE, _OBS_TYPE]


def test_getitem_reward_token():
    obs_dim = 2
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[0, obs, 9.9, False])
    result = store[torch.tensor([obs_dim + 1])]
    assert result["types"].tolist() == [_REWARD_TYPE]
    val = result["values"][0:1].view(np.float64)[0]
    assert val == pytest.approx(9.9, rel=1e-4)


def test_getitem_done_token():
    obs_dim = 2
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[0, obs, 0.0, True])
    result = store[torch.tensor([obs_dim + 2])]
    assert result["types"].tolist() == [_DONE_TYPE]
    # done=True is stored as int64(1)
    assert result["values"][0].item() == 1


def test_getitem_action_token():
    obs_dim = 2
    store = _make_store()
    obs = np.zeros(obs_dim, dtype=np.float32)
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[7, obs, 0.0, False])
    result = store[torch.tensor([0])]
    assert result["types"].tolist() == [_ACTION_TYPE]
    # action is stored as int64 directly (not a float bit-pattern)
    assert result["values"][0].item() == 7


def _append_steps(store, N, obs_dim):
    fields = ["action", "observation", "reward", "done"]
    for i in range(N):
        store.append(steps=[i] * 4, names=fields,
                     values=[i, np.ones(obs_dim) * i, float(i), False])


def test_getitem_3d_shape():
    obs_dim = 2
    N, tps = 4, obs_dim + 3
    store = _make_store()
    _append_steps(store, N, obs_dim)
    idx = torch.arange(N * tps).reshape(2, 2, tps)
    result = store[idx]
    assert result["types"].shape == (2, 2, tps)


def test_getitem_2d_shape():
    obs_dim = 3
    N, tps = 5, obs_dim + 3
    store = _make_store()
    _append_steps(store, N, obs_dim)
    result = store[torch.arange(N * tps).reshape(N, tps)]
    assert result["types"].shape == (N, tps)


def test_getitem_2d_types():
    obs_dim = 3
    N, tps = 4, obs_dim + 3
    store = _make_store()
    _append_steps(store, N, obs_dim)
    result = store[torch.arange(N * tps).reshape(N, tps)]
    types = result["types"]
    assert (types[:, 0] == _ACTION_TYPE).all()
    assert (types[:, 1:obs_dim + 1] == _OBS_TYPE).all()
    assert (types[:, obs_dim + 1] == _REWARD_TYPE).all()
    assert (types[:, obs_dim + 2] == _DONE_TYPE).all()


def test_getitem_context_window():
    obs_dim = 2
    tps = obs_dim + 3
    N = 5
    fields = ["action", "observation", "reward", "done"]
    store = _make_store()
    for i in range(N):
        store.append(steps=[i] * 4, names=fields,
                     values=[0, np.array([float(i), float(i + 1)], dtype=np.float32),
                             float(i), False])
    K = 2 * tps
    ctx = store[torch.arange(len(store) - K, len(store))]
    assert ctx["types"].shape == (K,)
    # First token of window is action of step N-2 (value = 0.0).
    # float64(0.0) has bit pattern 0, so raw int64 is also 0.
    assert ctx["values"][0].item() == 0


def test_getitem_returns_tensors():
    store = _make_store()
    store.append(steps=[0, 0, 0, 0], names=["action", "observation", "reward", "done"],
                 values=[0, np.zeros(2), 0.0, False])
    result = store[torch.tensor([0])]
    assert isinstance(result["types"], np.ndarray)
    assert isinstance(result["values"], np.ndarray)
    assert result["values"].dtype == np.int64


def test_getitem_raw_bits():
    """__getitem__ returns raw int64 bit patterns.

    - Float fields: store the float64 bit pattern as int64; recover via .view(float64).
    - Int/bool fields: store the int64 value directly; recover by reading as int64.

    Checks:
      1. Returned dtype is int64.
      2. Float fields: raw bits match np.float64(value).view(int64).
         Int/bool fields: raw bits equal the integer value directly.
      3. Callers can recover original values using the correct reinterpretation.
      4. Large integers (2^20) survive losslessly.
    """
    obs_val = np.pi
    reward  = -273.15
    done    = True
    action  = 2 ** 20

    store = _make_store()
    store.append(
        steps=[0, 0, 0, 0],
        names=["action", "observation", "reward", "done"],
        values=[action, np.array([obs_val]), reward, done],
    )

    vals = store[np.arange(4)]["values"]

    # 1. dtype is raw int64
    assert vals.dtype == np.int64

    # 2. Raw bit storage per field type
    # action (int64): stored directly as int64
    assert vals[0].item() == action
    # observation (float64 bit pattern of pi)
    assert vals[1].item() == int(np.float64(obs_val).view(np.int64))
    # reward (float64 bit pattern of -273.15)
    assert vals[2].item() == int(np.float64(reward).view(np.int64))
    # done (bool → int64): stored as 1
    assert vals[3].item() == int(done)

    # 3. Recovering values
    assert vals[0].item() == action                            # int: read directly
    assert vals[1:2].view(np.float64)[0] == obs_val            # float: view as float64
    assert vals[2:3].view(np.float64)[0] == reward             # float: view as float64
    assert bool(vals[3].item()) is done                        # bool: nonzero → True


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
            "action": [0] * num_rows,
            "observation": [[float(i)] * obs_dim for i in range(num_rows)],
            "reward": [float(i) for i in range(num_rows)],
            "done": [False] * num_rows,
        },
        features=Features(
            {
                "action": Value("int64"),
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
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
            "action": [i % 4 for i in range(num_rows)],
            "observation": observations,
            "reward": [float(i) * 0.2 for i in range(num_rows)],
            "done": [i % 5 == 4 for i in range(num_rows)],
        },
        features=Features(
            {
                "action": Value("int64"),
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
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


def test_to_dataset_roundtrip_features():
    """to_dataset uses the feature types from define_fields for the output schema."""
    num_rows = 10
    ds = Dataset.from_dict(
        {
            "action": [0] * num_rows,
            "observation": [[float(i)] for i in range(num_rows)],
            "reward": [float(i) for i in range(num_rows)],
            "done": [False] * num_rows,
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
    recovered = store.to_dataset()
    assert set(recovered.column_names) == {"action", "observation", "reward", "done"}
    assert len(recovered) == num_rows


# ---------------------------------------------------------------------------
# 5. to_dataset after deploy-style appends (no metadata in store)
# ---------------------------------------------------------------------------


def test_to_dataset_after_appends():
    obs_dim = 3
    N = 8
    act_vals  = [i % 3 for i in range(N)]
    obs_vals  = [(np.arange(obs_dim, dtype=np.float32) + i * obs_dim).tolist() for i in range(N)]
    rew_vals  = [float(i) * 0.1 for i in range(N)]
    done_vals = [i == N - 1 for i in range(N)]

    store = _make_store()
    for i in range(N):
        store.append(steps=[i] * 4, names=["action", "observation", "reward", "done"],
                     values=[act_vals[i], np.array(obs_vals[i], dtype=np.float32),
                             rew_vals[i], done_vals[i]])

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
    obs_vals = [[float(i), float(i + 1)] for i in range(N)]
    act_vals = [i % 4 for i in range(N)]
    for i in range(N):
        store.append(steps=[i], names=["action"], values=[act_vals[i]])
        store.append(steps=[i, i, i], names=["observation", "reward", "done"],
                     values=[np.array(obs_vals[i], dtype=np.float32), float(i), False])

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

    store.append(steps=[0] * 4, names=["action", "observation", "reward", "done"],
                 values=[0, obs3, 0.1, False])
    store.append(steps=[1] * 4, names=["action", "observation", "reward", "done"],
                 values=[1, obs5, 0.2, True])

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


def test_empty_store_returns_empty_dataset():
    store = _make_store()
    ds = store.to_dataset()
    assert len(ds) == 0


def test_partial_fields_produce_valid_dataset():
    """to_dataset works with any subset of the declared fields."""
    store = _make_store()
    store.append(steps=[0, 0, 0], names=["observation", "reward", "done"],
                 values=[np.zeros(3), 0.5, False])
    ds = store.to_dataset()
    assert len(ds) == 1
    assert "observation" in ds.column_names
    # action was not written for this step — it is absent from the yielded row
    assert ds[0].get("action") is None


def test_from_dataset_grows_beyond_initial_capacity():
    ds = _make_dataset(num_rows=100, obs_dim=OBS_DIM)
    store = _make_store(capacity=10)
    store.from_dataset(_core_columns(ds))
    assert len(store) == 100 * (OBS_DIM + 3)


def test_zero_capacity_raises():
    with pytest.raises(ValueError):
        StreamStore(capacity=0)


def test_duplicate_fields_raises():
    with pytest.raises(ValueError, match="duplicate"):
        StreamStore().define_fields(["a", "b", "a"])


def test_empty_fields_raises():
    with pytest.raises(ValueError):
        StreamStore().define_fields([])


def test_clear_resets_state():
    store = _make_store()
    store.append(steps=[0] * 4, names=["action", "observation", "reward", "done"],
                 values=[1, np.array([1.0, 2.0]), 0.5, False])
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
    assert recovered.column_names == ["action", "observation", "reward", "done"]


# ---------------------------------------------------------------------------
# 8. Performance benchmarks
# ---------------------------------------------------------------------------


def test_from_dataset_performance():
    N = 50_000
    obs_dim = 4
    ds = Dataset.from_dict(
        {
            "action": np.random.randint(0, 3, N).tolist(),
            "observation": np.random.randn(N, obs_dim).astype(np.float32).tolist(),
            "reward": np.random.randn(N).astype(np.float32).tolist(),
            "done": [bool(i % 100 == 99) for i in range(N)],
        },
        features=Features(
            {
                "action": Value("int64"),
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
            }
        ),
    )
    store = StreamStore(capacity=N * 10)
    store.define_fields(FIELDS, field_features=_STORE_FEATURES)
    t0 = time.time()
    store.from_dataset(ds)
    assert time.time() - t0 < 15.0, f"from_dataset too slow: {time.time() - t0:.2f}s"


def test_to_dataset_performance():
    N = 50_000
    obs_dim = 4
    ds = Dataset.from_dict(
        {
            "action": np.random.randint(0, 3, N).tolist(),
            "observation": np.random.randn(N, obs_dim).astype(np.float32).tolist(),
            "reward": np.random.randn(N).astype(np.float32).tolist(),
            "done": [bool(i % 100 == 99) for i in range(N)],
        },
        features=Features(
            {
                "action": Value("int64"),
                "observation": HFSequence(Value("float32")),
                "reward": Value("float32"),
                "done": Value("bool"),
            }
        ),
    )
    store = StreamStore(capacity=N * (obs_dim + 3) + 1000)
    store.define_fields(FIELDS, field_features=_STORE_FEATURES)
    store.from_dataset(ds)
    t0 = time.time()
    _ = store.to_dataset()
    assert time.time() - t0 < 10.0, f"to_dataset too slow: {time.time() - t0:.2f}s"
