"""Token stream storage and HuggingFace dataset conversion.

StreamStore holds a flat token stream in a single NumpyMemmapBuffer whose
elements are a structured numpy dtype — one record per token, all channels
stored together.  The buffer grows dynamically; pass ``capacity`` to control
the initial allocation chunk size (default 1 M tokens).

Design
------
The caller supplies a **step index** when calling ``append`` — the store does
not manage a counter internally.  All tokens produced by one ``append`` call
receive the same step value.

**Key invariant**: each step value is unique and corresponds to exactly one
row in the HuggingFace dataset.  This lets ``to_dataset`` extract scalar
fields (action, reward, done) with a single type mask across the whole
buffer rather than per-step loops.

    Typical field-call order per RL step:
        step = k : ACTION   — one token  (random on the very first step)
        step = k : OBS      — one token per observation dimension (variable!)
        step = k : REWARD   — one token
        step = k : DONE     — one token

    (Split appends — e.g. action first, then obs/reward/done — are fine as
    long as both calls pass the same ``step`` value.)

Because the ``type`` field records the token type and step values group tokens
by environment step, ``to_dataset`` can reconstruct any step layout using
simple type masks — no knowledge of ``obs_dim`` is needed in advance.

Token record layout (_TOKEN_DTYPE)
------------------------------------
Field   dtype     Content
------  -------   -------------------------------------------------------
step    int64     Caller-supplied environment step index
type    int64     TokenType enum value
data    int64     Raw bits of a float64 value (reinterpret via .view(float64))

All values — ints (done, action) and floats (obs, reward) — are
cast to float64 and stored as their raw int64 bit pattern.  On read, the
bits are reinterpreted back to float64 with ``.view(np.float64)``; callers
then cast to the appropriate Python type.  float64 represents all int64
values exactly up to 2^53, which covers all realistic RL action spaces.

``__getitem__`` returns a dict of numpy arrays; callers convert to torch if needed.

**Constraint on ``to_dataset``**: every step that has an action token must also
have at least one observation token, and vice versa.  Mismatches raise
``ValueError`` rather than silently producing wrong data.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from datasets import Dataset, Features, Sequence as HFSequence, Value, load_dataset
from definitions import TokenType
from memmap_buffer import MemmapBuffer

# Structured dtype: one record per token.
# All values are stored as float64 raw bits in `data` (lossless for ints up to 2^53).
_TOKEN_DTYPE = np.dtype([
    ("step", np.int64),
    ("type", np.int64),
    ("data", np.int64),
])


def _infer_hf_feature(sample: Any) -> Value:
    """Return the HuggingFace ``Value`` feature type for a Python scalar."""
    if isinstance(sample, bool):
        return Value("bool")
    if isinstance(sample, (int, np.integer)):
        return Value("int64")
    if isinstance(sample, (float, np.floating)):
        return Value("float32")
    return Value("string")


class StreamStore:
    """Flat token stream backed by a dynamically-growing NumpyMemmapBuffer.

    Internally holds a single structured buffer of ``_TOKEN_DTYPE`` records —
    one record per token.  The buffer grows in ``capacity``-sized chunks as
    needed; ``capacity`` is the growth increment, not a hard cap.

    ``__getitem__`` accepts an integer index array of **any shape** and returns
    a ``dict[str, np.ndarray]`` whose shape matches that of the index array —
    ``{"types": int64[...], "values": float64[...]}``.  Callers convert to
    torch tensors as needed.
    """

    def __init__(
        self,
        field_to_type: dict[str, int],
        env_name: str,
        env_number: int,
        capacity: int = 1_000_000,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._field_to_type = field_to_type
        self._type_to_field: dict[int, str] = {v: k for k, v in field_to_type.items()}
        self._env_name = env_name
        self._env_number = env_number

        # One structured buffer: each element is one token record (step, type, disc, cont).
        self._buf = MemmapBuffer(dtype=_TOKEN_DTYPE, size_increment=capacity)

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        size = len(self._buf)
        tail_n = min(size, 5)
        if tail_n:
            tail = self._buf[size - tail_n : size]
            rows = ", ".join(
                f"(step={r['step']}, type={r['type']}, "
                f"val={np.array([r['data']], dtype=np.int64).view(np.float64)[0]:.4g})"
                for r in tail
            )
            tail_str = f", tail=[{rows}]"
        else:
            tail_str = ""
        return f"StreamStore(size={size}{tail_str})"

    def __getitem__(self, indices: Any) -> dict[str, np.ndarray]:
        """Return token records at *indices* as a dict of numpy arrays.

        *indices* may be an integer array of **any shape** (numpy or array-like).
        Returns ``{"types": int64, "values": float64}`` with batch dimensions
        matching the index shape.
        """
        idx: np.ndarray = np.asarray(indices)
        tokens = self._buf[idx]   # structured array copy — one record per index
        # Field extraction from a structured array gives strides equal to the
        # struct itemsize, not the field's element size. .copy() produces a
        # fresh contiguous array; .view(float64) reinterprets the raw bits.
        return {
            "types":  tokens["type"].copy(),
            "values": tokens["data"].copy().view(np.float64),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, type_id: int, data: np.ndarray, step: int) -> None:
        """Append *n* tokens to the buffer.

        *data* is a 1-D int64 array of raw float64 bits (one element per token).
        All tokens receive the caller-supplied *step* value.
        """
        n = len(data)
        chunk = np.empty(n, dtype=_TOKEN_DTYPE)
        chunk["step"] = step
        chunk["type"] = type_id
        chunk["data"] = data
        self._buf.append(chunk)

    def _clear(self) -> None:
        self._buf.reset()

    # ------------------------------------------------------------------
    # Online append (environment rollouts / test runner)
    # ------------------------------------------------------------------

    def append(
        self,
        field_names: Sequence[str],
        field_values: Sequence[Any],
        step: int,
    ) -> None:
        """Append one or more step fields, tagging every token with *step*.

        Each element of *field_names* must be a key in ``field_to_type``
        (``"action"``, ``"observation"``, ``"reward"``, ``"done"``).
        Field lengths may vary freely — observation dimension is not locked.
        All tokens from this call share the same *step* value, so split
        appends (e.g. action first, then obs/reward/done) work correctly as
        long as both calls use the same *step*.
        """
        if len(field_names) != len(field_values):
            raise ValueError("field_names and field_values must have equal length.")
        if not field_names:
            raise ValueError("At least one field required.")

        for key, value in zip(field_names, field_values):
            if key not in self._field_to_type:
                raise ValueError(f"Unknown field {key!r}.")
            type_id = self._field_to_type[key]
            arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
            if arr.ndim != 1:
                raise ValueError(
                    f"Field {key!r} must be scalar or 1-D; got shape {arr.shape}."
                )
            self._write(type_id, arr.view(np.int64), step)

    # ------------------------------------------------------------------
    # HuggingFace Dataset I/O
    # ------------------------------------------------------------------

    def from_dataset(self, ds: Dataset) -> None:
        """Load a HuggingFace Dataset into the store (only known fields).

        Fields are appended in ``field_to_type`` key order (the canonical
        per-step token order) regardless of the dataset's column order.
        """
        self._clear()

        for i, row in enumerate(ds):
            field_names = list(row.keys())
            field_values = list(row.values())
            self.append(
                field_names=field_names,
                field_values=field_values,
                step=i,
            )

    def load_dataset(self, dataset_name: str | None, dataset_split: str) -> None:
        """Load a named HuggingFace split; sorts by env_number, env_name, step_id when present."""
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        ds = load_dataset(dataset_name, split=dataset_split, download_mode="force_redownload")
        ds = ds.sort(["env_name", "env_number", "step_id"], reverse=[False, False, False])
        canonical = list(self._field_to_type.keys())
        ds = ds.select_columns(canonical).cast(Features({c: ds.features[c] for c in canonical}))
        self.from_dataset(ds)

    def to_dataset(self, extra: dict[str, list[Any]] | None = None) -> Dataset:
        """Decode the token stream to a HuggingFace Dataset.

        Assumes each unique ``step`` value corresponds to exactly one dataset
        row.  Scalar fields (action, reward, done) are extracted with a single
        type mask across the full buffer — no per-step loop.  Observations are
        the only field sliced per step because their length may vary.
        ``extra`` adds caller-supplied per-step columns verbatim.

        Note that this will eventially need to be updated. We do not want to hardcode the field names here.
        """
        if len(self._buf) == 0:
            raise ValueError("Store is empty.")

        tokens  = self._buf[:]          # structured numpy view, shape [len(self._buf)]
        type_np = tokens["type"]
        step_np = tokens["step"]
        # .copy() makes data contiguous (structured-array field strides = struct
        # itemsize, not element size); .view(float64) reinterprets the raw bits.
        vals_np = tokens["data"].copy().view(np.float64)

        f2t = self._type_to_field
        obs_field    = f2t[int(TokenType.OBS)]
        reward_field = f2t[int(TokenType.REWARD)]
        done_field   = f2t[int(TokenType.DONE)]
        action_field = f2t[int(TokenType.ACTION)]

        # Scalar fields: one token per step — extract globally by type mask.
        action_mask = type_np == int(TokenType.ACTION)
        actions = vals_np[action_mask].astype(np.int64)
        step_ids = step_np[action_mask].astype(np.int64)
        num_steps = len(actions)
        if num_steps == 0:
            raise ValueError(
                f"No complete steps found (no {action_field!r} tokens in store). "
                f"Call append with {action_field!r} field before to_dataset."
            )
        rewards = vals_np[type_np == int(TokenType.REWARD)].astype(np.float32).tolist()
        dones   = vals_np[type_np == int(TokenType.DONE)].astype(bool).tolist()
        actions = actions.tolist()

        # Observations: variable length — group consecutive obs tokens by step.
        obs_mask = type_np == int(TokenType.OBS)
        obs_vals = vals_np[obs_mask]
        obs_step = step_np[obs_mask]
        obs_bounds = np.concatenate(
            [[0], np.flatnonzero(np.diff(obs_step)) + 1, [obs_step.size]]
        )
        num_obs_groups = len(obs_bounds) - 1
        if num_obs_groups != num_steps:
            raise ValueError(
                f"Observation group count ({num_obs_groups}) does not match step count "
                f"({num_steps} action tokens). Every step must have at least one observation "
                f"token. Check that append() is called with 'observation' for every step."
            )
        observations: list[list[float]] = [
            obs_vals[int(obs_bounds[i]):int(obs_bounds[i + 1])].astype(np.float32).tolist()
            for i in range(num_steps)
        ]

        dataset_dict: dict[str, Any] = {
            "env_name":   [self._env_name] * num_steps,
            "env_number": [self._env_number] * num_steps,
            "step_id":    step_ids.tolist(),
            action_field: actions,
            obs_field:    observations,
            reward_field: rewards,
            done_field:   dones,
        }
        features_dict: dict[str, Any] = {
            "env_name":   Value("string"),
            "env_number": Value("int64"),
            "step_id":    Value("int64"),
            action_field: Value("int64"),
            obs_field:    HFSequence(Value("float32")),
            reward_field: Value("float32"),
            done_field:   Value("bool"),
        }

        # Caller-supplied extra columns (e.g. string env_id, params JSON).
        if extra:
            for col_name, col_values in extra.items():
                if len(col_values) != num_steps:
                    raise ValueError(
                        f"extra column {col_name!r}: expected {num_steps} values, "
                        f"got {len(col_values)}."
                    )
                dataset_dict[col_name] = col_values
                features_dict[col_name] = _infer_hf_feature(col_values[0])

        return Dataset.from_dict(dataset_dict, features=Features(features_dict))
