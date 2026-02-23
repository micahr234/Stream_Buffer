"""Token stream storage and HuggingFace dataset conversion.

StreamStore holds a flat token stream in a single MemmapBuffer whose elements
are a structured numpy dtype — one record per token, all channels stored
together.  The buffer grows dynamically; pass ``capacity`` to control the
initial allocation chunk size (default 1 M tokens).

Design
------
The caller supplies a list of **fields** at construction.  Type IDs are
auto-assigned as consecutive integers (0, 1, 2, …) in the order the fields are
listed.  Every token written to the buffer carries that ID in its ``type``
field so the store can identify fields across the full buffer without storing
names repeatedly.

The caller also supplies a **step index** when calling ``append`` — the store
does not manage a counter internally.  All tokens produced by one ``append``
call receive the same step value.  Split appends (writing different fields for
the same step in separate calls) are fine as long as both calls pass the same
``step`` value.

Token record layout (_TOKEN_DTYPE)
------------------------------------
Field   dtype     Content
------  -------   -------------------------------------------------------
step    int64     Caller-supplied step index
type    int64     Auto-assigned field type ID (index in the fields list)
data    int64     Raw bits of a float64 value (reinterpret via .view(float64))

All values — ints, floats, bools — are cast to float64 and stored as their
raw int64 bit pattern.  On read, the bits are reinterpreted back to float64
with ``.view(np.float64)``; callers then cast to the appropriate Python type.
float64 represents all integers exactly up to 2^53.

``__getitem__`` returns raw int64 bit patterns in the ``"values"`` array.
Callers reinterpret as float64 via ``.view(np.float64)`` when needed.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
from datasets import Dataset, Features, Sequence as HFSequence, Value, load_dataset
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
    """Flat token stream backed by a dynamically-growing MemmapBuffer.

    Internally holds a single structured buffer of ``_TOKEN_DTYPE`` records —
    one record per token.  The buffer grows in ``capacity``-sized chunks as
    needed; ``capacity`` is the growth increment, not a hard cap."""

    def __init__(
        self,
        fields: list[str],
        capacity: int = 1_000_000,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        if not fields:
            raise ValueError("fields must not be empty.")
        if len(fields) != len(set(fields)):
            raise ValueError("fields must not contain duplicates.")
        self._field_to_type: dict[str, int] = {f: i for i, f in enumerate(fields)}
        self._type_to_field: dict[int, str] = {i: f for i, f in enumerate(fields)}
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
        Returns ``{"types": int64[...], "values": int64[...]}`` with batch
        dimensions matching the index shape.  ``"values"`` contains the raw
        int64 bit patterns stored in the buffer — callers reinterpret as
        float64 via ``.view(np.float64)`` when needed.
        """
        idx: np.ndarray = np.asarray(indices)
        chunk = self._buf[idx]
        # Field extraction from a structured array gives strides equal to the
        # struct itemsize, not the field's element size; .copy() produces a
        # fresh contiguous array with normal element-size strides.
        return {
            "step": chunk["step"].copy(),
            "types": chunk["type"].copy(),
            "values": chunk["data"].copy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, step: int, type_id: int, data: np.ndarray) -> None:
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
    # Online append
    # ------------------------------------------------------------------

    def append(
        self,
        steps: Sequence[int],
        names: Sequence[str],
        values: Sequence[Any],
    ) -> None:
        """Append tokens to the store.

        Each position in the three parallel sequences describes one field
        write: ``steps[i]`` is the step index, ``names[i]`` is the field
        name, and ``values[i]`` is the scalar or 1-D array to store.

        All three sequences must have equal length.  Within a given step,
        field names must appear in ``fields`` construction order; fields may
        be omitted and steps may repeat, but out-of-order fields for the
        same step raise ``ValueError``.
        """
        if not steps:
            raise ValueError("At least one token required.")
        if len(steps) != len(names) or len(steps) != len(values):
            raise ValueError(
                f"steps, names, and values must all have equal length "
                f"(got {len(steps)}, {len(names)}, {len(values)})."
            )

        if len(self._buf) > 0:
            last = self._buf[len(self._buf) - 1]
            prev_step, prev_type_id = int(last["step"]), int(last["type"])
        else:
            prev_step, prev_type_id = -1, -1
        
        for step, name, value in zip(steps, names, values):
            if name not in self._field_to_type:
                raise ValueError(f"Unknown field {name!r}.")
            type_id = self._field_to_type[name]
            if (step < prev_step) or (step == prev_step and type_id <= prev_type_id):
                raise ValueError(
                    f"Fields must be in construction order"
                )
            prev_step, prev_type_id = step, type_id
            arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
            if arr.ndim != 1:
                raise ValueError(
                    f"Field {name!r} must be scalar or 1-D; got shape {arr.shape}."
                )
            self._write(step, type_id, arr.view(np.int64))

    # ------------------------------------------------------------------
    # In-memory Dataset I/O  (from_dataset / to_dataset)
    # ------------------------------------------------------------------

    def from_dataset(self, ds: Dataset) -> None:
        """Load a HuggingFace Dataset object into the store.

        Counterpart to ``to_dataset``.  Every column in ``ds`` is appended;
        ``append`` raises ``ValueError`` if any column name is not a declared
        field.  Rows are assigned step indices 0, 1, 2, … in the order they
        appear in ``ds`` — sort the dataset before calling if a specific step
        order is required.
        """
        self._clear()
        for i, row in enumerate(ds):
            row = cast(dict[str, Any], row)
            self.append(
                steps=[i] * len(row),
                names=list(row.keys()),
                values=list(row.values()),
            )

    def to_dataset(
        self,
        field_features: dict[str, Any] | None = None,
        extra: dict[str, list[Any]] | None = None,
    ) -> Dataset:
        """Decode the token stream to a HuggingFace Dataset object.

        Counterpart to ``from_dataset``.  Each unique ``step`` value becomes
        one row.  Column order: ``step_id``, field columns in construction
        order, then any ``extra`` columns.

        A field is stored as a variable-length sequence if any step produces
        more than one token for it; otherwise it is stored as a scalar.
        Pass ``field_features`` to override the inferred HuggingFace feature
        type for any field.  Unspecified scalar fields default to
        ``Value("float64")``; unspecified variable-length fields default to
        ``Sequence(Value("float32"))``.

        Use ``extra`` for any per-step columns that are not part of the token
        stream — provenance, episode numbers, timestamps, etc.
        """
        if len(self._buf) == 0:
            raise ValueError("Store is empty.")

        tokens  = self._buf[:]
        type_np = tokens["type"]
        step_np = tokens["step"]
        vals_np = tokens["data"]

        unique_steps = np.unique(step_np)
        num_steps    = len(unique_steps)

        dataset_dict: dict[str, Any]  = {}
        features_dict: dict[str, Any] = {}

        ff = field_features or {}

        for field_name, type_id in self._field_to_type.items():
            mask = type_np == type_id
            if not mask.any():
                continue

            field_vals  = vals_np[mask]
            field_steps = step_np[mask]

            _, counts   = np.unique(field_steps, return_counts=True)
            is_variable = int(counts.max()) > 1

            if is_variable:
                bounds     = np.concatenate(
                    [[0], np.flatnonzero(np.diff(field_steps)) + 1, [field_steps.size]]
                )
                num_groups = len(bounds) - 1
                if num_groups != num_steps:
                    raise ValueError(
                        f"Field {field_name!r}: {num_groups} token groups but "
                        f"{num_steps} steps. Every step must have at least one "
                        f"token for this field."
                    )
                field_data: Any = [
                    field_vals[int(bounds[i]):int(bounds[i + 1])].astype(np.float32).tolist()
                    for i in range(num_groups)
                ]
                feature: Any = ff.get(field_name, HFSequence(Value("float32")))
            else:
                field_data = field_vals.tolist()
                feature    = ff.get(field_name, Value("float64"))

            dataset_dict[field_name]  = field_data
            features_dict[field_name] = feature

        if extra:
            for col_name, col_values in extra.items():
                if len(col_values) != num_steps:
                    raise ValueError(
                        f"extra column {col_name!r}: expected {num_steps} values, "
                        f"got {len(col_values)}."
                    )
                dataset_dict[col_name]  = col_values
                features_dict[col_name] = _infer_hf_feature(col_values[0])

        return Dataset.from_dict(dataset_dict, features=Features(features_dict))

    # ------------------------------------------------------------------
    # Hub I/O  (load_dataset / save_dataset)
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_name: str,
        dataset_split: str,
        sort_by: list[str] | None = None,
    ) -> None:
        """Load a HuggingFace Hub split into the store.

        Counterpart to ``save_dataset``.  Downloads the named split, optionally
        sorts it (pass ``sort_by`` with the column names that define step order
        for your schema), strips non-field columns, then calls ``from_dataset``.
        """
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        ds = load_dataset(dataset_name, split=dataset_split, download_mode="force_redownload")
        if sort_by:
            ds = ds.sort(sort_by, reverse=[False] * len(sort_by))
        canonical = list(self._field_to_type.keys())
        present = [c for c in canonical if c in ds.column_names]
        ds = ds.select_columns(present).cast(Features({c: ds.features[c] for c in present}))
        self.from_dataset(ds)

    def save_dataset(
        self,
        dataset_name: str,
        dataset_split: str,
        field_features: dict[str, Any] | None = None,
        extra: dict[str, list[Any]] | None = None,
    ) -> None:
        """Push the token stream to a HuggingFace Hub dataset.

        Counterpart to ``load_dataset``.  Calls ``to_dataset`` then pushes the
        result to the Hub under ``dataset_name`` / ``dataset_split``.
        ``field_features`` and ``extra`` are forwarded to ``to_dataset``
        unchanged.
        """
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        ds = self.to_dataset(field_features=field_features, extra=extra)
        ds.push_to_hub(dataset_name, split=dataset_split)
