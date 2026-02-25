"""Token stream storage and HuggingFace dataset conversion.

StreamStore holds a flat token stream in a memory-mapped buffer.  Fields are
declared via ``define_fields`` before any data is written; each field maps to
an integer type ID stored with every token.

Token record layout
-------------------
step  int64  caller-supplied step index (groups tokens into dataset rows)
type  int64  field type ID assigned by define_fields
data  int64  raw bit pattern of the value; interpreted via the field's feature

Float fields store the float64 bit pattern as int64.  Integer and bool fields
store the int64 value directly.  The HuggingFace feature type determines which
encoding is used on write and which reinterpretation is applied on read.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
from datasets import Dataset, Features, Sequence as HFSequence, Value, load_dataset
from memmap_buffer import MemmapBuffer

_TOKEN_DTYPE = np.dtype([
    ("step", np.int64),
    ("type", np.int64),
    ("data", np.int64),
])


def _leaf_dtype(feature: Any) -> str:
    """Return the HuggingFace leaf dtype string for a Value or Sequence feature."""
    return feature.feature.dtype if isinstance(feature, HFSequence) else feature.dtype


def _decode(raw: Any, dtype_str: str) -> Any:
    """Decode a stored int64 to the value indicated by dtype_str."""
    if np.issubdtype(np.dtype(dtype_str), np.floating):
        return np.array([raw], dtype=np.int64).view(np.float64)[0]
    return raw


class StreamStore:
    """Flat token stream backed by a dynamically-growing MemmapBuffer.

    Call ``define_fields`` before ``append`` or any dataset I/O.
    The buffer grows in ``capacity``-sized chunks as needed."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._field_to_type:   dict[str, int] = {}
        self._type_to_field:   dict[int, str] = {}
        self._field_features:  dict[str, Any] = {}
        self._field_dtype_str: dict[str, str] = {}
        self._buf = MemmapBuffer(dtype=_TOKEN_DTYPE, size_increment=capacity)

    def define_fields(
        self,
        fields: list[str],
        field_features: dict[str, Any] | None = None,
    ) -> None:
        """Set the field-to-type mapping and HuggingFace feature types.

        ``field_features`` maps field names to HuggingFace ``Value`` or
        ``Sequence`` objects used by ``to_dataset``.  Any field not listed
        defaults to ``Sequence(Value("int64"))``.

        Raises ``RuntimeError`` if data has already been written, since
        redefining fields would corrupt existing token type IDs.
        """
        if len(self._buf) > 0:
            raise RuntimeError("Cannot redefine fields after data has been appended.")
        if not fields:
            raise ValueError("fields must not be empty.")
        if len(fields) != len(set(fields)):
            raise ValueError("fields must not contain duplicates.")
        ff = dict(field_features) if field_features else {}
        self._field_to_type   = {f: i for i, f in enumerate(fields)}
        self._type_to_field   = {i: f for i, f in enumerate(fields)}
        self._field_features  = {f: ff.get(f, HFSequence(Value("int64"))) for f in fields}
        self._field_dtype_str = {f: _leaf_dtype(feat) for f, feat in self._field_features.items()}

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        size = len(self._buf)
        tail_n = min(size, 5)
        if tail_n:
            rows = []
            for r in self._buf[size - tail_n : size]:
                field = self._type_to_field.get(r["type"], "")
                val   = _decode(r["data"], self._field_dtype_str.get(field, "int64"))
                rows.append(f"(step={r['step']}, type={r['type']}, val={val:.4g})")
            tail_str = f", tail=[{', '.join(rows)}]"
        else:
            tail_str = ""
        return f"StreamStore(size={size}{tail_str})"

    def __getitem__(self, indices: Any) -> dict[str, np.ndarray]:
        """Return token records at *indices* as ``{"step", "types", "values"}``.

        *indices* may be an integer array of any shape.  ``"values"`` contains
        raw int64 bit patterns; use ``_decode`` with the field's dtype to read.
        """
        chunk = self._buf[np.asarray(indices)]
        # .copy() produces contiguous arrays with normal element-size strides.
        return {
            "step":   chunk["step"].copy(),
            "types":  chunk["type"].copy(),
            "values": chunk["data"].copy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fields(self) -> None:
        if not self._field_to_type:
            raise RuntimeError("Fields not defined. Call define_fields() first.")

    def _write(self, step: int, type_id: int, data: np.ndarray) -> None:
        chunk = np.empty(len(data), dtype=_TOKEN_DTYPE)
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
        be omitted but those present must be strictly ordered.  Out-of-order
        fields for the same step raise ``ValueError``.
        """
        if not steps:
            raise ValueError("At least one token required.")
        if len(steps) != len(names) or len(steps) != len(values):
            raise ValueError(
                f"steps, names, and values must all have equal length "
                f"(got {len(steps)}, {len(names)}, {len(values)})."
            )
        self._require_fields()

        if len(self._buf) > 0:
            last = self._buf[len(self._buf) - 1]
            prev_step, prev_type_id = int(last["step"]), int(last["type"])
        else:
            prev_step, prev_type_id = -1, -1

        for step, name, value in zip(steps, names, values):
            if name not in self._field_to_type:
                raise ValueError(f"Unknown field {name!r}.")
            type_id = self._field_to_type[name]
            if step < prev_step or (step == prev_step and type_id <= prev_type_id):
                raise ValueError("Fields must be in construction order.")
            prev_step, prev_type_id = step, type_id
            arr = np.atleast_1d(np.asarray(value))
            if arr.ndim != 1:
                raise ValueError(f"Field {value!r} must be scalar or 1-D; got shape {arr.shape}.")
            if arr.itemsize != 8:
                arr = arr.astype(np.float64 if np.issubdtype(arr.dtype, np.floating) else np.int64)
            self._write(step, type_id, arr.view(np.int64))

    # ------------------------------------------------------------------
    # Dataset I/O  (from_dataset / to_dataset)
    # ------------------------------------------------------------------

    def from_dataset(self, ds: Dataset) -> None:
        """Load a HuggingFace Dataset object into the store.

        Rows are assigned step indices 0, 1, 2, … in the order they appear in
        ``ds`` — sort before calling if a specific step order is required.
        If fields have not been defined yet, they are inferred from
        ``ds.column_names`` in dataset column order.
        """
        if not self._field_to_type:
            self.define_fields(list(ds.features.keys()), field_features=dict(ds.features))
        self._clear()
        for i, row in enumerate(ds):
            row = cast(dict[str, Any], row)
            self.append(
                steps=[i] * len(row),
                names=list(row.keys()),
                values=list(row.values()),
            )

    def to_dataset(self) -> Dataset:
        """Decode the token stream to a HuggingFace Dataset object.

        Counterpart to ``from_dataset``.  Each step becomes one row; each
        field is a list of values.  Feature types come from ``define_fields``;
        the default is ``Sequence(Value("float32"))``.

        Uses ``Dataset.from_generator`` so Arrow batches are written to disk
        incrementally — peak RAM is independent of buffer size.
        """
        self._require_fields()
        features = Features(self._field_features)
        if len(self._buf) == 0:
            return Dataset.from_dict({f: [] for f in self._field_features}, features=features)

        def gen() -> Any:
            current_step: int | None = None
            row: dict[str, list[Any]] = {}
            for token in self._buf:
                step  = token["step"]
                field = self._type_to_field[token["type"]]
                val   = _decode(token["data"], self._field_dtype_str[field])
                if step != current_step:
                    if current_step is not None:
                        yield {
                            f: (vs if isinstance(self._field_features[f], HFSequence) else vs[0])
                            for f, vs in row.items()
                        }
                    current_step = step
                    row = {}
                row.setdefault(field, []).append(val)
            if current_step is not None:
                yield {
                    f: (vs if isinstance(self._field_features[f], HFSequence) else vs[0])
                    for f, vs in row.items()
                }

        return Dataset.from_generator(gen, features=features)

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
        sorts it (pass ``sort_by`` with the column names that define step order),
        then calls ``from_dataset``.  Filter or rename columns on the dataset
        before calling if needed.
        """
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        ds = load_dataset(dataset_name, split=dataset_split, download_mode="force_redownload")
        if sort_by:
            ds = ds.sort(sort_by, reverse=[False] * len(sort_by))
        self.from_dataset(ds)

    def save_dataset(self, dataset_name: str, dataset_split: str) -> None:
        """Push the token stream to a HuggingFace Hub dataset.

        Counterpart to ``load_dataset``.  Calls ``to_dataset`` then pushes to
        the Hub under ``dataset_name`` / ``dataset_split``.
        """
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        self.to_dataset().push_to_hub(dataset_name, split=dataset_split)
