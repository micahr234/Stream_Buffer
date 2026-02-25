"""Lance-backed token stream storage and HuggingFace dataset conversion.

LanceStreamStore holds a flat token stream using the Lance columnar format.
Fields are declared via ``define_fields`` before any data is written; each
field maps to an integer type ID stored per token.

Lance dataset columns
---------------------
value  int64  raw bit pattern of the token value
type   int64  field type ID assigned by define_fields
step   int64  caller-supplied step index (groups tokens into dataset rows)

Lance stores each column separately on disk, so reads that only need values
skip the type/step columns entirely — achieving the same I/O separation as
the split-buffer design, with built-in compression and cloud-compatible
persistence.

Writes are buffered in memory and flushed to Lance in batches to avoid
creating many small fragments.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Sequence, cast

import lance
import numpy as np
import pyarrow as pa
from datasets import Dataset, Features, Sequence as HFSequence, Value
from datasets import load_dataset as hf_load_dataset

_LANCE_SCHEMA = pa.schema([
    pa.field("value", pa.int64()),
    pa.field("type", pa.int64()),
    pa.field("step", pa.int64()),
])


def _leaf_dtype(feature: Any) -> str:
    """Return the HuggingFace leaf dtype string for a Value or Sequence feature."""
    return feature.feature.dtype if isinstance(feature, HFSequence) else feature.dtype


def _decode(raw: Any, dtype_str: str) -> Any:
    """Decode a stored int64 to the value indicated by dtype_str."""
    if np.issubdtype(np.dtype(dtype_str), np.floating):
        return np.array([raw], dtype=np.int64).view(np.float64)[0]
    return raw


class LanceStreamStore:
    """Flat token stream backed by a Lance dataset.

    Call ``define_fields`` before ``append`` or any dataset I/O.

    Writes are buffered in memory and flushed to Lance when the buffer
    exceeds ``capacity`` tokens.  Reads serve from both the flushed Lance
    data and the pending buffer without forcing a flush, so no unnecessary
    fragments are created."""

    def __init__(self, capacity: int = 1_000_000, path: str | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._field_to_type:   dict[str, int] = {}
        self._type_to_field:   dict[int, str] = {}
        self._field_features:  dict[str, Any] = {}
        self._field_dtype_str: dict[str, str] = {}

        self._owns_path = path is None
        if path is None:
            self._dir = tempfile.mkdtemp()
            self._path = os.path.join(self._dir, "stream.lance")
        else:
            self._dir = os.path.dirname(os.path.abspath(path))
            self._path = path

        self._ds: lance.LanceDataset | None = None
        self._flushed_count: int = 0
        self._flush_threshold: int = capacity

        self._pending_values: list[int] = []
        self._pending_types:  list[int] = []
        self._pending_steps:  list[int] = []

        self._prev_step: int = -1
        self._prev_type_id: int = -1

    def close(self) -> None:
        """Release the Lance dataset and clean up temp files if owned."""
        self._ds = None
        if self._owns_path and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lance flush
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        if not self._pending_values:
            return
        table = pa.table(
            {
                "value": pa.array(self._pending_values, type=pa.int64()),
                "type":  pa.array(self._pending_types,  type=pa.int64()),
                "step":  pa.array(self._pending_steps,  type=pa.int64()),
            },
            schema=_LANCE_SCHEMA,
        )
        if self._ds is None:
            self._ds = lance.write_dataset(table, self._path)
        else:
            self._ds = lance.write_dataset(table, self._path, mode="append")
        self._flushed_count += len(self._pending_values)
        self._pending_values.clear()
        self._pending_types.clear()
        self._pending_steps.clear()

    # ------------------------------------------------------------------
    # Field definitions
    # ------------------------------------------------------------------

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
        if self._flushed_count > 0 or self._pending_values:
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
        return self._flushed_count + len(self._pending_values)

    def __repr__(self) -> str:
        size = len(self)
        tail_n = min(size, 5)
        if tail_n:
            positions = np.arange(size - tail_n, size)
            result = self[positions]
            rows = []
            for i in range(tail_n):
                field = self._type_to_field.get(int(result["types"][i]), "")
                val = _decode(int(result["values"][i]), self._field_dtype_str.get(field, "int64"))
                rows.append(f"(step={int(result['step'][i])}, type={int(result['types'][i])}, val={val:.4g})")
            tail_str = f", tail=[{', '.join(rows)}]"
        else:
            tail_str = ""
        return f"LanceStreamStore(size={size}{tail_str})"

    def __getitem__(self, indices: Any) -> dict[str, np.ndarray]:
        """Return token records at *indices* as ``{"step", "types", "values"}``.

        *indices* may be an integer array of any shape.  ``"values"`` contains
        raw int64 bit patterns; use ``_decode`` with the field's dtype to read.

        Reads serve from both the flushed Lance dataset and the in-memory
        pending buffer without forcing a flush.
        """
        indices = np.asarray(indices)
        shape = indices.shape
        flat = indices.ravel()
        n = len(flat)

        values = np.empty(n, dtype=np.int64)
        types  = np.empty(n, dtype=np.int64)
        steps  = np.empty(n, dtype=np.int64)

        flushed_mask = flat < self._flushed_count

        if flushed_mask.any() and self._ds is not None:
            lance_idx = flat[flushed_mask].tolist()
            table = self._ds.take(lance_idx, columns=["value", "type", "step"])
            values[flushed_mask] = table.column("value").to_numpy()
            types[flushed_mask]  = table.column("type").to_numpy()
            steps[flushed_mask]  = table.column("step").to_numpy()

        pending_mask = ~flushed_mask
        if pending_mask.any():
            pv = np.array(self._pending_values, dtype=np.int64)
            pt = np.array(self._pending_types,  dtype=np.int64)
            ps = np.array(self._pending_steps,  dtype=np.int64)
            local = flat[pending_mask] - self._flushed_count
            values[pending_mask] = pv[local]
            types[pending_mask]  = pt[local]
            steps[pending_mask]  = ps[local]

        return {
            "step":   steps.reshape(shape).copy(),
            "types":  types.reshape(shape).copy(),
            "values": values.reshape(shape).copy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fields(self) -> None:
        if not self._field_to_type:
            raise RuntimeError("Fields not defined. Call define_fields() first.")

    def _write(self, step: int, type_id: int, data: np.ndarray) -> None:
        n = len(data)
        self._pending_values.extend(data.tolist())
        self._pending_types.extend([type_id] * n)
        self._pending_steps.extend([step] * n)
        if len(self._pending_values) >= self._flush_threshold:
            self._flush()

    def _clear(self) -> None:
        self._pending_values.clear()
        self._pending_types.clear()
        self._pending_steps.clear()
        self._flushed_count = 0
        self._prev_step = -1
        self._prev_type_id = -1
        self._ds = None
        if os.path.exists(self._path):
            shutil.rmtree(self._path, ignore_errors=True)

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

        prev_step, prev_type_id = self._prev_step, self._prev_type_id

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

        self._prev_step, self._prev_type_id = prev_step, prev_type_id

    # ------------------------------------------------------------------
    # Dataset I/O  (from_dataset / to_dataset)
    # ------------------------------------------------------------------

    def from_dataset(self, ds: Dataset) -> None:
        """Load a HuggingFace Dataset object into the store.

        Rows are assigned step indices 0, 1, 2, ... in the order they appear in
        ``ds`` -- sort before calling if a specific step order is required.
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
        incrementally -- peak RAM is independent of buffer size.
        """
        self._require_fields()
        features = Features(self._field_features)
        total = len(self)
        if total == 0:
            return Dataset.from_dict({f: [] for f in self._field_features}, features=features)

        self._flush()

        def gen() -> Any:
            current_step: int | None = None
            row: dict[str, list[Any]] = {}

            for batch in self._ds.to_batches():  # type: ignore[union-attr]
                batch_values = batch.column("value").to_numpy()
                batch_types  = batch.column("type").to_numpy()
                batch_steps  = batch.column("step").to_numpy()

                for i in range(len(batch)):
                    step    = int(batch_steps[i])
                    type_id = int(batch_types[i])
                    field   = self._type_to_field[type_id]
                    val     = _decode(int(batch_values[i]), self._field_dtype_str[field])

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

        return cast(Dataset, Dataset.from_generator(gen, features=features))

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
        ds = hf_load_dataset(dataset_name, split=dataset_split, download_mode="force_redownload")
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
