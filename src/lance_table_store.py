"""Lance-backed tabular store with flatten-on-read token stream access.

LanceTableStore keeps data in its natural tabular form — one Lance row per
step, typed columns per field.  The flat token stream is reconstructed on the
fly during ``__getitem__`` by fetching the relevant step rows and walking
their fields.

This plays to Lance's strengths: far fewer rows to ``take()``, typed columnar
compression, and near-passthrough ``from_dataset`` / ``to_dataset``.

An in-memory step-offsets array (one int64 per step) maps flat token indices
to step ordinals via ``np.searchsorted``.  Variable-length fields are handled
naturally — each step's token count is derived from the actual field data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Sequence

import lance
import numpy as np
import pyarrow as pa
from datasets import Dataset, Features, Sequence as HFSequence, Value
from datasets import load_dataset as hf_load_dataset


def _leaf_dtype(feature: Any) -> str:
    """Return the HuggingFace leaf dtype string for a Value or Sequence feature."""
    return feature.feature.dtype if isinstance(feature, HFSequence) else feature.dtype


def _decode(raw: Any, dtype_str: str) -> Any:
    """Decode a stored int64 to the value indicated by dtype_str."""
    if np.issubdtype(np.dtype(dtype_str), np.floating):
        return np.array([raw], dtype=np.int64).view(np.float64)[0]
    return raw


class LanceTableStore:
    """Tabular token store backed by a Lance dataset.

    Call ``define_fields`` before ``append`` or any dataset I/O.

    Internally, each step is one Lance row with a column per field.  Values
    are stored in their natural types — float fields as ``float64``, integer
    and boolean fields as ``int64``.  Encoding to int64 bit patterns only
    happens at the ``__getitem__`` boundary when the flat token stream is
    produced.

    Writes accumulate step rows in memory and flush to Lance when the
    pending token count exceeds ``capacity``.  Reads flatten the relevant
    step rows into the ``{"step", "types", "values"}`` token stream on
    the fly."""

    def __init__(self, capacity: int = 1_000_000, path: str | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._field_to_type:   dict[str, int] = {}
        self._type_to_field:   dict[int, str] = {}
        self._field_features:  dict[str, Any] = {}
        self._field_dtype_str: dict[str, str] = {}
        self._lance_schema: pa.Schema | None = None

        self._owns_path = path is None
        if path is None:
            self._dir = tempfile.mkdtemp()
            self._path = os.path.join(self._dir, "table.lance")
        else:
            self._dir = os.path.dirname(os.path.abspath(path))
            self._path = path

        self._ds: lance.LanceDataset | None = None
        self._flush_threshold: int = capacity

        self._flushed_step_count: int = 0
        self._pending_rows: list[dict[str, list[Any]]] = []
        self._pending_token_count: int = 0

        self._current_step_value: int | None = None
        self._current_row: dict[str, list[Any]] = {}

        self._step_values: list[int] = []
        self._step_offsets: list[int] = []
        self._total_tokens: int = 0

        self._prev_step: int = -1
        self._prev_type_id: int = -1

    def close(self) -> None:
        self._ds = None
        if self._owns_path and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Field definitions
    # ------------------------------------------------------------------

    def define_fields(
        self,
        fields: list[str],
        field_features: dict[str, Any] | None = None,
    ) -> None:
        """Set the field-to-type mapping and HuggingFace feature types.

        Also builds the Lance PyArrow schema using native Arrow types —
        float fields become ``float64``, everything else ``int64``.
        Sequence fields are wrapped in ``list<>``.
        """
        if self._total_tokens > 0:
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

        schema_fields = []
        for name in fields:
            dtype_str = self._field_dtype_str[name]
            if np.issubdtype(np.dtype(dtype_str), np.floating):
                arrow_scalar = pa.float64()
            elif dtype_str == "bool":
                arrow_scalar = pa.bool_()
            else:
                arrow_scalar = pa.int64()
            if isinstance(self._field_features[name], HFSequence):
                schema_fields.append(pa.field(name, pa.list_(arrow_scalar)))
            else:
                schema_fields.append(pa.field(name, arrow_scalar))
        self._lance_schema = pa.schema(schema_fields)

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_tokens

    def __repr__(self) -> str:
        size = self._total_tokens
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
        return f"LanceTableStore(size={size}{tail_str})"

    def __getitem__(self, indices: Any) -> dict[str, np.ndarray]:
        """Return token records at *indices* as ``{"step", "types", "values"}``.

        Flat token indices are mapped to step rows via ``np.searchsorted``
        on the step-offsets array.  The relevant step rows are fetched from
        Lance (batched into a single ``take()``) or from the in-memory
        pending buffer, then flattened into the token stream on the fly.
        """
        indices = np.asarray(indices)
        shape = indices.shape
        flat = indices.ravel()
        n = len(flat)

        offsets = np.array(self._step_offsets, dtype=np.int64)
        step_ordinals = np.searchsorted(offsets, flat, side="right") - 1
        offset_in_step = flat - offsets[step_ordinals]

        unique_ordinals, inverse = np.unique(step_ordinals, return_inverse=True)

        # --- fetch all needed step rows ---
        step_data: dict[int, dict[str, list[Any]]] = {}

        lance_mask = unique_ordinals < self._flushed_step_count
        lance_ordinals = unique_ordinals[lance_mask]
        if len(lance_ordinals) > 0 and self._ds is not None:
            table = self._ds.take(lance_ordinals.tolist())
            for i, ordinal in enumerate(lance_ordinals):
                step_data[int(ordinal)] = self._parse_lance_row(table, i)

        for ordinal in unique_ordinals:
            ordinal = int(ordinal)
            if ordinal in step_data:
                continue
            pending_idx = ordinal - self._flushed_step_count
            if pending_idx < len(self._pending_rows):
                step_data[ordinal] = self._pending_rows[pending_idx]
            else:
                step_data[ordinal] = self._current_row

        # --- flatten each unique step into (types, values) arrays ---
        flat_cache: list[tuple[np.ndarray, np.ndarray]] = []
        for ordinal in unique_ordinals:
            row = step_data[int(ordinal)]
            types_list: list[int] = []
            vals_parts: list[np.ndarray] = []
            for field_name in self._field_to_type:
                if field_name not in row:
                    continue
                type_id = self._field_to_type[field_name]
                fdata = row[field_name]
                types_list.extend([type_id] * len(fdata))
                dtype_str = self._field_dtype_str[field_name]
                if np.issubdtype(np.dtype(dtype_str), np.floating):
                    vals_parts.append(
                        np.array(fdata, dtype=np.float64).view(np.int64)
                    )
                else:
                    vals_parts.append(np.array(fdata, dtype=np.int64))
            flat_cache.append((
                np.array(types_list, dtype=np.int64),
                np.concatenate(vals_parts) if vals_parts else np.empty(0, dtype=np.int64),
            ))

        # --- scatter into output arrays ---
        out_values = np.empty(n, dtype=np.int64)
        out_types  = np.empty(n, dtype=np.int64)
        out_steps  = np.empty(n, dtype=np.int64)

        for u_idx, ordinal in enumerate(unique_ordinals):
            mask = inverse == u_idx
            ofs = offset_in_step[mask]
            ft, fv = flat_cache[u_idx]
            out_types[mask]  = ft[ofs]
            out_values[mask] = fv[ofs]
            out_steps[mask]  = self._step_values[int(ordinal)]

        return {
            "step":   out_steps.reshape(shape).copy(),
            "types":  out_types.reshape(shape).copy(),
            "values": out_values.reshape(shape).copy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fields(self) -> None:
        if not self._field_to_type:
            raise RuntimeError("Fields not defined. Call define_fields() first.")

    def _parse_lance_row(self, table: pa.Table, row_idx: int) -> dict[str, list[Any]]:
        """Convert one row of a PyArrow table to the internal row format."""
        row: dict[str, list[Any]] = {}
        for field_name in self._field_to_type:
            col = table.column(field_name)
            val = col[row_idx]
            if not val.is_valid:
                continue
            if isinstance(self._field_features[field_name], HFSequence):
                row[field_name] = val.as_py()
            else:
                row[field_name] = [val.as_py()]
        return row

    def _write(self, step: int, type_id: int, data: np.ndarray) -> None:
        field_name = self._type_to_field[type_id]

        if self._current_step_value is not None and step != self._current_step_value:
            self._pending_rows.append(dict(self._current_row))
            prev_ordinal = len(self._step_values) - 1
            self._pending_token_count += self._total_tokens - self._step_offsets[prev_ordinal]
            self._current_row = {}
            self._current_step_value = None

            if self._pending_token_count >= self._flush_threshold:
                self._flush()

        if self._current_step_value is None:
            self._step_values.append(step)
            self._step_offsets.append(self._total_tokens)
            self._current_step_value = step

        self._current_row.setdefault(field_name, []).extend(data.tolist())
        self._total_tokens += len(data)

    def _flush(self) -> None:
        if not self._pending_rows:
            return
        columns: dict[str, list[Any]] = {}
        for field_name in self._field_to_type:
            is_seq = isinstance(self._field_features[field_name], HFSequence)
            col_data: list[Any] = []
            for row in self._pending_rows:
                if field_name in row:
                    col_data.append(row[field_name] if is_seq else row[field_name][0])
                else:
                    col_data.append(None)
            columns[field_name] = col_data

        table = pa.table(columns, schema=self._lance_schema)
        if self._ds is None:
            self._ds = lance.write_dataset(table, self._path)
        else:
            self._ds = lance.write_dataset(table, self._path, mode="append")
        self._flushed_step_count += len(self._pending_rows)
        self._pending_rows.clear()
        self._pending_token_count = 0

    def _flush_all(self) -> None:
        """Finalize the current step and flush everything to Lance."""
        if self._current_step_value is not None and self._current_row:
            self._pending_rows.append(dict(self._current_row))
            self._current_row = {}
            self._current_step_value = None
        self._flush()

    def _clear(self) -> None:
        self._pending_rows.clear()
        self._pending_token_count = 0
        self._current_row = {}
        self._current_step_value = None
        self._step_values.clear()
        self._step_offsets.clear()
        self._total_tokens = 0
        self._flushed_step_count = 0
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
            if arr.dtype == np.bool_:
                pass
            elif arr.itemsize != 8:
                arr = arr.astype(np.float64 if np.issubdtype(arr.dtype, np.floating) else np.int64)
            self._write(step, type_id, arr)

        self._prev_step, self._prev_type_id = prev_step, prev_type_id

    # ------------------------------------------------------------------
    # Dataset I/O  (from_dataset / to_dataset)
    # ------------------------------------------------------------------

    def _cast_arrow(
        self, table: pa.Table, target: pa.Schema,
    ) -> pa.Table:
        """Select and cast Arrow columns to match *target* schema."""
        cols = []
        for field in target:
            if field.name in table.column_names:
                col = table.column(field.name)
                if col.type != field.type:
                    col = col.cast(field.type)
            else:
                col = pa.nulls(table.num_rows, type=field.type)
            cols.append(col)
        return pa.table(cols, schema=target)

    def from_dataset(self, ds: Dataset) -> None:
        """Load a HuggingFace Dataset object into the store.

        The HF Dataset's Arrow table is cast to the Lance schema and written
        directly — no Python-list round-trip.
        """
        if not self._field_to_type:
            self.define_fields(list(ds.features.keys()), field_features=dict(ds.features))
        self._clear()

        n_rows = len(ds)
        if n_rows == 0:
            return

        lance_table = self._cast_arrow(ds.data.table, self._lance_schema)
        self._ds = lance.write_dataset(lance_table, self._path)
        self._flushed_step_count = n_rows

        for i in range(n_rows):
            self._step_values.append(i)
            self._step_offsets.append(self._total_tokens)
            tokens = 0
            for field in self._lance_schema:
                col = lance_table.column(field.name)
                val = col[i]
                if not val.is_valid:
                    continue
                if pa.types.is_list(field.type):
                    tokens += len(val.as_py())
                else:
                    tokens += 1
            self._total_tokens += tokens

        self._prev_step = n_rows - 1
        self._prev_type_id = max(
            self._field_to_type[f] for f in self._field_to_type
            if f in ds.column_names
        )

    def to_dataset(self) -> Dataset:
        """Export the store to a HuggingFace Dataset object.

        The Lance Arrow table is cast to the HF feature schema and wrapped
        directly — no Python-list round-trip.
        """
        self._require_fields()
        features = Features(self._field_features)
        if self._total_tokens == 0:
            return Dataset.from_dict({f: [] for f in self._field_features}, features=features)

        self._flush_all()
        lance_table = self._ds.to_table()  # type: ignore[union-attr]
        hf_table = self._cast_arrow(lance_table, features.arrow_schema)
        return Dataset(hf_table)

    # ------------------------------------------------------------------
    # Hub I/O  (load_dataset / save_dataset)
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_name: str,
        dataset_split: str,
        sort_by: list[str] | None = None,
    ) -> None:
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        ds = hf_load_dataset(dataset_name, split=dataset_split, download_mode="force_redownload")
        if sort_by:
            ds = ds.sort(sort_by, reverse=[False] * len(sort_by))
        self.from_dataset(ds)

    def save_dataset(self, dataset_name: str, dataset_split: str) -> None:
        if not dataset_name or not dataset_split:
            raise ValueError("dataset_name and dataset_split must be non-empty.")
        self.to_dataset().push_to_hub(dataset_name, split=dataset_split)
