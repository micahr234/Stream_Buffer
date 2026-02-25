"""Performance comparison across all four StreamStore backends.

Benchmarks:
  1. append     – incremental online insertion
  2. __getitem__ – random-access window sampling (the training hot path)
  3. from_dataset / to_dataset – bulk HuggingFace round-trip
  4. storage footprint

Each benchmark prints a table so results are easy to compare.
Run with:  pytest tests/test_compare_performance.py -s
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import Dataset, Features, Sequence as HFSequence, Value
from original_stream_store import OriginalStreamStore
from stream_store import StreamStore
from lance_stream_store import LanceStreamStore
from lance_table_store import LanceTableStore

FIELDS = ["action", "observation", "reward", "done"]
FEATURES = {
    "action":      Value("int64"),
    "observation": HFSequence(Value("float32")),
    "reward":      Value("float32"),
    "done":        Value("bool"),
}


def _make_store(cls, capacity):
    store = cls(capacity=capacity)
    store.define_fields(FIELDS, field_features=FEATURES)
    return store


def _make_dataset(n_rows: int, obs_dim: int) -> Dataset:
    return Dataset.from_dict(
        {
            "action":      np.random.randint(0, 3, n_rows).tolist(),
            "observation": np.random.randn(n_rows, obs_dim).astype(np.float32).tolist(),
            "reward":      np.random.randn(n_rows).astype(np.float32).tolist(),
            "done":        [bool(i % 100 == 99) for i in range(n_rows)],
        },
        features=Features(
            {
                "action":      Value("int64"),
                "observation": HFSequence(Value("float32")),
                "reward":      Value("float32"),
                "done":        Value("bool"),
            }
        ),
    )


BACKENDS = [
    ("Original (per-token meta)", OriginalStreamStore),
    ("IndexTable (split bufs)",   StreamStore),
    ("Lance stream (flat)",       LanceStreamStore),
    ("Lance table (tabular)",     LanceTableStore),
]


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _row(label: str, elapsed: float, extra: str = "") -> None:
    print(f"  {label:<30s}  {elapsed:8.4f}s  {extra}")


# ------------------------------------------------------------------
# 1. append benchmark
# ------------------------------------------------------------------

class TestAppendPerformance:
    """Incremental append: N steps, one append call per step."""

    N = 10_000
    OBS_DIM = 8

    def _run(self, cls):
        store = _make_store(cls, capacity=self.N * (self.OBS_DIM + 3) + 1000)
        fields = FIELDS
        t0 = time.perf_counter()
        for i in range(self.N):
            store.append(
                steps=[i] * 4,
                names=fields,
                values=[
                    i % 3,
                    np.random.randn(self.OBS_DIM).astype(np.float32),
                    float(i) * 0.1,
                    i % 100 == 99,
                ],
            )
        return time.perf_counter() - t0

    def test_append_comparison(self):
        _header(f"append – {self.N} steps, obs_dim={self.OBS_DIM}")
        for label, cls in BACKENDS:
            elapsed = self._run(cls)
            tps = self.N / elapsed
            _row(label, elapsed, f"({tps:,.0f} steps/s)")


# ------------------------------------------------------------------
# 2. __getitem__ benchmark
# ------------------------------------------------------------------

class TestGetitemPerformance:
    """Random-access window sampling – the training hot path."""

    N = 20_000
    OBS_DIM = 8
    WINDOW = 128
    N_SAMPLES = 5_000

    def _prepare(self, cls):
        store = _make_store(cls, capacity=self.N * (self.OBS_DIM + 3) + 1000)
        fields = FIELDS
        for i in range(self.N):
            store.append(
                steps=[i] * 4,
                names=fields,
                values=[
                    i % 3,
                    np.random.randn(self.OBS_DIM).astype(np.float32),
                    float(i) * 0.1,
                    i % 100 == 99,
                ],
            )
        return store

    def _run(self, store):
        total = len(store)
        max_start = total - self.WINDOW
        starts = np.random.randint(0, max_start, size=self.N_SAMPLES)
        indices = np.array([np.arange(s, s + self.WINDOW) for s in starts])

        t0 = time.perf_counter()
        for idx_row in indices:
            _ = store[idx_row]
        elapsed = time.perf_counter() - t0
        return elapsed

    def test_getitem_comparison(self):
        _header(
            f"__getitem__ – {self.N_SAMPLES} random windows of {self.WINDOW} "
            f"from {self.N} steps (obs_dim={self.OBS_DIM})"
        )
        for label, cls in BACKENDS:
            store = self._prepare(cls)
            elapsed = self._run(store)
            qps = self.N_SAMPLES / elapsed
            _row(label, elapsed, f"({qps:,.0f} samples/s)")

    def test_getitem_batched_comparison(self):
        """Single batched __getitem__ call with a 2-D index array."""
        _header(
            f"__getitem__ (batched) – ({self.N_SAMPLES}, {self.WINDOW}) "
            f"index from {self.N} steps"
        )
        for label, cls in BACKENDS:
            store = self._prepare(cls)
            total = len(store)
            max_start = total - self.WINDOW
            starts = np.random.randint(0, max_start, size=self.N_SAMPLES)
            indices = np.array([np.arange(s, s + self.WINDOW) for s in starts])

            t0 = time.perf_counter()
            _ = store[indices]
            elapsed = time.perf_counter() - t0
            _row(label, elapsed)


# ------------------------------------------------------------------
# 3. from_dataset / to_dataset round-trip benchmark
# ------------------------------------------------------------------

class TestRoundtripPerformance:
    """Bulk HuggingFace Dataset round-trip."""

    N = 20_000
    OBS_DIM = 8

    def test_from_dataset_comparison(self):
        ds = _make_dataset(self.N, self.OBS_DIM)
        _header(f"from_dataset – {self.N} rows, obs_dim={self.OBS_DIM}")
        for label, cls in BACKENDS:
            store = _make_store(cls, capacity=self.N * (self.OBS_DIM + 3) + 1000)
            t0 = time.perf_counter()
            store.from_dataset(ds)
            elapsed = time.perf_counter() - t0
            rps = self.N / elapsed
            _row(label, elapsed, f"({rps:,.0f} rows/s)")

    def test_to_dataset_comparison(self):
        ds = _make_dataset(self.N, self.OBS_DIM)
        _header(f"to_dataset – {self.N} rows, obs_dim={self.OBS_DIM}")
        for label, cls in BACKENDS:
            store = _make_store(cls, capacity=self.N * (self.OBS_DIM + 3) + 1000)
            store.from_dataset(ds)
            t0 = time.perf_counter()
            _ = store.to_dataset()
            elapsed = time.perf_counter() - t0
            rps = self.N / elapsed
            _row(label, elapsed, f"({rps:,.0f} rows/s)")


# ------------------------------------------------------------------
# 4. Memory footprint comparison
# ------------------------------------------------------------------

class TestMemoryFootprint:
    """Compare raw storage size across backends."""

    N = 10_000
    OBS_DIM = 16

    def test_storage_size(self):
        _header(f"Storage footprint – {self.N} steps, obs_dim={self.OBS_DIM}")
        tokens_per_step = self.OBS_DIM + 3
        total_tokens = self.N * tokens_per_step
        print(f"  Total tokens: {total_tokens:,}")
        print()

        for label, cls in BACKENDS:
            store = _make_store(cls, capacity=total_tokens + 1000)
            fields = FIELDS
            for i in range(self.N):
                store.append(
                    steps=[i] * 4,
                    names=fields,
                    values=[
                        i % 3,
                        np.random.randn(self.OBS_DIM).astype(np.float32),
                        float(i) * 0.1,
                        i % 100 == 99,
                    ],
                )

            if cls is OriginalStreamStore:
                bytes_used = store._buf.index * store._buf.dtype.itemsize
                _row(label, 0, f"{bytes_used:,} bytes  ({bytes_used / total_tokens:.1f} B/token)")
            elif cls is StreamStore:
                data_bytes = store._data_buf.index * store._data_buf.dtype.itemsize
                idx_bytes = store._idx_buf.index * store._idx_buf.dtype.itemsize
                total_bytes = data_bytes + idx_bytes
                _row(label, 0, f"{total_bytes:,} bytes  ({total_bytes / total_tokens:.1f} B/token)  [data={data_bytes:,} idx={idx_bytes:,}]")
            elif cls in (LanceStreamStore, LanceTableStore):
                if hasattr(store, '_flush_all'):
                    store._flush_all()
                else:
                    store._flush()
                if store._ds is not None:
                    disk_bytes = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk(store._path)
                        for f in fns
                    )
                    _row(label, 0, f"{disk_bytes:,} bytes on disk  ({disk_bytes / total_tokens:.1f} B/token)")
                else:
                    _row(label, 0, "no data flushed")
