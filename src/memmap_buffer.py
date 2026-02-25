"""Memory-mapped numpy buffer for large, incrementally appended data."""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np


class MemmapBuffer:
    """Memory-mapped numpy buffer for large, incrementally appended data.

    The underlying file grows in ``size_increment`` chunks whenever the
    current allocation is exhausted, so callers never need to know the
    final size in advance.
    """

    def __init__(self, dtype: Any, size_increment: int = 1_000_000):
        if size_increment <= 0:
            raise ValueError("size_increment must be > 0.")
        self.dtype = np.dtype(dtype)
        self.size_increment = int(size_increment)
        self.file = tempfile.NamedTemporaryFile(mode="w+b")
        self.storage = np.memmap(
            self.file.name,
            dtype=self.dtype,
            mode="w+",
            shape=(self.size_increment,),
        )
        self.index = 0

    def __len__(self) -> int:
        return self.index

    def close(self) -> None:
        self.storage.flush()
        del self.storage
        self.file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _grow(self) -> None:
        new_size = self.storage.shape[0] + self.size_increment
        self.storage.flush()
        del self.storage
        self.file.truncate(new_size * self.dtype.itemsize)
        self.file.flush()
        self.storage = np.memmap(
            self.file.name,
            dtype=self.dtype,
            mode="r+",
            shape=(new_size,),
        )

    def append(self, data: Any) -> None:
        values = np.atleast_1d(np.asarray(data, dtype=self.dtype))
        if values.ndim != 1:
            raise ValueError("append expects scalar or 1D array data.")
        new_index = self.index + values.shape[0]
        while self.storage.shape[0] < new_index:
            self._grow()
        self.storage[self.index:new_index] = values
        self.index = new_index

    def reset(self) -> None:
        """Reset the buffer to empty without freeing the underlying memory."""
        self.index = 0

    def __iter__(self):
        return iter(self.storage[: self.index])

    def __getitem__(self, idx: Any) -> np.ndarray:
        return self.storage[: self.index][idx]
