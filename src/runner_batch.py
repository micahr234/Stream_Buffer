from __future__ import annotations

import torch

from stream_store import StreamStore
from tensordict import TensorDict


class RunnerBatch:
    """Base class for batch-sampling runners (train and eval).

    Manages window shuffling, batch index progression, and sampling from a
    StreamStore. Subclasses add the forward pass (train or eval).
    """

    def __init__(
        self,
        store: StreamStore,
        sequence_length: int,
        batch_size: int,
    ):
        self.store = store
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.next_batch_index = 0
        self.total_batches = self._compute_num_batches()
        self._window_order: torch.Tensor = self._new_epoch_order()

    def _compute_num_batches(self) -> int:
        total_tokens = len(self.store)
        num_windows = total_tokens // self.sequence_length
        if num_windows <= 0:
            raise ValueError(
                f"Store has {total_tokens} tokens but sequence_length is "
                f"{self.sequence_length}; need at least sequence_length tokens."
            )
        return (num_windows + self.batch_size - 1) // self.batch_size

    def _new_epoch_order(self) -> torch.Tensor:
        """Return a freshly shuffled tensor of window indices for one epoch."""
        num_windows = len(self.store) // self.sequence_length
        return torch.randperm(num_windows)

    def next_batch(self) -> TensorDict:
        start = self.next_batch_index * self.batch_size
        end = min(start + self.batch_size, len(self._window_order))
        window_indices = self._window_order[start:end]
        start_indices = window_indices * self.sequence_length
        seq_offsets = torch.arange(self.sequence_length)
        batch_indices = start_indices[:, None] + seq_offsets[None, :]
        raw = self.store[batch_indices.numpy()]  # store requires numpy indices
        stream = TensorDict(
            {k: torch.from_numpy(v) for k, v in raw.items()},
            batch_size=raw["types"].shape,
        )
        self.next_batch_index += 1
        if self.next_batch_index >= self.total_batches:
            self.next_batch_index = 0
            self._window_order = self._new_epoch_order()
        return stream
