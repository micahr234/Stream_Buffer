# Stream Buffer

Token stream storage backed by memory-mapped numpy buffers, with HuggingFace Dataset I/O.

## Overview

`StreamStore` holds a flat token stream in a single memory-mapped numpy buffer. Each token is a structured record of `(step, type, data)` — where `data` stores any float or int as raw float64 bits. This lets the store handle variable-length observations, discrete actions, scalar rewards, and boolean done flags in a uniform, schema-free layout.

**Key properties:**

- **Variable-length fields** — no fixed length required; each step can have a different number of tokens per field.
- **Dynamic capacity** — the underlying `MemmapBuffer` grows in chunks as tokens are appended; no pre-allocation needed.
- **HuggingFace round-trip** — `to_dataset()` / `from_dataset()` convert between the token stream and a columnar HuggingFace `Dataset`.
- **Arbitrary index shapes** — `__getitem__` accepts an index array of any shape and returns `{"types": int64[...], "values": int64[...]}` with matching batch dimensions. Reinterpret `values` as float64 via `.view(np.float64)` when needed.

## Core API

```python
from stream_store import StreamStore
import numpy as np

store = StreamStore(fields=["action", "observation", "reward", "done"])

# Append one step (flat parallel sequences: one entry per field)
store.append(
    steps=[0, 0, 0, 0],
    names=["action", "observation", "reward", "done"],
    values=[1, np.array([0.1, -0.2, 0.05, 0.3]), 1.0, False],
)

# Sample by flat token indices (any shape)
batch = store[np.array([0, 1, 2, 3, 4])]
# batch["types"]  → int64 array of type IDs
# batch["values"] → int64 raw bits; use .view(np.float64) to reinterpret

# Convert to HuggingFace Dataset
ds = store.to_dataset()

# Load back (dataset columns must match store fields and be in construction order)
store2 = StreamStore(fields=["action", "observation", "reward", "done"])
store2.from_dataset(ds)
```

## Token layout

Each step appended produces tokens in field order. For `["action", "observation", "reward", "done"]` with a 4-D observation:

```
action (1 token) | obs_0 … obs_3 (4 tokens) | reward (1 token) | done (1 token)
```

All values are cast to `float64` and stored as raw `int64` bit patterns — lossless for ints up to 2^53 and exact for float64 payloads.

## Field definitions

Pass a list of field names at construction. Type IDs are auto-assigned as consecutive integers (0, 1, 2, …) in that order:

```python
store = StreamStore(fields=["action", "observation", "reward", "done"])
# action=0, observation=1, reward=2, done=3
```

Fields must appear in construction order when calling `append`. The dataset columns must match this order for `from_dataset` to succeed.

## Dataset schema

`to_dataset()` produces field columns in construction order:

```
action, observation, reward, done
```

Use `extra` for per-step columns that are not part of the token stream (e.g. provenance, episode IDs).

## Setup

```bash
source install.sh
```

## Tests

```bash
source .venv/bin/activate
pytest tests/
```
