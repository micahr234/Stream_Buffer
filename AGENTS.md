# Agent guidance – Stream Buffer

## Purpose

StreamStore is a fast, memory-mapped flat buffer for sequential, multi-field data. It supports quick incremental insertions, efficient fixed-width window sampling, and lossless round-trips to and from HuggingFace Datasets. The design is general-purpose — any application that collects data one step at a time and samples context windows for learning can use it.

## Design

1. **Field names map to token type IDs** — the caller supplies a `field_to_type` dict at construction, mapping each field name to an integer type ID. Every token carries that ID, so the store can identify field types across the full buffer without storing names repeatedly. Field names are never hardcoded inside the store — all field lookups go through the reverse mapping derived from `field_to_type`.

2. **Fields are stored as a sequentially concatenated flat stream** — all tokens from all steps are appended one after another in a single 1-D buffer. Any slice of token indices maps directly to a contiguous buffer region, enabling fast, zero-copy sampling of arbitrary fixed-width windows.

3. **Memmap-backed storage** — the buffer lives in a temporary memory-mapped file rather than RAM. It grows dynamically in fixed-size chunks, so arbitrarily large streams can be held without memory pressure and callers never need to know the final size in advance.

4. **Lossless encoding** — all values (floats, ints, bools) are cast to `float64` and stored as their raw `int64` bit pattern. On read the bits are reinterpreted back to `float64` via `.view(np.float64)`. `float64` is exact for all integers up to 2^53.

5. **Memory-efficient HuggingFace round-trip** — `to_dataset` / `from_dataset` convert the buffer in bulk. Scalar fields are extracted with a single type-mask over the entire buffer; only variable-length fields require a per-step grouping pass.

## Token schema

Every token in the buffer is a structured record with three fields:

| Field  | Role |
|--------|------|
| `step` | Corresponds to a **row** in a HuggingFace dataset — groups all tokens belonging to one time step. |
| `type` | Corresponds to a **column** in a HuggingFace dataset — identifies which field this token carries, via `field_to_type`. |
| `data` | The value itself, stored as raw `int64` bits of a `float64`. |

- **Insertion order**: within each `append` call, fields must appear in `field_to_type` key order — fields may be omitted, but those present must be strictly ordered. Steps must arrive in non-decreasing step order across calls.
- **Loading from HuggingFace**: `from_dataset` ignores any column not in `field_to_type` and appends the remaining columns in `field_to_type` key order. The dataset rows must be in step order before calling `from_dataset`; use `load_dataset`'s `sort_by` parameter or pre-sort manually.

## Codebase

- **`src/stream_store.py`** – `StreamStore`: the main class. Constructor takes `field_to_type` (field name → integer type ID) and `capacity`. Core API: `append` (insert), `__getitem__` (sample). Dataset I/O comes in two symmetric pairs:
  - `from_dataset(ds)` / `to_dataset(field_features, extra)` — in-memory Dataset object ↔ store.
  - `load_dataset(name, split, sort_by)` / `save_dataset(name, split, field_features, extra)` — HuggingFace Hub ↔ store.
- **`src/memmap_buffer.py`** – `MemmapBuffer`: the dynamically growing memory-mapped numpy buffer used internally by `StreamStore`.
- **`tests/test_stream_store.py`** – Comprehensive test suite: append semantics, `__getitem__` shapes, dataset round-trips, variable field lengths, and performance benchmarks.

## Setup and running

- **`install.sh`** – Run once (`source install.sh`) to set up Python via uv, git, and VSCode config.
- **Tests and scripts** — Activate the venv first (`source .venv/bin/activate`), then run directly — e.g. `pytest tests/`.

**Adding new packages**: never use `pip install` directly.
1. Add the package to `pyproject.toml` under `[project].dependencies` (or `.dev` for dev-only tools).
2. Re-run `bash install.sh` to rebuild the venv from scratch.
3. Use the venv at `./.venv` as normal.

## Specific Considerations

- **Critical paths are `append` and `__getitem__`** — keep these fast and simple. `from_dataset` / `to_dataset` are one-shot bulk operations; do not optimise them at the cost of complexity.
- **Never hardcode field name strings** — derive field names from `self._type_to_field` (the reverse of the caller-supplied `field_to_type`, built in `__init__`).

## General AI Expectations

- Always verify your work before considering it done.
- After any change, ask: do variable names still make sense?
- After any change, ask: could a refactor or restructure clarify the code?
- Do not automatically revert removed or changed code — assume the change was intentional.
- As you learn other ways this repo works or discover user preferences, suggest adding them to this document.
