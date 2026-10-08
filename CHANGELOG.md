# Changelog

## 0.0.1 — first public release

Reads and writes `.npy`, `.npz`, `.pkl`, `.pt`, `.joblib` and `.h5` from Rust
with no C, no C++, no FFI and no compiled artefacts from any other language.

### Formats

- **`.npy`** — every numpy descriptor with a `DType`, both byte orders, C and
  Fortran order, structured (record) dtypes in both directions, object arrays.
  Numeric arrays are read and written block by block, so the file image is
  never held in memory beside the array; the declared size is checked against
  the file length before anything is allocated.
- **`.npz`** — own ZIP implementation with ZIP64 and CRC-32; stored and
  deflated members; nested dicts flattened with `/` and rebuilt on read.
- **`.pkl`** — protocols 0–5 through a stack machine that never executes
  Python. numpy arrays recognised in all three forms numpy emits
  (`_reconstruct`, `scalar`, `_frombuffer`).
- **`.pt`** — the torch 1.6+ ZIP layout, with persistent-id storage resolution
  and strided-tensor gathering. Legacy checkpoints are refused explicitly.
  Written with pickle protocol 2, as `torch.save` does, so the default
  `torch.load(weights_only=True)` of PyTorch 2.6+ accepts the files.
- **`.joblib`** — inline out-of-band array blocks with alignment padding;
  uncompressed, zlib and gzip containers.
- **`.h5`** — superblocks 0–3, object headers v1 and v2 with continuations,
  symbol-table and link-message groups, contiguous, compact and chunked
  layouts, deflate, shuffle and fletcher32 filters. Writes a conformant
  version-0 subset that h5py and the reference C library read.

### Numerics

`f16`, `bf16`, `f8` (e4m3 and e5m2), `f128` with software arithmetic, x87
80-bit, and complex64/128/256 — all implemented from scratch. Casting between
any two element types under a `Strict`, `Saturate` or `Wrap` policy, with the
stored width configurable on both read and write.

### Everything else

- Chunked reading and writing for `.npy`, `.npz` and `.h5`.
- `Frame` as the neutral table type, with a pandas `BlockManager` reader and a
  feature-gated polars bridge.
- Write-time validation of container fit, element type, shape and
  representability, before any file is created (`.npz` archives are assembled
  in memory, so a rejected member leaves no partial file).
- 67 integration tests, 11 documentation tests (10 without `polars-interop`),
  and a Python cross-validation harness with 26 checks covering both
  directions, including the real `torch.load` when PyTorch is installed.
- Minimum supported Rust version 1.70 for the default features.
- Documentation in English and Russian.
