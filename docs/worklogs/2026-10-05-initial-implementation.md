# 2026-10-05: initial implementation

Plan: tensor4all/tprims-rs#62. tenferro-rs is unchanged.

## What was built

- `EinsumSpec`: integer labels per input and output, contraction order in SSA
  form (inputs `0..n`, step `k` makes `n + k`), validated up front.
- `EinsumPlan<T>`: one `tprims_contract::Plan` per step. An intermediate keeps
  the labels of its two operands that are in the output or in a still-live
  operand, in first-appearance order, column-major. The last step writes the
  caller's output directly. Intermediates are placed first fit in one scratch
  buffer over their lifetimes (written at step k, read at step c, so a step's
  result never overlaps what it reads).
- A single input is a contraction with a rank-0 operand holding one, which
  tprims-contract accepts; this covers trace, sums and permutation.
- `contract_into` (one-shot binary) and `einsum_into` (one-shot N-ary).
- No `unsafe`: per step the scratch is split with `split_at_mut` around the
  destination, and each source lies wholly on one side.

## Tests

`crates/cpueinsum/tests/reference.rs` compares with a brute-force einsum over
every label assignment: binary and chained contractions in every order,
diagonals, in-operand reductions, outer and Hadamard products, rank-0
intermediates, single inputs, an MPS chain, strided and negative-stride
inputs with a row-major output, complex, plan reuse under serial and a
4-thread Rayon pool, zero extents, 300 random einsums with random orders, and
spec, shape and aliasing errors. All pass in dev and release.

## Not done yet

- Performance: no measurement yet. Each execution builds a `StridedView` per
  operand, which allocates its dims and strides; whether this matters against
  the per-step gate is decided by the final benchmark on the #61 corpus.
- tprims-rs#63 (packed selection for large complex problems) is in tprims.
- Conjugation flags and `alpha`/`beta` accumulation are left for later.
