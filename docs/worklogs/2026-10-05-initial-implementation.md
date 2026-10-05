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

- Performance: the final benchmark on the #61 corpus is still to run.
  Building a `StridedView` per operand cost about 70 ns per step (two `Arc`
  allocations per view; chi=2 MPS overlap: 126 ns/step with views, 55 ns/step
  with slices, M5 Max 1T). Execution therefore goes through
  `Plan::execute_slices` (tprims-rs#66): inputs and the output are passed as
  their data slice and origin, intermediates as scratch subslices, and no view
  is built per step.
- tprims-rs#63 (packed selection for large complex problems) landed as
  `FaerLimit` in tprims-rs#65; the pin includes it.
- Conjugation flags and `alpha`/`beta` accumulation are left for later.
