# 2026-10-05: cpueinsum-blas (Phase 1 of cpueinsum-rs#1)

Plan: tensor4all/cpueinsum-rs#1. tenferro-rs is unchanged (Phases 2 and 3
come later).

## What was built

- `StepBackend<T>` in `cpueinsum`: offered every binary step at planning,
  takes it (returning its state and a work length) or declines it. The plan
  is generic over the backend (`EinsumPlan<T, B = Tprims>`), so routing is
  one branch per step and no dynamic call. A taken step's work space is a
  temporary in the scratch buffer, placed first fit like intermediates and
  split off with `split_at_mut`; `cpueinsum` stays free of `unsafe` and of
  any BLAS dependency (`cargo tree -p cpueinsum` has no cblas-sys).
- `crates/cpueinsum-blas`: the `Blas` backend on `cblas-sys` only, ported
  from tenferro `tenferro-cpu/src/gemm/blas_gemm.rs`. At planning it fuses
  the m, n and k groups of each operand to one matrix axis when the strides
  chain (the group order follows each carrier's stride order), uses the
  output in place (as `D` or as `D^T = B^T A^T`) and the inputs in place
  (`NoTrans`, `Trans`, or `ConjTrans` for conjugated inputs), and otherwise
  packs the operand column-major into the work space with strided-perm.
  Batch axes are looped. Execution checks
  every operand's addressed range against its slice before the FFI calls.
- Declined: steps whose GEMMs have fewer than `DEFAULT_MIN_MACS` (2^15)
  multiply-accumulates each (the BLAS path is for large matrices; see
  "Threads" below), and everything that is not a
  plain product (empty k or output, all-batch, reductions over an axis one
  input lacks, a non-injective output, an accumulated `C`).

## Tests

`crates/cpueinsum-blas/tests/reference.rs` reuses the naive reference of
`crates/cpueinsum/tests`: 11 binary shapes in 12 random layouts (permuted,
gapped, reversed, offset) for f32, f64, Complex32 and Complex64, chains that
mix the two backends, the thresholds, 300 random einsums, conjugated
operands driven through the backend directly (checking which operands were
packed), and bounds errors. Each case runs twice on dirty scratch and checks
untouched output markers. All pass in dev and release on macOS (Accelerate).
Not run here: Linux/OpenBLAS.

## Benchmark

tprims-rs#61 corpus, one thread, M5 Max, Accelerate, median of 11 samples of
20 ms; two runs each; ns per step. `call` plans and allocates per call,
`exec` reuses a plan and scratch. `base` is cpueinsum at 9773ead.

| case | call base | call new | blas call | exec base | blas exec | BLAS steps |
| --- | --- | --- | --- | --- | --- | --- |
| mps chi4 (c64) | 908 | 878 to 900 | 892 to 906 | 112 to 118 | 110 to 114 | 0 |
| mps chi8 | 1040 to 1064 | 1012 to 1020 | 1036 | 263 to 272 | 249 to 253 | 0 |
| mps chi32 | 12000 to 12300 | 11900 to 12000 | 10460 to 10520 | 11190 to 11460 | 9420 to 9580 | 64 |
| mps chi64 | 79700 to 80300 | 78100 to 81100 | 38100 to 38500 | 76400 | 37700 to 37800 | 64 |
| ikb,knb n16 b16 | 6146 to 6277 | 6081 to 6110 | 6000 to 6360 | 4650 to 4670 | 4140 to 4590 | 1 |
| ikb,knb n16 b256 | 75900 to 76700 | 75600 to 76100 | 65000 to 66900 | 72100 to 73600 | 63200 to 63600 | 1 |
| ij,jk c64 n32 | 6490 to 6570 | 6430 to 6520 | 6410 to 6450 | 5370 to 5490 | 5020 | 1 |
| ij,jk f64 n64 | 11300 to 11600 | 11160 to 11210 | 3570 to 3630 | 10230 to 10260 | 2120 | 1 |
| ij,jk,kl n64 | 11380 to 11550 | 11080 to 11140 | 3260 to 3290 | 10290 to 10910 | 1970 to 1990 | 2 |

Gates of cpueinsum-rs#1: the default backend keeps its per-call cost
(chi 4: 878 to 932 ns/step against 908 before); with `Blas::default()` the
small steps are declined and stay at 892 to 939 ns/step (gate 1.3 µs).
Every other case of the corpus with zero BLAS steps matches the default
backend within noise.

The per-GEMM threshold came from the first run, which had only the total
threshold: `ikb,knb->inb` with 8^3 items took BLAS and ran 1.6 to 2 times
slower (n8 b64: 4300 to 5730 ns against 2720; n8 b256: 16900 to 17600
against 10500), since a batch is a loop of small BLAS calls.

## Not done yet

- Linux with OpenBLAS (tests and the thresholds there).
- Phases 2 and 3 (tenferro bridge and delegation) in tenferro-rs.

## Threads (example `bench`)

`cargo run --release -p cpueinsum-blas --example bench` times tprims-contract
and `Blas::default()` at 1 and 4 threads (tprims on a Rayon pool, BLAS
threads through `VECLIB_MAXIMUM_THREADS` / `OPENBLAS_NUM_THREADS`, one child
process per thread count). M5 Max, Accelerate, `exec` medians in µs:

| case | tprims 1T | blas 1T | tprims 4T | blas 4T | BLAS steps |
| --- | --- | --- | --- | --- | --- |
| mps L32 chi4 (c64) | 6.92 | 6.96 | 7.03 | 7.10 | 0/64 |
| mps L32 chi32 | 705 | 613 | 711 | 619 | 64/64 |
| mps L32 chi64 | 4744 | 2648 | 2655 | 2525 | 64/64 |
| mps L32 chi128 | 34972 | 11776 | 11108 | 11799 | 64/64 |
| ikb,knb n16 b256 | 71.5 | 61.6 | 27.7 | 61.3 | 1/1 |
| ikb,knb n32 b256 | 338 | 99.4 | 107 | 99.7 | 1/1 |
| ij,jk f64 n256 | 573 | 82.7 | 159 | 84.0 | 1/1 |
| ij,jk c64 n256 | 2106 | 508 | 601 | 525 | 1/1 |
| ij,jk f64 n1024 | 36157 | 4472 | 9513 | 4476 | 1/1 |
| ij,jk,kl n512 | 9067 | 1156 | 2971 | 1267 | 2/2 |

Accelerate does not scale with threads on this machine (1024^3 f64: 4.5 ms
with `VECLIB_MAXIMUM_THREADS` 1, 4, 18 or unset), presumably because its
GEMM runs on the matrix unit. tprims scales about 3 to 4 times at 4 threads,
so at 4 threads it beats the BLAS backend on batches of small GEMMs
(n16 b256: 27.7 against 61.3 µs, the batch loop being serial) and ties it on
MPS chi 128. The thresholds do not depend on the thread count yet.

Decision: the BLAS path is for large matrices only. The two thresholds of
the first version (2^15 per step, 2^12 per GEMM) became one, 2^15 per GEMM:
from 32^3 BLAS wins or ties at both thread counts, and the batches of 16^3
that it took before go back to tprims-contract (27.7 µs at 4 threads).

Removed afterwards: the `vendor-batch` feature (`cblas_?gemm_batch` for
batches of items up to 16 per extent). Under the 2^15 per-GEMM threshold it
never ran by default, Accelerate lacks the symbol, and tenferro itself keeps
strided batches (the shape of every cpueinsum step) off the vendor batch
under its default strategy: on OpenBLAS 0.3.32 at one thread it measured
1.8x slower at 8^3 and 3.8x at 16^3 than per-item GEMMs
(`tenferro-cpu/src/dot_runtime.rs`, `strided_batch_route` bench).
