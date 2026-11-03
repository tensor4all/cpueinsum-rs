# Thin Complex64 products on `cblas_zgemv` (#67)

## Decisions

The reroute is decided in one place: `BlasScalar::gemm` for `Complex64`
(`crates/cpueinsum-blas/src/ffi.rs`). The gate is the *presented* GEMM's row
count, `Gemm.m <= 2`, because that is the geometry the vendor kernel sees. In
the ordinary case `Gemm.m` is the row role; when the planner reads the output
transposed (`layout.rs`, `swap`) it is the column role, and the same reroute is
the right one there.

One GEMM becomes `m` calls to `cblas_zgemm`'s replacement `cblas_zgemv`: row `i`
of `C` is row `i` of `op(A)` against `op(B)^T`. That reading needs no new
transposition logic: `op(B)` stored as itself is passed with `CblasTrans`, and a
transposed `op(B)` with `CblasNoTrans`, both with `ldb` and the dimensions the
planner already computed; row `i` of `op(A)` is contiguous when that operand is
read transposed and one element per `lda` otherwise. A conjugated reading
(`Trans::C`) has no GEMV form, so those keep the GEMM: f32/f64/Complex32 and
every conjugated Complex64 step are untouched.

`m` up to 2 wins by 5.4x to 15.5x; `m = 3` is already level with the GEMM, so
the threshold is 2 and no larger row count is rerouted. Batches are unaffected:
the reroute is inside the existing per-item call.

## Measurement

M5 Max (macOS), `VECLIB_MAXIMUM_THREADS=1`, `Exec::serial()`, so the measured
process ran one thread (`ps -M` samples during the run: 1). Release profile
(`opt-level=3`, `lto="thin"`, `codegen-units=1`). Workload: one
`cpueinsum_blas::GroupedPlan` job over contiguous column-major operands,
`D[i,l] = A[i,j] B[j,l]`, `alpha = 1`, `beta = 0`, deterministic
`ChaCha8Rng(67)` inputs.

Both arms are timed **in one process, interleaved** with the order flipped every
round, 300 rounds per cell, medians of per-round nanosecond timings; the
`before` binary is `origin/main` (`79ca304`), the `after` binary is this branch,
and each arm's raw `cblas_zgemm`/`cblas_zgemv` pair (the arm the crate called
before the change) is measured beside the crate's plan. Three runs of each
binary; the table reports the median run.

| dtype | m | k | n | before plan (ns) | after plan (ns) | faster by | relerr | output vs zgemm |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c64 | 1 | 141 | 141 | 36250 | 2417 | 15.00x | 2.3e-16 | in-run raw/plan 14.36x (was 1.00x) |
| c64 | 1 | 141 | 282 | 60416 | 4042 | 14.95x | 2.0e-16 | in-run raw/plan 15.42x (was 1.02x) |
| c64 | 1 | 282 | 141 | 58709 | 3833 | 15.32x | 2.5e-16 | in-run raw/plan 16.48x (was 1.00x) |
| c64 | 1 | 282 | 282 | 117625 | 7583 | 15.51x | 2.7e-16 | in-run raw/plan 15.17x (was 1.00x) |
| c64 | 2 | 141 | 141 | 26375 | 4875 | 5.41x | 9.6e-16 | in-run raw/plan 8.74x (was 1.01x) |
| c64 | 2 | 141 | 282 | 79542 | 7958 | 10.00x | 9.2e-16 | in-run raw/plan 10.09x (was 1.01x) |
| c64 | 2 | 282 | 141 | 77375 | 7292 | 10.61x | 1.4e-15 | in-run raw/plan 11.04x (was 1.04x) |
| c64 | 2 | 282 | 282 | 157125 | 13708 | 11.46x | 1.4e-15 | in-run raw/plan 11.51x (was 1.00x) |
| c64 | 3 | 141 | 282 | 79333 | 80542 | 0.98x | bit-identical | in-run raw/plan 1.00x (was 1.01x) |
| c64 | 4 | 141 | 282 | 79125 | 79458 | 1.00x | bit-identical | in-run raw/plan 1.00x (was 1.01x) |
| c64 | 8 | 141 | 282 | 78875 | 74959 | 1.05x | bit-identical | in-run raw/plan 1.06x (was 0.96x) |
| c64 | 8 | 282 | 282 | 157375 | 158917 | 0.99x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f32 | 1 | 141 | 141 | 292 | 458 | 0.64x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f64 | 1 | 141 | 141 | 375 | 458 | 0.82x | bit-identical | in-run raw/plan 1.00x (was 0.89x) |
| c32 | 1 | 141 | 141 | 11000 | 11042 | 1.00x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f32 | 1 | 141 | 282 | 542 | 708 | 0.77x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f64 | 1 | 141 | 282 | 666 | 667 | 1.00x | bit-identical | in-run raw/plan 1.00x (was 0.94x) |
| c32 | 1 | 141 | 282 | 21125 | 21083 | 1.00x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f32 | 2 | 141 | 282 | 3709 | 3750 | 0.99x | bit-identical | in-run raw/plan 0.99x (was 1.00x) |
| f64 | 2 | 141 | 282 | 5542 | 5542 | 1.00x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| c32 | 2 | 141 | 282 | 37167 | 36834 | 1.01x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f32 | 8 | 141 | 282 | 3459 | 3125 | 1.11x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| f64 | 8 | 141 | 282 | 4084 | 4042 | 1.01x | bit-identical | in-run raw/plan 1.00x (was 1.00x) |
| c32 | 8 | 141 | 282 | 32959 | 32959 | 1.00x | bit-identical | in-run raw/plan 1.04x (was 1.00x) |

f32/f64/Complex32 and Complex64 with `m >= 3` are bit-identical between the two
binaries (equal element hashes of every output), and their in-run raw/plan ratio
stays 1.00, which is what the unchanged GEMM route looks like. Their remaining
sub-microsecond differences in the table are this host's noise floor
(`f32 1x141x141` moves by 166 ns in both directions across runs).

Thin Complex64 is not bit-identical to the GEMM any more: Accelerate's `zgemv`
sums in a different order, and `tensor4all-rs#2416` measured 1.4e-14 for the same
pair. Here the largest elementwise relative difference over all thin cells and
runs is 1.4e-15.

Determinism: within every cell, two consecutive plan runs over a zeroed output
produced identical element hashes (asserted by the harness), so the same inputs
and thread count reproduce the same bits.

## Verification

- `thin_complex64_matches_zgemm` (`src/ffi/tests.rs`) reruns 64 `Complex64`
  GEMMs — both non-conjugated `ta`/`tb` readings, `m = 1` and `2`, four shapes
  with and without leading-dimension padding, `beta = 0` and nonzero — straight
  against `cblas_zgemm` and asserts agreement within `1e-13` relative plus one
  GEMV per row (so the reroute cannot pass vacuously).
- `thin_complex64_products` (`tests/reference.rs`) runs the #67 shapes
  (`m = 1,2` x `k = 141,282` x `n = 141,282`, stored and transposed output,
  random layouts) plus `m = 8` through `Blas` and compares with the naive sum at
  `1e-13`.
- `products_complex64` already covers the one-row vector cases against the naive
  sum on random layouts.

## Not done

- `m >= 3` (measured, no gain), batches (the reroute is per item anyway).
- The already-vector geometry `n == 1` with many rows is still a `zgemm`; it is
  a different reroute, not measured here.
- Conjugated `Complex64` operands keep the GEMM: no GEMV reading of the stored
  operand expresses `conj(B)`, and folding the conjugation into the vector would
  need a copy.
