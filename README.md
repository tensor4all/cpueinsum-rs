# cpueinsum-rs

CPU einsum over integer labels, with the contraction order supplied by the
caller.

Each binary step of an einsum is planned once by
[tprims-contract](https://github.com/tensor4all/tprims-rs) (a packed
TBLIS-style driver for general strides, faer for copy-free GEMM fusions, an
elementwise pass for all-batch steps). cpueinsum adds what an N-ary einsum
needs on top:

- an explicit contraction order in SSA form,
- the labels each intermediate keeps (the output's and those a later operand
  still needs; everything else is summed as early as possible),
- one reusable scratch buffer for all intermediates, with dead storage reused
  first fit,
- a small, stable integer-label API.

Deliberately not included: contraction-order search, string notation
(`"ij,jk->ik"`), and GPUs. Those belong to the caller.

```rust
use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Layout, Scratch};

// d[i, l] = sum_{j,k} a[i, j] b[j, k] c[k, l], contracting (a b) first.
// Inputs are operands 0, 1, 2; step 0 makes operand 3.
let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [3, 2]])?;
let cm = Layout::new(&[2, 2], &[1, 2])?;
let plan = EinsumPlan::<f64>::new(&spec, &[cm, cm, cm], cm)?;
let mut scratch = Scratch::new();

let a = [1.0, 2.0, 3.0, 4.0];
let id = [1.0, 0.0, 0.0, 1.0];
let mut d = [0.0; 4];
let views = [
    StridedView::new(&a, &[2, 2], &[1, 2], 0)?,
    StridedView::new(&id, &[2, 2], &[1, 2], 0)?,
    StridedView::new(&id, &[2, 2], &[1, 2], 0)?,
];
let mut out = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0)?;
plan.execute_into(&Exec::serial(), &views, &mut out, &mut scratch)?;
assert_eq!(d, a);
```

`einsum_into` plans and runs in one call; `contract_into` is a one-shot binary
contraction. Threading is explicit: pass `Exec::rayon(&Pool::borrow(&pool))`
to use a caller-owned Rayon pool.

## Prepared binary and grouped execution

[`BinaryPlan`](crates/cpueinsum/src/prepared.rs) prepares a tprims `Problem`
once and accepts arbitrary alpha/beta, Absent/Output/Separate C, and initialized
views or slices. Its fresh-output entry accepts `&mut [MaybeUninit<T>]` and
returns initialized storage only after successful full physical coverage;
padding/diagonal holes and nonzero-beta Output accumulation are rejected.
Execution returns the actual owning-layer route, not just the plan's default
algorithm. Neither binary nor grouped plans retain executors or data buffers.

[`GroupedPlan`](crates/cpueinsum/src/grouped.rs) prepares heterogeneous compact
column-major jobs over shared flat buffers. It checks every job before writing,
rejects overlapping output ranges, and uses one bounded outer scheduler with
serial children. Grouped output is initialized; output holes stay unchanged.
Serial callers can retain `ArenaProvider`, and parallel callers retain `Pool`.

The shared-input overwrite/accumulation example lives in
[`GroupedPlan`'s executable rustdoc](crates/cpueinsum/src/grouped.rs), checked
with `cargo test -p cpueinsum --doc`; there is no duplicate snippet to drift.

See the [design and boundary constraints](docs/design/2026-10-06-prepared-binary-and-grouped.md).
Grouped dispatch currently allocates borrow-only job metadata (one item vector,
and one lane vector for parallel execution); it creates no per-job arena, view
metadata, or product buffer. This is not an allocation-free boundary or a
measured speedup claim.

## BLAS

`cpueinsum` links no BLAS. A `StepBackend` can take binary steps over from
tprims-contract at planning time, and the separate `cpueinsum-blas` crate is
one that runs them on CBLAS. It is for large matrices: steps whose GEMMs have
at least 2^15 multiply-accumulates each become one GEMM per batch item, in place when the operands fuse to matrices BLAS can
read and after packing the others; smaller steps stay on tprims-contract.
`cpueinsum-blas` depends on `cblas-sys` only, so the final binary chooses the
provider (for example through `blas-src`).

```rust
use cpueinsum::{EinsumPlan, EinsumSpec, Layout};
use cpueinsum_blas::Blas;

let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]])?;
let cm = Layout::new(&[256, 256], &[1, 256])?;
let plan = EinsumPlan::<f64, Blas>::with_backend(Blas::default(), &spec, &[cm, cm], cm)?;
assert_eq!(plan.backend_steps(), 1);
```

For direct prepared output contracts, `cpueinsum_blas::BinaryPlan` and
`cpueinsum_blas::GroupedPlan` are concrete adapters, separate from the
unchanged overwrite-only `StepBackend`. They borrow explicit packing storage
(`work_len()` elements) and an `Exec` for native alternatives. Compatible
layouts write output directly through coordinator-inline CBLAS calls with
alpha/beta. Unsupported vendor geometry/C modes select native plans before
execution; scalar-degenerate calls use a prepared native branch, never retry.
Fresh binary output requires exact physical coverage and no old-output reads;
grouped output remains initialized and every job is preflighted before writes.
A group is entirely vendor or entirely native, not mixed after mutation.
Executable examples live in the adapters' rustdoc; see the
[prepared BLAS design](docs/design/prepared-blas-integration.md).

The default backend, `Tprims`, takes no step, so `EinsumPlan<T>` runs exactly
as without backends.

Element types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Strides may be
arbitrary, including negative; a label repeated within an input selects a
diagonal.

## Status

Early. The plan, layering and gates are in
[tensor4all/tprims-rs#62](https://github.com/tensor4all/tprims-rs/issues/62);
the BLAS backend in
[tensor4all/cpueinsum-rs#1](https://github.com/tensor4all/cpueinsum-rs/issues/1).
Not published to crates.io; tprims is a git dependency at a pinned rev.
MSRV 1.89.

## License

MIT OR Apache-2.0.
