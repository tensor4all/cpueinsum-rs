# cpueinsum-rs

CPU einsum over integer labels, with the contraction order supplied by the
caller.

Each binary step of an einsum is planned once by
[tcontract](https://github.com/tensor4all/tcontract-rs) (a packed
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
plan.execute_into(&mut Exec::serial(), &views, &mut out, &mut scratch)?;
assert_eq!(d, a);
```

`einsum_into` plans and runs in one call; `contract_into` is a one-shot binary
contraction. Threading is explicit: build a `threadteam::Pool` once
(`threadteam::Pool::builder().threads(n).build()?`), wrap it with
`Pool::borrow(&pool)`, and mint `Exec::new(&pool, budget)?` per call. Every
operation takes the `Exec` by `&mut`, because the token that places its work is
not copyable.

## Prepared binary and grouped execution

[`BinaryPlan`](crates/cpueinsum/src/prepared.rs) prepares a tcontract `Problem`
once and accepts arbitrary alpha/beta, Absent/Output/Separate C, and initialized
views or slices. Its fresh-output entry accepts `&mut [MaybeUninit<T>]` and
returns initialized storage only after successful full physical coverage;
padding/diagonal holes and nonzero-beta Output accumulation are rejected.
Execution returns the actual owning-layer route, not just the plan's default
algorithm. Neither binary nor grouped plans retain executors or data buffers.

[`GroupedPlan`](crates/cpueinsum/src/grouped.rs) prepares heterogeneous compact
column-major jobs over shared flat buffers. It checks every job before writing,
rejects overlapping output ranges, and resolves its scheduler from the caller's
`Exec` batch policy: `Auto` keeps the historical lane decision, a forced
`Sequential`/`Outer`/`Inner`/`Split` is honored (clamped to the items and the
budget, never silently downgraded).
Children run with the decided item width. Grouped output is initialized; output
holes stay unchanged. Serial callers can retain `ArenaProvider`, and parallel
callers retain `Pool`.

The shared-input overwrite/accumulation example lives in
[`GroupedPlan`'s executable rustdoc](crates/cpueinsum/src/grouped.rs), checked
with `cargo test -p cpueinsum --doc`; there is no duplicate snippet to drift.

See the [design and boundary constraints](docs/design/2026-10-06-prepared-binary-and-grouped.md).
Grouped dispatch allocates no per-job metadata: the serial route drives the jobs
by index and the parallel route gives each lane a contiguous job range with its
own output chunk. A `tests/alloc_free.rs` gate holds zero Rust-owned allocations
on the warmed serial steady state, and the parallel bound is parameterized by
the decided lane count, never by the batch size.

## BLAS

`cpueinsum` links no BLAS. A `StepBackend` can take binary steps over from
tcontract at planning time, and the separate `cpueinsum-blas` crate is
one that runs them on CBLAS. It is for large matrices: steps whose GEMMs have
at least 2^15 multiply-accumulates each become one GEMM per batch item, in place when the operands fuse to matrices BLAS can
read and after packing the others; smaller steps stay on tcontract.
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

Every BLAS route here runs a batch as a serial loop of one call per item, and
all of them resolve the caller's `Exec` batch policy with the same rule
(`cpueinsum::VendorLoopRoute`): `Auto` and `Sequential` keep the loop, a forced
`Outer`/`Inner`/`Split` runs the route's tcontract alternative. A `StepBackend`
declares such a loop with
`StepBackend::vendor_loop`, so the same rule reaches binary steps of an N-ary
`EinsumPlan`, whose plan-wide admission resolves every step — backend or native,
including the native alternative a forced lane strategy selects — after view
validation and before the arena grows or the first step runs.

The default backend, `Tcontract`, takes no step, so `EinsumPlan<T>` runs exactly
as without backends.

Element types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Strides may be
arbitrary, including negative; a label repeated within an input selects a
diagonal.

## 0.5 changes

The 0.5.0 release tracks tcontract-exec 0.5.0 and strided 0.7.0. Breaking changes:

- Every operation takes `&mut Exec` (`execute_into`, `execute_slices`,
  `execute_into_accum`, `execute_uninit_slices`, `einsum_into`, `contract_into`,
  and `StepBackend::execute`). `native_workspace_req` and `execution_route` still
  take `&Exec`.
- `Exec::rayon(&pool).with_budget(n)` is `Exec::new(&pool, n)`, and a pool is a
  `threadteam::Pool`, not a Rayon pool. `Exec::serial()` is unchanged.
- `Exec` is neither `Clone` nor `Copy`; `Exec::token()` lends the
  `threadteam` placement token for one call.
- The `tcontract` and `tcontract-exec` requirements are crates.io 0.5.0.
  `tcontract-capi` (unpublished) is a git dependency on the tcontract-rs
  v0.5.0 commit, and a `[patch.crates-io]` entry points the other two at that
  commit so the graph holds one copy of each.
- A multi-lane `Split` no longer refuses a child route that needs a packed
  SPMD team. tcontract 0.5 runs such a child barrier-free on its pool worker,
  so the forced strategy is honored and the result is unchanged.

## 0.3 changes

The 0.3.0 release tracks tcontract 0.3.0 and strided 0.6.0. Breaking changes:

- `cpueinsum::Exec` now carries a tcontract-exec `BatchPolicy`; force a
  scheduler with `Exec::with_batch_policy(BatchPolicy::new(BatchStrategy::…))`.
  `BatchPolicy`/`BatchQuery`/`BatchDecision` are re-exported from `cpueinsum`.
- `GroupedRoute` gained `Inner { threads }` and `Split { lanes, item_threads }`.
- `EinsumPlan`, `BinaryPlan` and `GroupedPlan` expose `workspace_len()` (the
  intermediate/backend element arena) and `native_workspace_req()` (the native
  tcontract `WorkspaceReq`, folded component-wise over the steps).
- `EinsumSpecRef` and `EinsumPlan::compile` plan from a borrowed description;
  `EinsumPlan::execute_bounded` replays from a caller `Workspace` and arena and
  refuses a short one before writing.
- `cpueinsum-blas` compiles `strided_basic::CopyPlan` for packing and no longer
depends on `strided-perm`, so its workspace layout is plan-time metadata.

See [`docs/worklogs/2026-10-18-cpueinsum-0.3.md`](docs/worklogs/2026-10-18-cpueinsum-0.3.md)
for the recorded measurements and the allocation/overhead gates.

## Status

Early. The plan, layering and gates are in
[tensor4all/tcontract-rs#62](https://github.com/tensor4all/tcontract-rs/issues/62);
the BLAS backend in
[tensor4all/cpueinsum-rs#1](https://github.com/tensor4all/cpueinsum-rs/issues/1).
The 0.5.0 release is prepared on tcontract-exec 0.5.0 through path dependencies
(temporary, until those crates are published). MSRV 1.89.

## License

MIT OR Apache-2.0.
