# Prepared binary/grouped verification (2026-10-06)

Implements issue #4 from base `e9794de`, branch `prepared-binary-grouped`.
No tenferro or tensor4all production paths are changed. Existing N-ary plans
and overwrite-only StepBackend remain intact. The new facade contains no unsafe
code; the library now enforces `#![forbid(unsafe_code)]`.

## Exact owning-layer prerequisites

- tprims-rs#71 already merged at `bb91172`: caller-owned execution resources.
- strided-rs#290: terminal pointwise pointer arithmetic correction,
  `722dc6bece0c845ee612fa21242521c2e5ea83ac`.
- tprims-rs#72: safe fresh-output boundary and shared route reporting,
  `c893a7b3d54e16c4db3a190e2cf78d2575c24d65`, based on merged #71 and the
  corrected strided pin.

The manifest and lockfile use those published git revisions, not local path
patches. Required merge order is strided #290 → tprims #72 → this change.
None of these new PRs has been automatically merged.

## Requirement-to-evidence map

| Issue #4 requirement | Implementation / executable evidence |
|---|---|
| Immutable binary preparation separates metadata from alpha/beta and buffers | `prepared.rs::BinaryPlan` owns only a tprims Plan; each execution lends Exec and views/slices. |
| Absent/Output/Separate semantics | `prepared_binary.rs` modes, typed source/error cases and initialized view paths; `prepared_semantics.rs` all four dtypes and sixteen A/B/C/D conjugation masks under default/packed configurations. |
| Fresh storage remains MaybeUninit until successful completion | Owning tprims slice API proves reduced injective cardinality and exact physical span; fresh reference tests, padding/repeated-label/rank-zero/empty rejection/completion regressions in the prerequisite. No unsafe conversion in cpueinsum. |
| Zero scales, NaNs, empty K, negative strides/origins | `prepared_semantics.rs` K=3/513, empty-K/zero-scale NaNs, reversed/gapped operands and pooled complex broadcast/reversal; `prepared_binary.rs` negative origins and failure sentinels. |
| Aliasing and failure mutation | Safe slice/view borrows exclude Separate C/A/B overlap with unique D; lower raw/C ABI tests reject overlap and mismapped C==D. Bounds, mode, capability and group-overlap failures occur before writes. Runtime backend failures may partially write; no rollback promise. |
| Prepared observable numerical routes | Binary entries return owning-layer route; fresh Faer uses its preprepared packed alternative. Geometry is shared with execution, not duplicated here. |
| Group owns scheduling rather than looping one-shot binaries | `grouped.rs` caches per-job plans and one sorted disjoint output metadata order, prevalidates all jobs/routes, safely splits shared D and dispatches one bounded outer partition scheduler; serial children borrow the same workspace. |
| Group disjointness, original identity and exactly once | `grouped.rs` tests overlap/checked-byte overflow, late bad jobs in both output orders, shared heterogeneous input ranges and unchanged holes. Parallel accumulation (D initially one, expected 65) detects duplicate execution. |
| Budget and route | Explicit four-worker pool tests budgets 1/2/4; GroupedRoute records actual outer lanes, PoolStats verifies one outer entry and no SPMD child broadcasts, including nested caller execution. |
| Warm reusable resources | Two independent traced arenas have real nonzero retention and no warm growth; grouped all-dtype warm tests and the allocation measurement below distinguish dispatch metadata from numerical workspace. |

## Local gates

Against the actual published pins, all passed:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --release --workspace
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
cargo +1.89.0 check --workspace --all-targets
cargo tree -p cpueinsum
```

The core tree contains no BLAS/provider/C-toolchain dependency; BLAS remains
in cpueinsum-blas. Tests use the existing naive reference. Environment:
`CARGO_BUILD_JOBS=16 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` (compile jobs
are not numerical threads). Logs: `/tmp/cpueinsum-stage2-pinned-*.log`.
The final fresh-contract documentation wording was additionally checked with
core doctests and rustdoc `-D warnings`. There is no hosted cpueinsum CI.

## Warm allocation measurement — explicit 1T, not timing

Machine: Linux x86_64, AMD EPYC 7702P 64-Core Processor; rustc
`1.99.0 (b940084d7 2026-09-28)`, release profile. Backend is explicitly
`Exec::serial_with_workspace` with retained traced `ArenaProvider`; the harness
asserts budget **1** and actual `Packed { Serial, width: 1 }`, not the machine's
all-core default. No pool or BLAS backend is used.

An isolated temporary executable counts allocation/reallocation calls with a
System-delegating global allocator. This is the only unsafe measurement code;
it is outside the library/repositories and does not perform numerical pointer
operations or initialization casts. Plans, buffers and three warmup rounds are
outside each counting window; each window covers 64 facade executions.

- Binary: f64 compact column-major 8×8 times 8×8, forced packed, alpha=1,
  beta=0, initialized and fresh destinations.
- Group: shared inputs, 8×8×8 and 4×8×8 jobs, plus a validated empty job;
  output prefix length 96 and untouched tail through length 128, serial children.

| Warm case (64 calls) | Allocation/reallocation calls |
|---|---:|
| Binary initialized | 0 |
| Binary fresh | 0 |
| Grouped serial | 64 (one borrow-only item vector per execution) |

Cold arena growth: 2 events. Warm arena growth: 0. Retained bytes: 49,712;
leased bytes at quiescence: 0. Numerical values, actual routes, unchanged group
output tail and unchanged complete arena statistics are asserted. These are
scoped allocation counts, not an allocation-free promise for every layout,
dtype, route or parallel group, and not timing/throughput/speedup results.
Parallel dispatch additionally allocates a lane vector; no parallel allocation
measurement is claimed. No per-job arena, view metadata or dense product is
constructed.

Measurement command (no dependency path patches):

```
cd /tmp/cpueinsum-prepared-allocation-probe
CARGO_TARGET_DIR=/tmp/cpueinsum-prepared-allocation-probe/target \
CARGO_BUILD_JOBS=16 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

Output: `/tmp/cpueinsum-prepared-allocation-final.log`; temporary harness source
`/tmp/cpueinsum-prepared-allocation-probe/src/main.rs`, SHA-256
`77d14b75eb29a9f6d965befd9fe3cbf807db721eba4290263e9f2d5c54f57908`.
Its cpueinsum dependency points to this checkout; tprims and strided resolve the
published pins above. The temporary probe is intentionally not library/test
infrastructure and is not required by the repository's gate.

## Review and limitations

Completed gpt-6.1-sol independent pre-review rejected reference-before-write
leaves and accepted the corrected design plus default-off fresh capability.
The owning-layer post-review found mutable-provenance, strided terminal advance
and C batch admission defects; corrections live in the owning PRs. A completed
focused gpt-6.1-sol corrective review closed all three, conditional on exact-pin
gates which subsequently passed. The finished cpueinsum facade receives its
own completed gpt-6.1-sol post-review before publication. That completed facade
review found no BLOCKER/IMPORTANT implementation issue and two MINOR doc issues:
the equations omitted outer op_D, and the new README example was not executable.
The equations now include op_D; the example was moved (not duplicated) to
GroupedPlan's runnable rustdoc, linked from README. Debug/release core doctests
and rustdoc warnings are checked after those documentation-only corrections.

No Miri/sanitizer/interpreter safety proof, CPU speedup, GPU result or unchanged
hosted CI status is implied by numerical successes. Local code reuses existing
APIs and reference tests; no third-party body was copied and no upstream notice
was removed. Existing inherited tprims/strided lineage still applies.
