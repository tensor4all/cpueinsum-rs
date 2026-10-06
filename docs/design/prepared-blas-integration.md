# Prepared BLAS integration

Status: implemented for tenferro-rs#2004; independent post-review completed and
contract corrections applied. Local gates passed; coordinated delivery is pending.

The merged native `BinaryPlan` and `GroupedPlan` do not accept the BLAS
`StepBackend`. Its overwrite-only contract remains unchanged. Prepared BLAS
support belongs in cpueinsum-blas, not in host alpha/beta postprocessing.
Core cpueinsum remains free of unsafe code and BLAS dependencies.

## Binary adapter

Add a concrete prepared `cpueinsum_blas::BinaryPlan<T>` with immutable native
alternatives and optional vendor metadata, accepting the existing `Problem`
and `PlanConfig`. Execution borrows `Exec`, operand slices/origins, alpha,
beta, accumulation source and an explicit caller-owned packing workspace.
Expose the required packing element count. Plans retain no pool, data pointer,
workspace or lease. Return the actual vendor/native execution route.

Initially vendor execution supports direct-D layouts and compatible Absent or
Output C. Output packing, Separate C and unsupported operation/layout
combinations select native execution at preparation. `no_materialize` also
selects native when vendor execution would copy a whole input; direct vendor
layouts remain eligible. Reuse existing input packing. Extend the owning CBLAS
helper for alpha/beta; preserve the old StepBackend's alpha-one/beta-zero behavior. Folding output conjugation must
also conjugate the scalars. Alpha-zero and other no-read scalar-degenerate
cases select an already prepared native branch before writing. There is no
execution-time capability decline, replanning or retry.

Validate bounds, origins, packing workspace and all fallible packing views
before mutation. Tighten the existing packing geometry's product/addition and
isize conversions locally; overflow must not wrap. Vendor numerical calls
execute inline on the coordinator, not inside an Exec install. If packing uses
parallel primitives, only that primitive is permitted to enter the supplied
pool; returning to the coordinator precedes the CBLAS call.

Fresh binary output remains `MaybeUninit<T>` through raw vendor writes.
Publication requires an injective output layout and exact physical coverage
of the entire supplied storage, including its origin: no untouched holes or
padding. Require no output reads, including beta-zero handling. Never create
an initialized slice before completion. Empty/degenerate cases use the
prepared native alternative. Initialization unsafe stays in cpueinsum-blas,
with explicit safety proofs and preserved provenance notices.

## Grouped adapter

Keep grouped output initialized. Prepare a vendor group only when every child
qualifies; otherwise prepare one whole-group native alternative. A mixed-plan
framework and fresh grouped output are unnecessary.

Before the first write, validate all jobs' checked arithmetic and byte spans,
output disjointness, runtime bounds, workspace capacity, packing-view
construction and selected native execution capability. A loop of individually
validating binary calls is insufficient. Gaps between jobs stay initialized;
they do not constitute proof of fresh backing-allocation completion. Preserve
original job IDs and deterministic failures. Vendor groups execute a
library-owned serial loop on the coordinator; native groups retain native
scheduling. Expose actual observed routes rather than a declared host label.

## Verification and delivery

Check four scalar dtypes against the shared reference, complex conjugations
and scales, Absent/Output/Separate C through their selected routes, zero/NaN
no-read semantics, reversed/gapped/broadcast inputs, exact fresh publication
and rejected holes, bounds/workspace/overflow failures, later-job failure
before any earlier output write, coordinator identity, and resource reuse.
Run the repository debug/release, clippy and formatting gates and verify the
core dependency tree still excludes BLAS/C toolchain requirements.

This is one coordinated prerequisite PR. The integration consumes its reviewed
remote revision; no merge or publication is authorized. Independent
pre-review required concrete preflight/full-write contracts and accepted this
owning-layer direction; independent post-review must cover the finished diff.
No speedup or allocation-free claim follows from the architecture alone.
