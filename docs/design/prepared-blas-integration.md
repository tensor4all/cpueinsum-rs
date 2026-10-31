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

`execute_into`/`execute_into_accum` keep grouped output initialized. Prepare a
vendor group only when every child qualifies; otherwise prepare one whole-group
native alternative. A mixed-plan framework is unnecessary.

Issue #53 adds `GroupedPlan::execute_uninit(.., out: &mut [MaybeUninit<T>], ..)`:
the shared native checker validates `beta = 0` and exact coverage by the jobs
before any write, so a gap or a nonzero beta is a typed error with the storage
untouched. Every vendor job is then preflighted with an Absent source before the
first write; the vendor loop splits fresh storage by increasing output offset and
publishes one initialized receipt per job, and tcontract's `execute_uninit_with`
publishes the whole destination once after every job succeeded. The initialized
vendor order and its error selection are unchanged.

Before the first write, validate all jobs' checked arithmetic and byte spans,
output disjointness, runtime bounds, workspace capacity, packing-view
construction and selected native execution capability. A loop of individually
validating binary calls is insufficient. Gaps between jobs stay initialized in
the initialized entry points; the fresh entry point refuses them instead of
treating them as completion. Preserve
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

## Batch policy on every BLAS route

Every route here — the bare `StepBackend`, the prepared binary adapter and the
grouped adapter — is a serial loop of one library call per item. Such a loop has
no outer lanes, no item threads and no whole-batch call, so all three resolve
the caller's `BatchPolicy` with one rule, `cpueinsum::VendorLoopRoute`:
`Auto` and `Sequential` keep the loop (the historical choice, and exactly one
serial coordinator), a forced `Outer`/`Inner`/`Split` runs the route's tcontract
alternative, which resolves the same policy (and can still refuse a `Split` it
cannot honour). The rule lives in one place because the same policy must not
resolve three different ways.

The batch policy is an `Exec` property, so it is resolved at execution, not at
planning: a `StepBackend` step declares the same serial-loop shape with the
defaulted `StepBackend::vendor_loop`, and the N-ary driver keeps the tcontract
alternative planned beside every such step. The plan then resolves **every**
step's admission in one pass, after view validation and before the element arena
is reserved and before the first step: backend steps through the shared rule,
and every native step — including the alternative a forced lane strategy selects
— through its own `Plan::execution_route`, which refuses a forced whole-batch
vendor route before its empty-output shortcut and a forced lane strategy the
native route cannot honour (tcontract's packed driver rejects lanes *and* item
threads). A refusal therefore never follows a written intermediate, output or
scratch element, and a refusal that originates in a later step still stops the
plan before step 0 runs. A backend step that is not such a loop (the default)
serves every policy and needs no native admission. A backend step contributes
the alternative's `WorkspaceReq` to `native_workspace_req` only under a policy
that routes to it.

That admission costs one route resolution per native step per execution, which
is what tcontract's own execution already does; it adds no element-arena
workspace. It does not close one upstream gap: tcontract's initialized Faer
route, given more than one lane, discards the decided item threads instead of
refusing, so a forced `Split` can end up as serial lanes there. That belongs to
tcontract (pinned `3239c86`, `strategy/faer.rs`), not to a workaround here.
