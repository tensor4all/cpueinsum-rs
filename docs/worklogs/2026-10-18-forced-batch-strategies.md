# Forced batch strategies on the BLAS routes (0.3.1)

Date: 2026-10-18. Branch `fix/blas-forced-strategy`. Predecessor:
`docs/worklogs/2026-10-18-cpueinsum-0.3.md`.

## Defect

0.3.0 resolved the `Exec` batch policy only on the grouped adapter. The two
other routes that are also serial loops of per-item vendor calls ignored it:

* the prepared binary adapter (`cpueinsum_blas::BinaryPlan`) refused a forced
  `WholeBatchVendor` but ran the vendor loop for a forced
  `Outer`/`Inner`/`Split` whenever `alpha != 0`;
* the N-ary route's CBLAS `StepBackend` neither honored a forced lane strategy
  nor refused `WholeBatchVendor`, because `StepBackend::plan` sees no `Exec`.

## Approach

One rule, one place. `Exec` is an execution property and plans are
executor-independent, so passing the policy into selection would break that
split; instead `cpueinsum::VendorLoopRoute::resolve` is the single decision —
`Auto`/`Sequential` keep the loop, a forced lane strategy belongs to the native
alternative, a forced whole-batch vendor route is the typed refusal — and the
grouped adapter, the prepared binary adapter and the N-ary driver all call it.

The N-ary driver needs one fact it cannot infer: whether a step is such a loop.
`StepBackend` gained the defaulted `vendor_loop(step)` (default `false`, so the
addition is non-breaking and an existing backend keeps serving every policy).
For a step that reports it, planning keeps the tcontract alternative beside the
backend step, and the plan resolves every step's admission once, after the shape
check and before `Scratch::reserve` and the first step: a refusal therefore
never follows a written intermediate, output or scratch element, and a refusal
that arrives from a later step still stops the plan before step 0 runs. Dispatch
then picks the backend or its alternative once per step. The alternative's
`WorkspaceReq` joins `native_workspace_req` only under a policy that routes to
it, so an `Auto` caller pays no extra workspace.

Planning both routes costs one tcontract plan per such step (plan-time memory
for that step's problem, retained for the plan's life). Execution is
branch-only: the admission resolves each native step's route once per call,
which is what tcontract's own execution does anyway, and the N-ary element
arena is unchanged. The alternative is built from the same `Problem` and
`PlanConfig` as an all-tcontract plan, so a forced lane strategy reproduces that
plan bitwise.

## Verification

New tests observe the route, never timing. `crates/cpueinsum/tests/backend_routes.rs`
drives a marker backend (it counts executions and zero-fills) through
`Auto`/`Sequential`/`Outer`/`Inner`/`Split`/`WholeBatchVendor`, and pins that a
backend reporting no vendor loop keeps running under every policy.
`crates/cpueinsum-blas/tests/forced_strategy.rs` repeats the N-ary cases with
the real `Blas` step behind a counting delegate and compares the forced-lane
result with the all-tcontract plan bitwise. `crates/cpueinsum-blas/tests/prepared_binary.rs`
covers both prepared entries (alpha nonzero, alpha zero, empty batch) and
compares native against the sequential vendor run bitwise on integer-valued
operands. The grouped tests are unchanged and stay green.

Both defect reproductions fail with the old behavior (driver routing disabled,
or `vendor_loop_route` replaced by `true`), and pass with the fix. The
plan-wide admission has its own reproducers: removing it leaves `Scratch` grown
by a refused plan, accepts a forced whole-batch vendor route on an empty batch
whose steps are all native, accepts a native `Split` refusal at the second step
of a packed chain, and lets the refusal follow an earlier backend step.

## Constraints

No version bump and no publication in this change; the patch release is a
follow-up release PR. BLAS stays confined to `cpueinsum-blas`, and `cpueinsum`
still needs no C toolchain (`cargo tree -p cpueinsum`).

One known ceiling is upstream: tcontract's initialized Faer route, given more
than one lane, runs each item with `faer::Par::Seq` and discards the decided
item threads, so a forced `Split { lanes, .. }` can be served by serial lanes
there instead of being refused. That is a tcontract defect (pinned `3239c86`,
`strategy/faer.rs`), fixed in tcontract rather than worked around here.
