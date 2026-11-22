# Caller-supplied grouped child plans (#79)

## Decisions

- `GroupedPlan` stores its children as `Arc<Plan<T>>`. The existing
  constructors (`new`, `new_with_ops`, `from_prepared`) share the new
  construction with the built-in builder (`ChildKey::build`), so their
  behaviour, child order and error ids are unchanged. They skip the
  supplied-plan check, since their children are planned from the key itself.
- `GroupedPlan::from_prepared_with(prepared, ops, config, child)` calls
  `child(&ChildKey)` exactly once per distinct geometry, in child-index order,
  and keeps the returned `Arc`s. Nothing is rebuilt per call or per job.
- Child-index order (the assignment the issue asks to be deterministic) is the
  distinct raw `(rows, contracted, cols)` in ascending lexicographic order. Jobs
  of one geometry share its child; the tie-break among them is the lowest
  original job id, which is also the id an error names. The order is
  independent of the job order, so the same shape set gives the same key
  sequence and the same assignment.
- `ChildKey` carries dtype, geometry, `(lda, ldb, ldc)`, the three ops and the
  effective child config (`fresh_output` cleared, as before). Today's jobs are
  compact, so the leading dimensions are `max(rows,1)`, `max(contracted,1)`,
  `max(rows,1)`; they are in the key now so it keeps its meaning after the
  tcontract-rs#209 descriptor migration. `Hash` covers everything but the config
  (tcontract's `PlanConfig` has no `Hash`); `Eq` compares the config too.
- Supplied plans are checked: `plan.problem()` must equal `ChildKey::problem()`
  and the plan must not be a rank-k plan, else `GroupedError::ChildMismatch`
  with the lowest job id of that geometry, before any execution. The config a
  plan was built under is not observable on `Plan` and is not checked.
- The closure returns `tcontract::Result<Arc<Plan<T>>>`; its error becomes
  `Error::Contract { step: <lowest job id of that geometry> }`. Construction
  stops at the first failing child in child-index order, as `from_prepared`
  already did.

## Not in this change

- The t4a-desc job descriptor migration (issue Update 2026-11-20) is left for a
  separate PR, by the maintainer's decision for this run.
- Work-weighted (LPT) lane placement stays with #80; lane splitting is unchanged.
- Strided runs (#78) have not landed, so run children do not exist yet.

## Verification

- `tests/grouped_child_plans.rs`: once-per-geometry (200 jobs / 3 geometries),
  bitwise equality with the rebuilt path and a naive GEMM (serial and 2/3 outer
  lanes), one cached `Arc` serving two plans and repeated calls with no rebuild,
  same keys for permuted job lists, mismatched shape and ops rejected before any
  write, closure error id, key equality/hash.
- Allocation fixtures: one extra request per distinct child (its `Arc`), +24
  bytes per child on 64-bit. `grouped_prepare_alloc` shared geometry 3 -> 4
  requests independent of job count; distinct geometry bound `3 + G` ->
  `4 + 2G`. `cpueinsum-blas` `alloc_free`: the native grouped plan is 4
  requests, the vendor plan stays 3, so vendor = native - 1 for a nonempty group.
- No timing claim is made.
