# 2026-11-25: work-weighted lane placement of the per-job route (#80)

## Change

Under `Auto`, without batched strided runs, `GroupedPlan` used to split the
jobs into `min(width, jobs)` lanes of equal job **count** in output order and
run every job on one thread (`Outer { lanes }`), so a job holding most of the
work ran serially while the rest of the budget idled or ran small jobs. It now
places the jobs by estimated work on lanes of their own widths
(`crates/cpueinsum/src/grouped/placement.rs`), and a job on a wider lane hands
its tcontract child that width under the caller's `Auto` policy.

Unchanged: the width (`Exec::width_for` over the call's estimated serial time,
one section), the cap on job-holding lanes (`min(width, jobs)`, the historical
lane count, not raised), every forced strategy, the batched-strided-runs route,
and the serial route of a call that does not pay for a second thread.

## Provenance

Ported (close translation, adapted) from symtn-rs,
<https://github.com/Ryo-wtnb11/symtn-rs>, commit `09a3ebb`
(`09a3ebbd`, "Let workers first-touch every fresh GEMM output (#2582)"),
MIT OR Apache-2.0, Copyright (c) 2026 Ryo-wtnb11:

| cpueinsum | symtn-rs `09a3ebb` |
|---|---|
| `placement::item_time` | `symtn-dense/src/lanes/placement.rs` `item_time` |
| `placement::place` | `symtn-dense/src/lanes/placement.rs` `place` (the search; no cache, one stage) |
| `placement::equal_shares` | `symtn-dense/src/lanes/placement.rs` `equal_shares` |
| `placement::Moldable`, `Reservation`, `layout` | `symtn-dense/src/lanes/placement.rs` `Moldable::place`, `Reservation`, `layout` |
| `placement::compact`, `Placement` | `symtn-dense/src/lanes/placement.rs` `compact`, `symtn-dense/src/lanes.rs` `Placement` |
| `grouped::recycle`, `run_placed`'s `fork_sized` lanes | `symtn-dense/src/lanes/runner.rs` `recycle`, `run_placed` |

The upstream MIT text is `crates/cpueinsum/third_party/symtn-rs/LICENSE-MIT`;
both crates are `MIT OR Apache-2.0`.

## Cost model: what comes from `t4a-cost`/tcontract, what is ported

| quantity | source | ported? |
|---|---|---|
| serial time of job `j`: `2 m k n (x4 complex) ns_per_flop` | `tcontract::ns_per_flop` (tcontract-exec `host`), the formula `lane_decision` already used for the whole call | no |
| the width job `j`'s child takes on a lane of `w` slots | `tcontract::Plan::execution_width(exec, w, alpha, contract)`: tcontract's own decision; for a Faer child the argmin of its `t4a-cost` `ItemTime` row fitted for the host class, for a Packed child its grid | no (replaces symtn's `block_kernel/gemm.rs::grant`, which composes the same answer from batch decisions) |
| a job's modeled time at a granted width `g` | `t / g + dispatch_cost(g)` (`t` at `g = 1`) | **ported** (`item_time`): the fitted per-width time behind `execution_width` is crate-private in tcontract 0.13 (`strategy/faer_time`), so it is not readable here; ideal division is what `t4a_cost::ItemTime` itself gives without a fitted row |
| dispatch cost at a width | the threadteam token | no |

Follow-up: exposing a plan's fitted item time at a width from tcontract would
let the score use the `t4a-cost` row instead of ideal division.

## Placement, bound, determinism

Jobs longest first (ties to the lower stored index). Candidates: equal shares
over every lane count from the cap down to one (list scheduling over modeled
item times), and symtn's two large-job reservations (shared and exclusive).
The lowest modeled makespan wins; ties keep the larger lane count, then equal
shares, then the shared reservation. Everything is integer nanoseconds and a
pure function of the job list and the context, so the placement is
deterministic.

**Bound.** Every lane holding two or more jobs has a modeled time at most the
least lane time plus the largest item time (at any lane's share) of a job on
it. The equal-share placement at the cap satisfies it by the list-scheduling
argument; any other candidate that does not is discarded.
`placement/tests.rs` checks it on fixed lists, with exact lane times.

**Outputs.** A lane runs its jobs in stored order and every job writes only its
own output slice, so a placement changes which thread computes a job and with
how many threads, not what it computes: `tests/grouped_placement.rs` checks the
f64 (packed child) and c64 (default child) outputs bit for bit against the
serial per-job route for all three entry points at width 8, and the benchmark
driver checks every measured call against the serial route at 16, 32 and 64
threads. No accumulation order changes in cpueinsum; a child's own order is
tcontract's.

**Allocation.** The placement and its tables are cached in the plan, keyed by
every input of the search (budget, slot count, cap, batch policy, team kind,
zero alpha, output contract and the token's dispatch-cost row). A warm call on
the same context places nothing (`grouped/tests.rs`) and, when every lane is
one thread wide, allocates nothing (`tests/alloc_placed.rs`; the historical
`Outer` route allocated its lane-chunk vector). A child on a wider lane runs on
a team that may allocate inside tcontract/threadteam (one or two reallocations
measured), within the per-lane bound the parallel route already had. A plan
built per call (as symtn builds them) searches once per call.

## Pre-existing finding, not changed here

On the serial route, a c64 child's `execute_uninit` output differs from its
`execute_into` output in the last bit for some elements (1765 of 40704 in the
`tests/grouped_placement.rs` fixture, on one thread). The tests therefore
compare each entry point with itself.

## Measurement

See the pull request for the protocol and the table (keyaki, AMD EPYC 7702P,
not the Xeon Platinum 9242 the #80 table came from).
