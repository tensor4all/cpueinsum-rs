# #46 broccoli scheduling diagnostic (not official performance)

## Scope and source

Baseline: `adcf0c4e090af1c7753f84facfb33342f59c3890` (main after #56).
Read the complete issue #46 body and all three comments (including Ryo-wtnb11,
2026-10-31T04:17:23Z). #57 is open and remains out of scope. No production
scheduler change will be submitted; a patch under this directory is a diagnostic
candidate only. No merge, release, dependency upgrade, RunPod, or evo.

The current Auto uses total estimated time to choose lanes, capped by jobs and
budget, always item_threads=1. Inner grants one lane and the full item budget.
Split grants min(requested lanes, jobs, budget) lanes and budget/lanes item
threads. child_context converts width=1 to Exec::serial, otherwise limits the
token and forces Inner. faer_auto tests the single item's width first: inner
when width>1, otherwise outer lanes from total work. Width is a grant, not proof
of utilization; record CPU utilization separately.

## Workloads (f64, forced faer, compact column-major, disjoint inputs/outputs)

Dimension order here is (m,k,n); multiplication is A[m,k] B[k,n] -> D[m,n].
- A: two (1024,256,1024) jobs: 536870912 MAC, 1073741824 conventional FLOP.
- B: 16384 (32,32,32) jobs: exactly the same MAC/FLOP count.
- C: one (1024,256,1024) plus 8192 (32,32,32) jobs: same MAC/FLOP count;
  this is the issue's dominant-plus-many-small regression-side supplement.
- Per-job controls: one large and one small job, measured separately; these are
  isolated-job timings, not intrusive instrumentation inside the group.
B has more data traffic and dispatches than A: equal arithmetic work is not
an equal working set or an end-to-end symtn reproduction. No contraction order
search, sparse symmetry logic, or tenferro 0.7.1 baseline is included.

## Candidate (review before implementation)

A minimal patch in lane_decision only: on Auto, calculate the estimated time of
the largest job, and ask the existing Exec::batch_decision using a single-item
BatchQuery with outer_only_when_item_serial=true. If item_threads>1, use that
one-lane width; otherwise retain the original Auto outer calculation using the
actual total work. Use the existing WorkModel cost (complex weight where
applicable), no new threshold, allocation, dependency, public API, or tuning.
This is a *largest-item group-wide* prototype, not a completed heterogeneous
per-item scheduler: C explicitly tests the risk of serializing its small jobs.
Only initialized f64 output is exercised here. No claim of release readiness.
The checked-in patch, its SHA256, and the hashes of both benchmark executables
identify the candidate; the library source is restored before measurements.

## Build, affinity, and hardware

Release/bench profile (opt-level=3, thin LTO, codegen-units=1), unchanged Cargo.lock,
Rust version and resolved dependencies recorded. Build sequentially with
`nice -n 5 cargo ... -j8`, pinned to observer cores. No concurrent build or
measurement. Run binaries directly (not cargo bench, which can introduce a
runner). Set PlanConfig.faer_limit=FaerLimit::NONE and check Algorithm::Faer on a
separate plan for every geometry. Hold a threadteam owned Team across warmup
and timing, build Exec::from_token(team.exec(width), workspace, policy), and
assert carries_team and actual team width equal the request. This matters:
tcontract's faer route cannot use inner parallelism on a plain pool token.
Keep pin_participants(false); all workers inherit the whole taskset mask.

Discover and record L3 shared_cpu_list from sysfs; form measured sets from whole
L3 domains sorted by CPU id. On this host domains contain four cores, so the
expected measured sets are 0-7, 0-15, 0-31, 0-63 (2/4/8/16 L3 domains).
Observer is the last L3 domain, 60-63 here. The agent and all observer commands
are pinned there. 64T shares observer cores and is **reference only**. The user
explicitly waived single-L3 confinement and 64T observer separation for this
diagnostic. 8-32T retain separation. Child workers inherit the measured mask;
record actual /proc/self/status affinity and nice level. No per-worker pinning.

## Timing and validation, declared before results

Explicit widths 1 (overhead/control), 8,16,32,64. Strategies Auto, Inner,
Split{2}, Split{4}; add candidate Auto as the fifth arm. Seven rounds, alternate
forward/reverse arm order and rotate condition order between rounds (no running
all repeats of one arm together). Each subprocess constructs plan/pool/input
once outside timing, validates every output against a rank-one analytical
reference, warms for >=250ms, then times repeated complete execute_into calls
for >=250ms in a separate CPU diagnostic window. The primary timing is one
complete execute_into call immediately after warmup (before that CPU window),
recorded as integer nanoseconds; cell time is the median of seven primary
observations, ms as exact decimal nanoseconds/1000000, no rounded or interpolated
entries. Relative value is each median divided by the fastest median for the
same workload and width; report an exact rational ratio (1.00 for the fastest)
so a nonterminating decimal does not need rounding. All errors, failures, and discarded attempts remain
in the raw log. No selective reruns, exclusions, or fallback settings.

Use varying exactly representable rank-one inputs (A row factor times a
contracted-index sign, B contracted-index sign times column factor). Check all
outputs after warmup and after timing, tolerance 1e-11 relative/absolute.
Black-box operands/output. Per-process CPU ticks (/proc/self/stat, including
worker threads) over each timing window give aggregate CPU% (100%=one core);
include busy-spin/dispatch and document tick quantization. Record GroupedRoute
(actual selected lanes/item width), lane job counts (contiguous near-even
partition), and isolated-job controls. Capture per-CPU /proc/stat before/after
and pre/post idle windows, plus process snapshots; warn on pre/post busy >5%.
Raw failures invalidate that cell. Timing noise: max/min >1.25 is flagged
unstable; do not cherry-pick. A qualitative large-job reproduction requires
Auto/Inner >1.25 and Auto/best-Split >1.25 with consistent paired direction
(at least six of seven rounds) at >=two of 8/16/32T; 64T never decides the gate.
Small-job candidate regression is inconclusive if unstable; otherwise flag
median candidate/Auto >1.10 or six/seven slower by >10%. This diagnostic does
not establish a production non-regression gate on Ryo's workload.

## Attribution and limits

(a) Compare A Auto's two serial job grants, CPU%, width-invariant time and
isolated serial job time against Inner/Split at the same shapes/data.
(b) Compare selected and nonempty lanes, balanced A/B vs heterogeneous C, and
Split{2} vs Split{4} (A has only two jobs, so these should clamp identically).
(c) Same-shape/same-data changes that speed up with inner width refute bandwidth
as the sole explanation; they do not prove lack of bandwidth limits. perf is
not installed on this host (checked before timing); no hardware bandwidth
counter or STREAM claim will be fabricated. Report memory footprint/arithmetic
intensity and any remaining uncertainty rather than identifying bandwidth from
CPU% alone.

Existing overhead_smoke is a small 1T overhead gate; the existing cpueinsum-bench
corpus does not expose this heterogeneous grouped strategy grid. Add one
standalone bench and a stdlib-only recorder here instead of extending unrelated
campaign infrastructure. Harness self-checks cover analytical results, pool
width, shape work equality, strategy routes, and recorder aggregation. Run local
fmt, workspace clippy, debug/release tests and relevant docs before one finished
push/PR. Commit benchmark, protocol, raw data, generated tables and report.
