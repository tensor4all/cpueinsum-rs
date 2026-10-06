# Prepared binary and grouped CPU contraction

Status: accepted for implementation after independent pre-review. Numerical
leaf safety remains an implementation verification requirement. Implements cpueinsum-rs#4,
the next prerequisite for tenferro-rs#2004 after tprims-rs#71.

## Scope and owning layers

cpueinsum becomes the prepared binary and grouped entry point, not another
numerical kernel implementation. tprims owns contraction kernels, layout
validation, scalar semantics, and execution geometry. cpueinsum owns grouped
job descriptions, aggregate validation and scheduling. No changes to tenferro
or tensor4all consumers in this stage; no contraction-order search, GPU, new
provider abstraction, vendor batch backend, or hidden dense result temporary.

Baseline: cpueinsum e9794de; its workspace tests pass. Its existing tprims pin
is fed859ac, before the caller-owned workspace change. The new base is the
merged tprims bb91172. The strided revision must match tprims's own manifest.
The existing N-ary StepBackend stays overwrite-only: this feature does not
force alpha/beta/uninitialized support onto external backends which have no
such capabilities. New prepared binary/grouped plans use tprims directly.

## Actual current seams

- binary.rs builds a tprims Plan per call, CSpec::Absent, alpha=ONE.
- EinsumPlan already owns immutable per-step plans; Scratch holds intermediates.
  It calls execute_slices to avoid allocating view metadata per binary step.
- tprims Plan has execute_into_accum and the three CSpec forms. Its safe
  execute_slices is overwrite-only; execute_raw supports all C modes but is
  unsafe. There is no safe MaybeUninit output entry.
- tprims Plan::report describes the prepared algorithm, not an executed packed
  route. driver::execute_capped resolves packed lanes/grid before writes, but
  that decision is private.
- tprims contract_batched is a real outer scheduler with all-item layout
  preflight and serial children when items >= width. With fewer items it runs
  inner parallelism instead. It lends the outer workspace to serial children.
- tensor4all's production grouped bridge uses flat shared lhs/rhs/output
  buffers with element offsets and heterogeneous (rows, contracted, cols),
  compact column-major per job. It currently requests overwrite. tenferro's
  descriptor has the same six facts. There is no payload packing in the bridge.

## Required narrow tprims prerequisite

No unsafe is permitted in cpueinsum. Putting a raw call or MaybeUninit-to-T
cast there, wrapping it in a new local bridge crate, zero-filling a fresh D,
or computing a temporary product would violate the owning-layer boundary.
Extend the existing tprims Plan surface instead, in a separate dependency PR
(the repository boundary is a real reason to split). cpueinsum pins the tested
prerequisite commit; dependency status is explicit in its PR. Do not merge
anything without user authorization.

1. Safe slice accumulation, parallel to execute_slices: a typed source
   distinguishes absent, Output, and a Separate slice+origin. It validates
   A/B/D spans, C mode and C span before any write; Absent rejects nonzero beta.
   beta=0 reads no C value, but supplied metadata/source compatibility is still
   checked. Input/output aliasing is prevented by slice borrowing; raw APIs
   keep their existing explicit alias contract.
2. Safe MaybeUninit slice execution: D remains &mut [MaybeUninit<T>] until the
   call succeeds. Input and Separate C stay initialized slices. Output mode
   is permitted only when beta=0 (otherwise reject before writes). Before
   execution prove the validated injective D layout covers the entire supplied
   slice exactly: addressed span shifted to the origin is [0,len), and its
   logical element count equals len. Count is the checked product of extents
   of roles.m + roles.n + roles.h, each reduced output label exactly once:
   rank-0 counts one, any zero extent counts zero. It is NOT the product of
   original D dims (repeated labels can mean diagonal-only writes).
   This deliberately restricts fresh
   uninitialized output to dense physical coverage, including dense permutations
   and reversals, without gaps/padding. Reject noncompact output with a typed
   unsupported/layout error, not zero-fill. On success return &mut [T] from
   the owning-layer cast; on error or unwind no initialized slice escapes.
   Empty output succeeds only for an empty D slice. The proof relies on the
   existing Problem reduced-output injectivity validation and checked cardinality.
   This proof is necessary but not sufficient: numerical leaves must also
   avoid creating initialized references before writing.

   Fix ElementPlan's overwrite and Separate-source write leaves to construct
   a MaybeUninit destination view and write MaybeUninit::new(value) through
   existing generic strided map/zip operations. In-place leaves keep an
   initialized view because their contract requires live D. No new coordinate
   traversal, zero-fill, or dense temporary is introduced.

   Faer 0.24.4 Accum::Replace cannot be used indiscriminately for fresh D:
   row-major matvec forms &mut dst[i] before storing. Introduce PlanConfig's
   explicit fresh_output preparation capability (default false). When requested,
   a Plan whose initialized strategy is Faer prepares one packed overwrite
   alternative at construction, using the same validated Problem/config and
   bounded packing. BinaryPlan always requests this capability; the C boundary
   does too because it promises write-only raw output. Ordinary initialized and
   N-ary plans do not pay for an unused alternative. An actual allocation
   regression test found unconditional preparation changes small Faer plan
   construction from its existing <=2 allocation contract to 59 allocations;
   do not relax that gate. Fresh execution on a Faer Plan without preparation
   returns a typed Unsupported error before writes with the remedy to request
   fresh_output at plan construction. The raw entry's safety contract requires
   initialized D for an unprepared Faer route; freshly prepared raw plans choose
   the fresh strategy. All safe typed entries enforce this, never trust a caller
   bool to manufacture initialized storage.
   It owns metadata only and is invoked for fresh product execution, never
   planned or probed during execution. Packed and Elementwise plans need no
   duplicate. Report the selected prepared strategy explicitly. Audit packed
   writeback/direct-output leaves for the same reference-before-write issue
   before enabling them. Default safe initialized execution retains Faer.
   The existing raw write-only boundary must use the same preprepared fresh
   route when beta=0, and when Separate C does not require reading old D;
   Output/nonzero beta uses initialized semantics. Raw safety docs distinguish
   write-only from read-before-write and report docs cannot claim raw execution
   always uses the initialized PlanReport algorithm. No unsupported claim of
   Faer uninit safety or an execution-time serial/provider fallback.
3. Executed route observability from the owning layer: a small ExecutionRoute
   reports Empty, OutputOnly (including in-place identity), Elementwise, Faer,
   or Packed { Serial / BatchLanes / Cells / Spmd, active width }. A pure
   Plan::execution_route(exec, alpha, output_contract) resolves availability
   before writes, where the small output-contract enum selects initialized
   versus fresh prepared execution. This enum describes required storage
   semantics, not a runtime provider choice; safe typed entries choose it
   themselves. Fresh capability/C-mode validation is independent of this pure
   geometry query and still precedes execution.
   Extract packed geometry/routing into one private helper used by both the
   query and driver; do not duplicate partition policy downstream. Faer and
   Elementwise report their prepared numerical route, not an invented active
   width; packed active width is exact. The report is not a worker-ID/affinity
   promise and does not claim output-only passes are always nonempty.
   Scalar zero or empty K resolves OutputOnly; nonzero/NaN alpha follows the
   prepared product route. Empty output still validates buffers in execution.
   No reservation, retry, fallback provider, or busy error is added. The fresh
   packed alternative is explicitly prepared, not a runtime failed-provider
   fallback.

Validate/execute use the same route resolution. SPMD query checks same-pool
worker status only; the existing execution-time blocking gate serializes
concurrent broadcasts. Query is not an acquired workspace or reservation.
If ownership of the executor or calling worker changes, query again on the
actual executing thread. cpueinsum resolves and executes together.

## Prepared binary public contract

Introduce BinaryPlan<T> wrapping an immutable tprims Plan<T>, constructed
with fresh_output preparation enabled from a tprims Problem (re-export the semantic descriptors already exposed by
cpueinsum::tprims_contract) and optional PlanConfig. Avoid duplicate public
layout/conjugation/C-mode enums; labels and all four operand ops are fixed by
Problem. No executor, buffer pointer, arena or lease is stored in BinaryPlan.

Methods:

- execute_into(exec, alpha, a, b, d): overwrite, beta=0, no previous D read;
  initialized strided views must match the prepared layouts.
- execute_into_accum(exec, alpha, a, b, beta, source, d): initialized update
  with a source matching the planned C mode; absent source accepts beta=0 only.
- execute_slices variants: same contracts with fixed planned layouts and
  runtime slice origins; no view allocation on warm repeated execution.
- execute_uninit_slices: the typed lower-layer entry, dense physical D
  coverage only, returns initialized D and its ExecutionRoute after success.
- report(): immutable PlanReport; execution methods return ExecutionRoute
  after success. Their scalar/resource route is resolved before writes and
  errors preserve typed lower-layer sources.

View and slice entries share owning-layer C-mode validation: Absent accepts
no source only at beta=0; Output and Separate require matching sources for
nonzero beta; overwrite can omit the C source in any plan. Validation of an
explicitly supplied source still occurs when beta=0. Introduce the smallest
typed absent-source representation needed in the owning accumulation enum,
not a separate cpueinsum meaning rule. Nonzero/NaN beta under Absent is rejected
in view, slice and raw entries (an intentional early-development correction
to raw's previous silent beta discard).

Use existing cpueinsum Error::Contract { step:0, source } for lower errors.
Do not introduce a new backend error string. Existing contract_into remains
one-shot overwrite and can delegate through BinaryPlan; N-ary behavior and
StepBackend are unchanged. Re-export ArenaProvider for discoverable serial
reuse rather than inventing a cpueinsum workspace facade.

Validation and semantics:

- Validate layouts, origins, source, bounds and capability before output-empty
  or scalar-zero shortcuts. Absent/nonzero beta is an error, including NaN beta.
- beta=0 does not read C/D; alpha=0 or empty K does not read A/B values.
  Values may contain NaNs; do not replace algebraic zero shortcuts with
  multiplication by zero. Runtime metadata is nevertheless validated.
- Separate C and D are disjoint in safe APIs. Output uses the unique D borrow,
  never constructs an immutable alias of it. Output-mode uninit/nonzero beta
  is rejected. Negative strides and offsets are preserved, not normalized.
- Validation/resource/capability errors leave caller D unchanged. Runtime
  numerical/backend failures or panics may partially write D. No rollback.
  MaybeUninit storage stays typed on failure, even if some bytes were written.

## Grouped plan public contract

Introduce GroupedGemmJob with six checked-on-plan facts: lhs_offset,
 rhs_offset, out_offset, rows, contracted, cols (element units). GroupedPlan<T>
prepares compact column-major per-job ordinary tprims Plans (initialized-only;
no unused fresh-output preparation) with C mode Output, so
execute_into is overwrite (beta=0) and execute_into_accum explicitly reads
previous D for nonzero beta. Separate C is part of the general BinaryPlan
contract, not a new grouped descriptor requirement; do not add a speculative
c_offset. The six-fact job constructor matches the actual downstream descriptor.
Conjugation is a plan-wide operand-op configuration, with real/complex T at
compile time. Each job may have different dimensions and reused input ranges.

Planning:

- Compute products, offsets+length, byte limits and isize dimensions with
  checked arithmetic. Plans know required buffer lengths, not buffer pointers.
- Sort nonempty D intervals by offset once, reject any overlap in a linear
  scan, and retain sorted execution order/job IDs. Empty ranges do not overlap.
  Keep empty jobs separately for bounds validation and successful exactly-once
  accounting; give each an independent empty mutable slice, never move the
  nonempty split cursor backwards. Empty offsets must still be <= buffer len.
  Output holes are permitted for initialized execution and are unchanged.
- Cache no hidden global state; a host holds one GroupedPlan and lends its
  resources for repeated calls. Repeated input offsets with different shapes
  need not be forbidden here: each declared range is independently validated;
  tensor4all's stronger shared-shape policy may stay in its boundary.
- Build tprims plans once, no planning in execution. Optional plan deduplication
  is not needed for correctness and is not part of the first implementation.

Execution:

1. Validate complete A/B/D slice bounds for every job before any write.
   No-op/empty jobs get the same metadata checks.
2. Choose GroupedRoute::Empty (zero jobs only), Serial, or Outer { lanes } from job count,
   total prepared work and exec.budget(), using existing tprims WidthPolicy
   rather than adding thresholds. Clamp lanes to number of jobs. Children
   always run serially with the outer workspace; there is no inner fan-out or
   SPMD child and no resource-dependent failure after a different job writes.
3. Split the D buffer safely with split_at_mut in prepared offset order,
   preserving holes and unique mutable slices. Build borrow-only job items
   before dispatch; inputs are shared immutable slices.
4. Use one safe slice partition scheduler: lane-owned mutable item slices
   protected by one uncontended lock per lane to adapt Exec's Fn+Sync callback,
   not one lock per numerical job. Keep each original job ID with its item and
   wrap a lower-layer error in Error::Contract { step:original_id, source }.
   Serial execution bypasses scheduler locks. Before dispatch, every child's
   bounds, C mode and serial resource capability is validated; no planning,
   numerical temporary, per-job view allocation or provider probing occurs.
   Children execute the prepared slice calls with SerialWithWorkspace from
   the outer Exec provider (or bare serial only if the caller chose no reuse).
   contract_batched is not reused because its BatchItem API requires views,
   builds reads and per-item Mutex arrays, and loses original failing job IDs. Reuse Exec's partition
   primitive and existing WidthPolicy; do not add a second numerical kernel.
5. Return GroupedRoute and preserve original job identity in failures.
   Exactly one execution per job on success; order between disjoint jobs is
   unspecified. Runtime failure need not roll back already written jobs.

Warm execution must never construct an ArenaProvider per item/call. Bare
Exec::Serial remains explicitly call-local and is not the reuse idiom. The
normal caller retains ArenaProvider for serial or Pool for parallel and lends
it through Exec. Metadata item construction may allocate; document/count it,
then remove avoidable per-job metadata allocation with slice entries rather
than claiming an allocation-free boundary without a test.

Grouped execution in this PR accepts initialized output only. The requested
fresh MaybeUninit contract is supplied by BinaryPlan; no grouped uninit contract
is required by issue #4 or the existing downstream grouped caller. Do not add
an aggregate conversion or pretend output holes are initialized.

## Verification and evidence

Binary reference matrix: f32/f64/Complex32/Complex64, A/B/C/D conjugation,
alpha/beta (0,1,negative,complex,NaN), all C modes, empty output, zero K,
negative strides, offset bounds, mode mismatch, output injectivity, repeat
with distinct buffers and arenas. Test failure sentinels and input preservation.
Uninit tests verify full initialized values after success, rejection of padding
and Output/nonzero beta, zero-scale NaN behavior, and typed storage on failure.
Safe external callers cannot make an initialized slice before successful
lower-layer completion; document ownership lifetimes with compile-fail checks
where useful.

Grouped reference: heterogeneous sizes, shared inputs, unsorted outputs,
gaps, zero jobs/dimensions/K, overlap, overflowing arithmetic, late bad job
(bounds/C metadata) leaves all outputs unchanged. Serial and bounded parallel
results match per-job reference; job count instrumentation proves exactly-once,
PoolStats proves barrier-free outer dispatch and no SPMD children, worker entry
runs without serial retry, budgets 1/2/4 never exceed host resources. Warm serial
with retained arena has no per-call arena construction; distinguish numerical
scratch from metadata allocations using an allocation counter.

Local gate is the repository-required fmt/clippy/all-targets/workspace debug
and release tests, plus Rustdoc -D warnings and cargo tree -p cpueinsum
(no BLAS/C toolchain dependency). Numerical tests use the existing naive
reference. Any tprims prerequisite gets its own required gate and independent
review before publication. Final integrated diff gets gpt-6.1-sol post-review.
No speedup claim; if grouped scheduling changes algorithmic performance, add
representative 1T and separately 4T measurements with effective widths recorded,
not a machine-default overhead baseline. No extra CI/framework is added merely
because cpueinsum currently lacks hosted CI.

## Decisions for pre-review to falsify

- Is the proposed owner-level safe uninit slice proof sufficient for every
  strategy, especially faer's overwrite path and conjugated D?
- Is report granularity sufficient without exposing/faer/elementwise internal
  active widths? Do not conflate budget with active width.
- Does the single safe slice scheduler retain job IDs, safely split outputs
  including empty jobs, and avoid per-job view/arena construction?
- Is initialized-only grouped output faithful to the actual downstream contract,
  while BinaryPlan supplies the separate fresh MaybeUninit entry?
- Remove duplicate descriptors/helpers, capability flags without current
  callers, and unsupported promises. Prefer the smallest coherent boundary.
