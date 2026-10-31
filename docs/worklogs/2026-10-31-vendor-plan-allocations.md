# Vendor plan construction allocations (#52)

## Decisions

Keep prepared plans immutable and retain the existing eager native alternative.
An accepted vendor plan still routes zero-alpha calls and forced lane strategies
to native execution. Removing that alternative or deferring its construction
would change validation and execution-allocation timing. This change does not
add lazy planning, mutable shared scratch, a new binding API or a rank cap.

Allocation ownership along `GroupedPlan::new_with_ops` is:

- Native preparation allocates one temporary `JobMeta` vector for checked ranges,
  overlap validation and sorting, and one retained `PreparedJob<T>` vector for
  child plans. These are group-sized, not separate allocations for each job.
- The BLAS adapter allocates one retained vendor-job vector for the whole group.
  Each entry stores its checked offsets and immutable compiled layout.
- Previously `layout::choose` allocated two candidate orders, their vector
  container and the selected-order clone for each of M/N/K: 12 allocations per
  ordinary GEMM job. Empty/single-axis groups now borrow static orders. Multiple
  multi-axis groups share disjoint slices of one temporary order vector, reused
  across the two candidate-selection passes. The temporary is dropped after
  compilation; retaining it in the plan would serve no execution-time purpose.
- Batched stride vectors and packing dimensions/strides/compiled copy plans are
  retained metadata when needed. They are not rebuilt during execution. The
  representative non-BLAS-readable input needs four metadata allocations.
- Execution already accepts caller-owned packing storage sized by `work_len()`;
  one buffer is reused across jobs and subsequent calls. Concurrent calls share
  immutable plans, but own distinct output, packing storage and executors.

## Verification conclusions and constraints

The serialized global-allocator test counts Rust allocation/reallocation calls,
not CBLAS-provider mallocs. Using identical fixtures with baseline `124aa12`'s
layout selector and the changed selector gives:

| Constructor fixture | Before | After |
|---|---:|---:|
| Direct binary | 12 | 0 |
| Packed binary | 16 | 4 |
| Binary with multi-axis M/N groups | 12 | 1 |
| Empty group | 0 | 0 |
| 1 ordinary grouped GEMM | 15 | 3 |
| 5 ordinary grouped GEMMs | 63 | 3 |
| 32 ordinary grouped GEMMs | 387 | 3 |

For these nonempty groups the baseline is `12 * jobs + 3`; afterwards it is three
allocations per group and zero per-job layout allocations. Two of the three are
native preparation, one is the vendor-job vector. The regression gate rejects
the original selector, caps native construction at two allocations, and permits
only the single vendor vector beyond native construction. Retained-plan direct,
packed and grouped vendor execution continues to require zero Rust allocations.
These bounds describe the tested geometries, not arbitrary-rank packing; the
upstream stable stride sort can allocate for sufficiently large groups.

The selector oracle checks 216 cases covering candidate priority, stable ties,
singleton extents and zero/negative strides. The additional reuse tests compare
f64/c64 grouped results with separate native single-einsum plans and the shared
naive reference across rectangular, dot/GEMV/vector-matrix/outer-product
geometries, zero reduction and empty outputs. They assert Send + Sync for binary
and grouped plans, share each grouped plan across four callers for three repeated
executions, preserve output gaps, and exercise complex-conjugate packing with
nonempty caller workspace as well as whole-group native fallback.

No numerical kernel, public API, route policy or dependency changes. Native-plan
bytes are not eliminated. A reusable/bindable API or separating native validation
from planning remains outside this compatibility-preserving change. No timing,
throughput, downstream speedup or GPU result is claimed.
