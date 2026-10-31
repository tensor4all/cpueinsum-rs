# Group-local child geometry sharing (#51)

## Decisions

Keep the agreed indexed representation: each private `PreparedJob` stores its
original metadata and a checked `u32` child index; the immutable grouped plan
owns a single vector of distinct `Plan<T>` children. The standard-library
HashMap uses raw `(rows, contracted, cols)` keys, with expected O(J) total
lookups in addition to the existing output-range sort. Operations, scalar type
and effective child configuration are uniform within the group.

The issue originally said to deduplicate after `.max(1)`. The maintainer
explicitly approved using raw dimensions instead: `(2,0,2)` and `(2,1,2)` must
not share a child because empty reduction and a one-element reduction have
different numerical behavior. Only strides retain the existing `.max(1)`;
`Problem::from_labels` still receives the original dimensions.

Validate every descriptor and output overlap first, unchanged. Enumerate distinct
keys in sorted job order before allocating the child vector, so its requested
capacity is exactly G rather than J. Build `Problem` and `Plan` only at each key's
first occurrence. The temporary geometry map grows with G and is discarded after
construction. There is no cross-call cache, per-job reference-counted allocation,
new dependency, public API, or vendor/native-plan split.

Three aggregate length comparisons admit full-size runtime buffers without a
per-job bounds scan. Short buffers still run the original lowest-original-ID
scan before any writes. Runtime route admission still visits every job with the
current alpha/output contract. Serial and parallel execution borrow the indexed
child; native workspace requirements fold the distinct children.

## Verification conclusions and constraints

Test-only thread-local counters check 200 jobs construct exactly three children,
with child length/capacity both three; empty, singleton and 17-all-distinct groups
are covered too. Another counter proves full-size/excess buffers bypass the
bounds scan and short lhs/rhs/output buffers scan every job and preserve the
lowest original error ID without writes. Checked u32 conversion is tested at
zero, u32::MAX and its first unrepresentable successor on 64-bit targets.

The zero-versus-one regression verifies distinct indices and correct output in
both output orders. A 200-job mixed-geometry regression compares grouped output
bits against independently prepared single-job plans for f32/f64/c32/c64, all
eight conjugation combinations, zero/nonzero beta and zero alpha. Existing
empty-K, fresh-output, forced-route and lowest-ID grouped coverage remains.

The fixed-single-geometry allocator fixture now permits four native constructor
allocations: temporary job metadata and geometry map, retained compact jobs and
unique children. It still enforces job-count-independent allocation calls and
only one extra vendor job vector. This bound is not universal as G grows.
No numerical kernels, scheduling policy or vendor production source changed.
No timing, throughput, downstream speedup or GPU claim is made.
