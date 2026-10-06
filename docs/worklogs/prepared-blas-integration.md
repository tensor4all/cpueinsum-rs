# Prepared BLAS integration

## Decision and reason

Tenferro-rs#2004 needs prepared accumulation and fresh binary output through
its compiled BLAS adapter. The merged cpueinsum prepared APIs are native-only;
its BLAS StepBackend is intentionally overwrite-only. Extending that unrelated
N-ary contract or restoring host GEMM/post-axpby code would put execution in the
wrong owner. Concrete cpueinsum-blas BinaryPlan/GroupedPlan adapters preserve
the existing StepBackend and keep core cpueinsum independent of BLAS.

## Boundaries

Plans own metadata, not pools, operands, packing buffers or leases. Vendor
packing borrows explicit caller storage. Compatible geometry writes D directly
with alpha/beta and correct complex output conjugation. Unsupported geometry/C
modes and scalar-degenerate execution select already prepared native routes
before writes, never decline/replan/retry after mutation. `no_materialize`
selects native when the vendor geometry requires a whole-input copy; direct
vendor execution without copies remains eligible.

Fresh publication requires exact injective physical coverage and no old-output
reads. Grouped output remains initialized. Whole-group fallback avoids a mixed
vendor/native framework; every vendor job and packing view is preflighted
before the first output write. Original IDs survive vendor failures. Shared
private vendor metadata avoids retaining unused per-child native alternatives.
Checked packing geometry and terminal odometer wrapping prevent overflow from
being confused with active valid offsets.

## Verification conclusions and limits

Debug and release workspace tests, formatting, all-target clippy with warnings
as errors, executable rustdoc, warning-free docs and Rust 1.89 all-target checks
passed. New reference checks cover four dtypes, complex scales/conjugations,
initialized and fresh Separate C native
alternatives, reversed/gapped/broadcast inputs, beta-zero NaN handling, rejected
fresh holes/padding/output reads, grouped late bounds and workspace failure,
output overlap/byte overflow and whole-group degenerate native execution.
Test-only recording at the actual CBLAS wrappers verifies coordinator identity
and zero selected-pool entries for vendor binary/grouped calls with a supplied
four-worker pool. The core normal/build dependency tree still has no BLAS/C build
requirement. Existing BLAS/N-ary reference tests remain green. Removing output
scalar conjugation made the four-dtype reference test fail; restoring it passed.

The whole-input-copy refusal was reproduced for reversed binary input and
conjugated grouped input, then corrected in shared vendor preparation. Both
regressions pass, including retained direct vendor eligibility. A reduced
output-label test separately protects against publishing untouched diagonal
backing slots even when the original axes span the entire allocation.

These checks establish neither integrated tenferro performance nor universal
allocation freedom, a provider-internal thread ceiling, Miri or sanitizer
coverage. Packing view construction can allocate metadata. Independent
post-review completed and its contract/documentation corrections were applied;
final local gates passed and coordinated PR delivery remains pending. No merge
or publication is authorized.
