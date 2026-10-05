# Project guidance

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here.

cpueinsum is a CPU einsum over integer labels with a caller-supplied
contraction order, built on tprims-contract (tensor4all/tprims-rs). The plan
is tensor4all/tprims-rs#62.

- Out of scope by design: contraction-order search, string notation, GPUs.
  Do not add them; they belong to the caller.
- BLAS lives only in `crates/cpueinsum-blas`, a `StepBackend` that depends on
  `cblas-sys` and links no provider (tests link one through dev-dependencies:
  Accelerate on macOS, OpenBLAS elsewhere). `cpueinsum` itself never depends
  on a BLAS crate and must build without a C toolchain; check with
  `cargo tree -p cpueinsum`. The plan is tensor4all/cpueinsum-rs#1.
- Binary contraction kernels and strategy selection belong in tprims-contract.
  Fix planner behavior there (for example tensor4all/tprims-rs#63), not by
  working around it here.
- Every numerical change is checked against the naive reference in
  `crates/cpueinsum/tests/` (shared by `crates/cpueinsum-blas/tests/`). Performance claims need a recorded measurement
  (CPU, dtype, shapes, threads, timed boundary); benchmark once the whole path
  is implemented rather than per part.
- No `unsafe` in `cpueinsum`. Intermediates and backend work space are
  disjoint slices of one scratch buffer obtained with `split_at_mut`.
  `cpueinsum-blas` has `unsafe` only for the CBLAS calls, each with a
  `SAFETY` comment; pointers are checked against the operand slices at
  execution and dimensions at planning.
- tprims crates and strided-view are git dependencies at pinned revs; bump
  them together with tprims-rs's own pin.
- Do not publish the crates (`publish = false` until tprims is published).

## Build

There is no CI for now. Run all of the following locally before every push.

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --release --workspace
```
