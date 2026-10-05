# Project guidance

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here.

cpueinsum is a CPU einsum over integer labels with a caller-supplied
contraction order, built on tprims-contract (tensor4all/tprims-rs). The plan
is tensor4all/tprims-rs#62.

- Out of scope by design: contraction-order search, string notation, BLAS,
  GPUs. Do not add them; they belong to the caller (tenferro keeps its BLAS
  path).
- Binary contraction kernels and strategy selection belong in tprims-contract.
  Fix planner behavior there (for example tensor4all/tprims-rs#63), not by
  working around it here.
- Every numerical change is checked against the naive reference in
  `crates/cpueinsum/tests/`. Performance claims need a recorded measurement
  (CPU, dtype, shapes, threads, timed boundary); benchmark once the whole path
  is implemented rather than per part.
- No `unsafe` in this crate. Intermediates are disjoint slices of one scratch
  buffer obtained with `split_at_mut`.
- tprims crates and strided-view are git dependencies at pinned revs; bump
  them together with tprims-rs's own pin.
- Do not publish the crate (`publish = false` until tprims is published).

## Build

There is no CI for now. Run all of the following locally before every push.

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --release --workspace
```
