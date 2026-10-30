# Project guidance

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here.

cpueinsum is a CPU einsum over integer labels with a caller-supplied
contraction order, built on tcontract (tensor4all/tcontract-rs). The plan
is tensor4all/tcontract-rs#62.

- Out of scope by design: contraction-order search, string notation, GPUs.
  Do not add them; they belong to the caller.
- BLAS lives only in `crates/cpueinsum-blas`: the overwrite-only `StepBackend`
  and concrete prepared binary/grouped adapters depend on `cblas-sys` and
  link no provider (tests link one through dev-dependencies:
  Accelerate on macOS, OpenBLAS elsewhere). `cpueinsum` itself never depends
  on a BLAS crate and must build without a C toolchain; check with
  `cargo tree -p cpueinsum`. The plan is tensor4all/cpueinsum-rs#1.
- Binary contraction kernels and strategy selection belong in tcontract.
  Fix planner behavior there (for example tensor4all/tcontract-rs#63), not by
  working around it here.
- Every numerical change is checked against the naive reference in
  `crates/cpueinsum/tests/` (shared by `crates/cpueinsum-blas/tests/`). Performance claims need a recorded measurement
  (CPU, dtype, shapes, threads, timed boundary); benchmark once the whole path
  is implemented rather than per part.
- No `unsafe` in `cpueinsum`. Intermediates and backend work space are
  disjoint slices of one scratch buffer obtained with `split_at_mut`.
  `cpueinsum-blas` has `unsafe` only for CBLAS calls and exact-full-write
  `MaybeUninit` publication, each with a `SAFETY` comment. Pointers are checked
  against operand slices at execution and dimensions at planning; fresh output
  is never represented by initialized references before completion.
- tcontract, tcontract-exec, threadteam, strided-basic and strided-view are
  crates.io version requirements (0.9.0, 0.9.0, 0.2.0, 0.9.0, 0.7.0). The private
  `tcontract-capi` is a git dependency on the tcontract-rs commit that released
  tcontract 0.9.0, and the `[patch."https://github.com/tensor4all/tcontract-rs.git"]`
  block points that repository's own crates at the registry releases, so each crate
  exists once in the graph (check with `cargo tree -d`). Bump the tcontract and
  strided requirements, and the capi rev, together.
- Publishable crates are released from `main` in dependency order, with a tag
  per release. A crate with a git dependency must stay `publish = false`.

## Build

`.github/workflows/ci.yml` runs all of the following on every push and pull
request. Run them locally before every push.

On Linux the `cpueinsum-blas` tests link the system OpenBLAS, which
`openblas-src`'s `system` feature requires to be installed (`libopenblas-dev`); the
workflow installs it, and macOS links Accelerate instead. `cpueinsum` alone needs
no C toolchain, so `cargo test -p cpueinsum` runs anywhere.

The declared `rust-version` is 1.89 (tcontract's MSRV, whose packed driver uses
AVX-512 intrinsics); the workflow's `msrv` job builds the workspace on 1.89.0 so the
claim cannot drift silently.

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --release --workspace
```
