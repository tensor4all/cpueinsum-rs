<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to cpueinsum-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `cpueinsum-rs`:

## Local facts

- **MSRV**: `rust-version = "1.89"` in the workspace `Cargo.toml`, inherited by every crate
  (`rust-version.workspace = true`). The `msrv` job builds `cargo build --workspace --all-targets` on `1.89.0`; every
  other Rust job uses `stable`.
- **Feature combinations CI builds**: one — the default features. No crate in the workspace declares a feature, so
  the lanes differ by profile and operating system, not by feature set: `cargo test --workspace` (dev) and
  `cargo test --release --workspace` on Linux. `cpueinsum-blas`'s tests link a provider: the system OpenBLAS on Linux
  (installed by the workflow, because `openblas-src`'s `system` feature needs `libopenblas-dev`) and Accelerate on
  macOS. `cpueinsum` itself needs no C toolchain.
- **Platforms claimed**: `ubuntu-latest` and `macos-latest` for `clippy` and `test`; `fmt`, `deny`, `test-release`,
  `overhead` and `msrv` on `ubuntu-latest` only.
- **Local gate**:

  ```sh
  sudo apt-get install -y libopenblas-dev   # Linux only; macOS links Accelerate
  cargo test --workspace
  ```

  The install is the step people forget: `cpueinsum-blas`'s dev-dependencies link the system OpenBLAS through
  `openblas-src`'s `system` feature, and without it the workspace test fails with
  `unable to find library -lopenblas`. `cargo test -p cpueinsum` alone needs no C toolchain.

## Anything here that departs from the shared rules

Nothing departs from the rules in the shared page. This repository adds a `deny` lane (only `threadteam`, and the
vendor wrappers listed in `deny.toml`, may depend on `rayon`), a release-profile test lane, and the ratio-gated
`overhead_smoke` benchmarks.
