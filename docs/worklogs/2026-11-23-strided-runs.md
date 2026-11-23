# 2026-11-23: batched strided runs in `GroupedPlan` (#78)

## Change

`GroupedOptions::with_strided_runs` (opt-in; the default is unchanged) makes
`GroupedPlan` take the strided-run partition of its job list from
`t4a_desc::gemm::GemmBatch::runs` (tensor4all/tcontract-rs#209) at prepare
time, and execute every batched run as one tcontract problem with a batch
label `h` over the run: A `[m, k, B]` strides `[1, lda, sA]`, B `[k, n, B]`
strides `[1, ldb, sB]`, D `[m, n, B]` strides `[1, ldc, sD]`, origin at the
run's first job. The run child goes through the same `Plan` slice call as a
single job, so there is no second execution path in cpueinsum.

A run is batched when it has at least two jobs, a non-empty output, and its
output span holds no other job's output (so it can own one disjoint output
slice without `unsafe`). Every other job keeps the per-job route; without a
batched run the item list is the job list itself.

## Protocol (recorded before the measured runs)

- Baseline: `main` at `ebc74c3`, per-job route only. The bench file is the
  candidate's with `build` replaced by `GroupedPlan::new` (pre-#78 API).
- Candidate: branch `feat/78-strided-runs` at `ae1bc79`.
- Source: `crates/cpueinsum/benches/issue78_strided_runs.rs`, `cargo bench`
  profile (`release`: opt-level 3, thin LTO, one codegen unit), rustc 1.100.0.
- Host: EVO-X2, AMD Ryzen AI MAX+ 395 (Zen 5, 16 cores / 32 threads, two
  CCDs with cores 0-7 and 8-15). 1 thread pinned with `taskset -c 2`; 8
  threads with `taskset -c 0-7` (one CCD, no SMT siblings). Pool
  `pin_participants(false)`, batch policy `Auto`.
- threadteam: crates.io 0.6.0 (Cargo.lock checksum `fbc7aeed…`, tag `v0.6.0`
  = `358f8ac`). tlinalg#119 moved Pack/Spread placement to threadteam
  (proposed threadteam-rs#88); that change moves every small-team number, so
  nothing below is a baseline beyond threadteam 0.6.0.
- Cases: `J ∈ {64, 256}` jobs × `m = k = n ∈ {2, 4, 8, 16, 32}`, f64, one
  compact run (offsets advance by the job sizes), `execute_into`, alpha 1.
- Statistic: one sample is the mean per-call time over enough calls to last
  about 2 ms; 21 samples per arm; arms interleaved within a process with the
  order alternating per repetition. Reported: median and IQR of the samples,
  and heap allocation requests per call (counting global allocator, 100
  calls).
- Batched vs per-job (candidate binary, both arms): *batched wins* when its
  q3 is below per-job's q1; *batched loses* when its q1 is above per-job's
  q3; otherwise *no difference*.
- Per-job non-regression: baseline and candidate binaries, per-job arm only,
  three alternating process pairs (base, cand, base, cand, base, cand) per
  thread count. *Regression* when the candidate's median of the three
  process medians exceeds the baseline's by more than 3 % and the candidate's
  smallest process median is above the baseline's largest.
- Host noise: the 1-minute load average is recorded before every process.
  Other agents share this host; if it exceeds 8 before any process, the whole
  paired experiment is INCONCLUSIVE and is rerun complete.
- One smoke run of the candidate (1 thread, 3 samples) preceded this protocol
  to check the harness; its numbers are not used.

## Results

Pending.
