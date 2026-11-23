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

Raw CSVs, binaries' SHA-256 and the load log of every run are in
`benchmarks/issue78/results/`; `summarize.py` there produces each run's
`summary.md`. All runs: 1-minute load at most 2.9, so the noise gate held.

### Run 1 (candidate `ae1bc79`): per-job regression, candidate revised

The predeclared gate flagged a per-job regression of 3-8 % on 2³-8³ jobs at
both widths: the item loops built a generic item and branched on the item
list per job. The batched route also lost at 8 threads for 16³ and 32³,
because under `Auto` every item was serial and a run is one item. Both fixed
in `d7e0e68` (one dispatch between job list and item list; run items get the
width the lanes leave unused under the child's own `Auto`).

### Run 2 (candidate `d7e0e68`): per-job regression remains, candidate revised

Batched now won at 8 threads up to 16³, but the per-job gate still flagged
3-6 % on 2³-8³. Diagnostic (`results/diag-layout`, 1 thread, three
alternating processes per binary): a build of `main` with only the `t4a-desc`
dependency added was -1.5 % to +0.8 % against `main` on 2³-8³, the `d7e0e68`
code +0.2 % to +4.0 %, so the cost was code, not the dependency. (The first
pass of this diagnostic printed to the terminal only; `rows.csv` and
`load.log` there are a rerun of the same binaries, checked against
`binaries.sha256`, with the output kept.) Fixed in
`1f35056`: the job source builds each child view straight from the job
metadata with literal zero origins as `main` does, and the default item policy
(a host-model read) is built only where needed.

### Run 3 (candidate `1f35056`): the final candidate

Per-job non-regression: 19 of 20 cases pass. On 2³-8³ the candidate/base
ratio of the median process medians ranges from 0.990 to 1.041; the largest,
64 × 8³ at 8 threads (+4.1 %), passes only because the process ranges overlap
(base 6066-6303 ns, candidate 6270-6465 ns). The same final code was -1.3 %
to +2.1 % on 2³-8³ at 1 thread in the diag-layout rerun. The gate flags **64 × 16³ at 8 threads, +10.2 %** (base process medians 7517,
7256, 7323 ns; candidate 8071, 7956, 8140 ns). The same case was +8.5 % in
run 1 and -1.8 % in run 2. A follow-up diagnostic, not the gated experiment
(`results/diag-64x16-t8`, 10 alternating process pairs of the run-3
binaries), gave base 7595-7905 ns and candidate 7780-8032 ns (medians about
7790 and 7890, +1.3 %). This 8-lane case runs on an unpinned pool
(`pin_participants(false)`), whose placement is exactly what threadteam-rs#88
will change; the run-3 verdict stands as recorded.

Batched vs per-job (run 3; ratio = per-job / batched median, > 1 means
batched is faster; allocation requests per call 0 for batched, 0 per-job
serial, 1 per-job with outer lanes):

| jobs | m=k=n | 1 thread | 8 threads |
|---:|---:|---:|---:|
| 64 | 2 | 3.23 wins | 3.19 wins |
| 64 | 4 | 2.54 wins | 2.57 wins |
| 64 | 8 | 2.08 wins | 3.00 wins |
| 64 | 16 | 1.26 wins | 2.17 wins |
| 64 | 32 | 1.07 wins | **0.18 loses** |
| 256 | 2 | 3.34 wins | 3.32 wins |
| 256 | 4 | 2.62 wins | 2.62 wins |
| 256 | 8 | 2.18 wins | 7.40 wins |
| 256 | 16 | 1.27 wins | 2.50 wins |
| 256 | 32 | 1.08 wins | **0.16 loses** |

One cell is less precise than the protocol describes: in run 3's 8-thread
paired process, the 64 × 16³ calibration call took over 2 ms, so both arms of
that cell ran 1 call per sample instead of about 260 (`calls_per_sample` in
`paired-t8.csv`; IQRs 760 and 120 ns). Run 2's cell, at 263 calls per sample,
gives the same verdict (2.11, batched wins). Every other cell has at least 7
calls per sample (7 only at 256 × 32³ on 1 thread, about 2 ms as designed).

The batched route wins everywhere on one thread and up to 16³ on eight; at
32³ on eight threads it loses 5-6×. There tcontract plans the batched problem
with its Faer strategy, which runs at serial speed (about 65 and 260 µs, the
same as one thread) although the child is granted 8 threads, while the
per-job route spreads 64 or 256 jobs over 8 outer lanes. Spreading that
batch is tcontract's planner decision (AGENTS.md: fix planner behaviour
there), so it is a follow-up, and the reason the route stays opt-in.

### issue46 rows (diagnostic, single-call wall time, `1f35056`)

`issue46_grouped <B|C> <1|8> Auto 250 1000 <per-job|runs>`, 1 thread on CPU 2,
8 threads on CPUs 0-7 (`results/20261123-evo2-issue46/rows.csv`):

| workload | threads | per-job | runs | per-job route | runs route |
|---|---:|---:|---:|---|---|
| B (16384 × 32³) | 1 | 40.7 ms | 35.6 ms | Serial | Serial |
| B (16384 × 32³) | 8 | 8.9 ms | 40.2 ms | 8 lanes | 1 lane × 8 |
| C (32³ ×8192 + one 1024×256×1024) | 1 | 25.3 ms | 22.6 ms | Serial | Serial |
| C | 8 | 10.0 ms | 18.4 ms | 8 lanes | 2 lanes × 4 |

Same picture as the 32³ rows above.

## Not checked

Other CPUs (only Zen 5 here), other thread counts than 1 and 8, complex
dtypes in timing (tests cover them for correctness), the BLAS backend
(`cpueinsum-blas` ignores the option), and anything after threadteam-rs#88.
