# cpueinsum-rs#80: work-weighted lane placement, GP/native on keyaki

**Different host from the #80 table.** That table was measured on an Intel Xeon
Platinum 9242 at up to 96 threads. These numbers come from **keyaki: AMD EPYC
7702P, 64 physical cores, SMT off, one NUMA node, L3 per 4-core CCX**, at 16, 32
and 64 threads. They are the closest measurement available to us, not a
reproduction of the #80 table.

## Arms

| arm | what | revision |
|---|---|---|
| `native` | symtn-dense's own grouped GEMM (`DefaultDenseExecutor::matmul_batch_axpby_into` on `CpuEngine`, a `Pool` of T threads, `strided_batch_runs` partition), with symtn's own lane placement | symtn-rs `09a3ebb`; **tcontract 0.12.0** children (symtn's lock), not the 0.13 children the #80 table used |
| `base` (GP before) | `cpueinsum::GroupedPlan` under `Auto` | cpueinsum-rs `b11f3bd` (main) |
| `cand` (GP after) | the same, with this change | run 1: `b31a4e4`; run 2: `48c7edb` (PR head) |

Both GP arms use tcontract at `ca44b47` (0.13.0) through cpueinsum's own `Cargo.lock`. The drivers were
built as untracked examples inside each repository, so every arm resolved through its own lock file.

## Protocol (recorded before the runs)

- Timed boundary, as in #80: one whole grouped call **including the `GroupedPlan`
  construction** from cached children (`prepare_jobs` + `from_prepared_with` with a
  `ChildKey` cache + `execute_into_accum`), so the placement search is paid every call.
  `native` times one `matmul_batch_axpby_into` call.
- Per cell: 1000 ms time-based priming per workload, then the median of 11 warm calls.
  Two rounds of (native, base, cand) per thread count, in that order; the table
  shows the mean of the two round medians and, for the ratio, the per-round range.
- Cores: T=16 on `taskset -c 16-31`, T=32 on `32-63`, T=64 on `0-63`. For T=16/32
  every Claude process on the host was pinned to `0-3` (the observer) for the
  run and restored after. **T=64 uses every core**: the agents were idle but not
  separated from the measured set.
- Checked cells: before timing, each GP arm runs the call once and compares
  its output with the same plan on `Exec::serial()` (`==` on values). All 480 GP cells
  in both runs: equal.
- Workloads (`workloads.txt`): `gm-dom`, `gm-mix` and `c u1f` built from the
  #80 description (`c u1f` = 768³, 2 x 512³, 2 x 256³, the shares 0.60/0.18/0.18/0.02/0.02;
  run as `into` with beta 0 here, where #80 timed it as `fresh#0`), and the
  heff calls taken from the symtn-gemm-trace corpus at `f8206b55`,
  `itebd-heff-c853d94f-t96-x86_64.jsonl.gz` (sha256 `57ef0a59…eceb`, the
  Xeon full-width capture): u1 1024/256/64 `heff.L_theta into#0`, u1 1024
  `heff.W1 into#0`, su2 1024 (935 states) `heff.L_theta into#4` and `heff.W1 into#1`,
  su2 256 `heff.L_theta into#3`, with their own alpha and beta.
- Build: `cargo build --release` (each workspace's release profile), rustc 1.100.0.

`results/20261125-keyaki-run2/table.md` is the PR head. Run 1 measured `b31a4e4`,
before the item-time memo and the serial-children shortcut. Its search cost
showed up as +2 % on gm-dom f64 at 64T, the reason for `48c7edb`.

## Reproduce

```sh
BENCH_DIR=<dir with workloads.txt and the three binaries> drivers/run.sh <T> <cpu list>
python3 drivers/table.py <dir with raw-t*.tsv>
```

`drivers/diag80.rs` is the diagnostic that measured the search cost and the f64 child widths
(not part of the timed protocol).
