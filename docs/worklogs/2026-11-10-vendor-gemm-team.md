# The vendor GEMM runs inside the caller's team (#73)

## Decisions

The rule is ported from `tenferro-rs#2562` (issue `tenferro-rs#2553`): threadteam
0.5 pins the thread that creates a `Team` to `{S[0]}` for the team's life, and a
vendor library creates its own thread team from the calling thread, so a CBLAS
call from that leader confines the whole vendor team to one CPU.

`crates/cpueinsum-blas/src/lib.rs` gains one `pub(crate) fn vendor<R>(call)`,
which runs the call inside `Exec::serial().team(|mut team| team.external(..))`.
That team has one participant and reserves nothing; its only effect is
`TeamExec::external`, which widens the calling thread to `S` for the call and
sets `{S[0]}` back afterwards, on return and on unwind. On a thread threadteam
placed nobody it is a plain call. All three GEMM loops go through it: the N-ary
step (`run`), the prepared binary step (`VendorPlan::run_vendor`) and the
grouped jobs, which reach it through the same `run_vendor`.

The vendor thread count stays the vendor's own setting (`MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, ...). This decides which CPUs those threads may run on,
not how many the library creates.

The pack and the output unpack take the caller's `Exec` (`exec.token()`), not
`Exec::serial()`. For a serial token that is the serial kernel the old call ran;
for a team it splits the copy over the caller's slots. Both are pure copies with
no arithmetic, so the output does not change.

The test hook in `src/ffi/tests.rs` records the calling thread's CPU count at
every GEMM. `src/tests.rs` drives one call of each of the three loops from a
four-member team's leader — the N-ary backend step, a prepared `BinaryPlan` and a
two-job `GroupedPlan` — and asserts that every GEMM each of them issued saw more
than one CPU and that the leader is on `{S[0]}` again afterwards. Deleting
`crate::vendor` from `run` fails it with `backend step: a GEMM saw [1] CPUs`, and
deleting it from `VendorPlan::run_vendor`, which the prepared and the grouped
adapters both reach, fails it with `prepared binary: a GEMM saw [1, 1] CPUs`.

The test skips (with a `skipped:` line) where the process has fewer than four
CPUs, because threadteam places nothing there; a skipped run is not evidence.

## Measurement

Evo2 (`EVO-X2`, AMD Ryzen AI Max+ 395, 32 CPUs in 2 L3 instances; CPU 0's
instance is `0-7,16-23`), Linux 7.0.0, MKL 2026.2 (`libmkl_rt.so` + `libiomp5.so`
preloaded, provider confirmed from `/proc/self/maps`). The measured cell is one
L3 domain — CPUs `0-3` for 4T, CPU `0` for 1T — and the observer (the driving
shell) is pinned to `8-15`, the other domain. Load average 1.3–2.0.

Workload: one `cpueinsum_blas::Blas` plan step, `D[i,l] = A[i,j] B[j,l]` with
`n = 2048` column-major `f64` (17.2 GFLOP per call), `alpha = 1`, `beta = 0`, 5
calls per process, median reported. `1T` runs the plan from `Exec::serial()` on
one CPU; `4T` runs it from a `Team::exec(4)` token of a
`Pool::builder().threads(4)` team on four CPUs, so the calling thread is the team
leader. The mask inside the CBLAS call is read by a preloaded shim
(`libaffprobe.so`: `sched_getaffinity(0)` in `cblas_dgemm`, then
`dlsym(RTLD_NEXT, "cblas_dgemm")`); `leader before/after` is the leader's own
mask around the call, still inside the live team; `cpu/wall` is
`/proc/self/stat` utime+stime over wall time for the timed region.

```text
export LD_PRELOAD="/tmp/libaffprobe.so /opt/intel/oneapi/compiler/latest/lib/libiomp5.so \
                   /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so"
taskset -c 0   target/release/examples/vendor_team_probe serial 2048 5   # 1T
taskset -c 0-3 target/release/examples/vendor_team_probe team   2048 5   # 4T
```

`before` is `origin/main` (`8a12232`), `after` is this branch; the same probe
source is built against both.

| threads | `MKL_NUM_THREADS` | tree | CPUs inside `cblas_dgemm` | leader before/after | cpu/wall | median ms |
|---|---|---|---:|---|---:|---:|
| 1T | unset | before | 1 | 1 / 1 | 0.99 | 395.9 |
| 1T | unset | after | 1 | 1 / 1 | 0.99 | 403.6 |
| 1T | 4 | before | 1 | 1 / 1 | 1.00 | 409.2 |
| 1T | 4 | after | 1 | 1 / 1 | 0.98 | 417.0 |
| 4T | unset | before | **1** | 1 / 1 | 1.02 | 401.3 |
| 4T | unset | after | **4** | 1 / 1 | 3.94 | 122.7 |
| 4T | 4 | before | **1** | 1 / 1 | 1.00 | 419.0 |
| 4T | 4 | after | **4** | 1 / 1 | 3.99 | 123.4 |

Every one of the eight processes printed the same output checksum
(`995f58ecd4222325` over the 32 MiB output, including the 1-thread arms before
and after the change).

The two failure shapes are both there: with `MKL_NUM_THREADS` unset the confined
leader made MKL see one CPU and run one thread (401.3 ms, 3.3x slower than the
122.7 ms the same call takes on four CPUs); with it set to 4 the four MKL threads
shared one CPU and did not even reach the single-thread case (419.0 ms).

The 1T rows are unchanged. The sequential readings differ by up to 2% (395.9 and
409.2 before against 403.6 and 417.0 after), which arm order, not the change,
could explain, so the 1T arm was re-measured interleaved (`after before before
after`, twice, `MKL_NUM_THREADS` unset): `after` 389.5, 403.3, 407.4, 393.0
against `before` 403.3, 404.1, 401.0, 399.2 — no separation. Nothing on the 1T
path changes anyway: a serial token's `team` of one has no placement to widen,
and the packs are the same serial copy.

## What is left as it was

A GEMM issued from a team *member* — a forked child token passed to
`execute_into` — keeps that member's single CPU. threadteam's member half is
`Exec::external`, which for a pool worker restores its own saved mask (the whole
`S`) and runs the call inline: that would be the right widening. It cannot be
used here for the same region that serves the leader, for two reasons. It takes a
`Send` closure, and a vendor region captures raw operand pointers and a
`&mut [T]`. And it is gated on `budget > 1`, so the common member shape — a fork
child with a share of one slot — skips it entirely, while on any caller that is
not a pool worker (the leader of a root scope) it installs the call onto a pool
worker instead of widening the calling thread. One wrapper cannot serve both
halves without knowing which thread it is on, which threadteam does not expose.
The crate's own routing issues a backend step from the token it is handed and
dispatches no vendor call to a member, so this needs a caller that forks a step
itself; the leader half is what #2553 measured.
