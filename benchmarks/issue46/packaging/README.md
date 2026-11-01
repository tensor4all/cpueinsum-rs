# PR #59 packaging correction

Only diagnostic artifact placement and its path references changed. No new
measurement, result change, production scheduler change, or dependency change.

## Package inventory (real Cargo output)

Both lists were captured with exit status 0 using:

```bash
CARGO_BUILD_JOBS=8 nice -n 5 cargo package -p cpueinsum --list --allow-dirty
```

| Check | Before | After |
|---|---:|---:|
| `wc -l` | 71 | 29 |
| `grep -c issue46` | 43 | 1 |

The remaining match is **only `benches/issue46_grouped.rs`**, the intentionally
retained benchmark target source. No result data, raw JSONL, helper scripts,
protocol, patch or report is in the package. The 42 other issue46 files were
moved outside the crate. Lists: [before](package-before.list),
[after](package-after.list). Stderr: [before](package-before.stderr),
[after](package-after.stderr).

## Bench and local gates

- `nice -n 5 cargo bench -p cpueinsum --no-run -j8`: exit 0;
  [log](bench-no-run.log) contains the `issue46_grouped` executable.
- fmt check, all-target workspace clippy (-D warnings), debug and release
  workspace tests/doctests: all exit 0, sequential, nice=5 and Cargo build -j8;
  [commands and output](local-gates.log).
- Relocated build/record repository roots and bench source paths resolve;
  recorder self-check passes; all local Markdown links resolve; all 36 original
  result/evidence files are byte-identical to the pre-relocation commit.
  [Checks](path-checks.log).

Historical paths and hashes in the original evidence are intentionally retained
as recorded, not rewritten to suggest the measurement ran from its new location.
The actual benchmark source changed only its introductory path comment.
