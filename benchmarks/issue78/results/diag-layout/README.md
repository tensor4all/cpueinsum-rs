Diagnostic, not the gated experiment: per-job route, 1 thread on CPU 2, three
alternating processes per binary (`rows.csv`, `load.log`).

- `base`: `main` `ebc74c3`
- `vdep`: `ebc74c3` plus only the `t4a-desc` dependency and re-export
- `cand`: `d7e0e68`
- `v2`: the `1f35056` code (built before that commit, label `diag-v2`)
