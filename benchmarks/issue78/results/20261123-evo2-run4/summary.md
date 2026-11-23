### Batched vs per-job (cand-518aed8, one paired process per width)

| jobs | m=k=n | threads | per-job median (IQR) ns | batched median (IQR) ns | per-job/batched | verdict | allocs/call (per-job, batched) | route (per-job / batched) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 64 | 2 | 1 | 6085 (17) | 1907 (15) | 3.19 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 1 | 6542 (29) | 2626 (10) | 2.49 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 1 | 7936 (24) | 3801 (18) | 2.09 | batched wins | 0, 0 | Serial / Serial |
| 64 | 16 | 1 | 20226 (105) | 15984 (38) | 1.27 | batched wins | 0, 0 | Serial / Serial |
| 64 | 32 | 1 | 68031 (304) | 62084 (289) | 1.10 | batched wins | 0, 0 | Serial / Serial |
| 256 | 2 | 1 | 24417 (192) | 7393 (51) | 3.30 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 1 | 25994 (34) | 10088 (19) | 2.58 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 1 | 31609 (48) | 14520 (28) | 2.18 | batched wins | 0, 0 | Serial / Serial |
| 256 | 16 | 1 | 85017 (340) | 67700 (635) | 1.26 | batched wins | 0, 0 | Serial / Serial |
| 256 | 32 | 1 | 274614 (1978) | 249173 (731) | 1.10 | batched wins | 0, 0 | Serial / Serial |
| 64 | 2 | 8 | 6130 (20) | 1937 (13) | 3.16 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 8 | 6642 (109) | 2672 (10) | 2.49 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 8 | 6109 (35) | 2108 (19) | 2.90 | batched wins | 1, 0 | Outer { lanes: 3 } / Inner { threads: 3 } |
| 64 | 16 | 8 | 6972 (36) | 3336 (24) | 2.09 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 64 | 32 | 8 | 11194 (320) | 64258 (287) | 0.17 | batched loses | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 2 | 8 | 24284 (23) | 7346 (34) | 3.31 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 8 | 26050 (82) | 10055 (61) | 2.59 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 8 | 23277 (129) | 3282 (56) | 7.09 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 16 | 8 | 22540 (189) | 9417 (150) | 2.39 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 32 | 8 | 40070 (1274) | 257676 (1190) | 0.16 | batched loses | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |

### Per-job non-regression (base-ebc74c3 vs cand-518aed8, per-job arm, 3 process pairs)

| jobs | m=k=n | threads | base process medians ns | cand process medians ns | cand/base (median of medians) | verdict |
|---:|---:|---:|---|---|---:|---|
| 64 | 2 | 1 | 6026, 6071, 6042 | 6142, 6116, 6089 | 1.012 | no regression |
| 64 | 4 | 1 | 6434, 6441, 6541 | 6589, 6550, 6542 | 1.017 | no regression |
| 64 | 8 | 1 | 7756, 7825, 7806 | 8031, 7956, 7967 | 1.021 | no regression |
| 64 | 16 | 1 | 20230, 20236, 20233 | 20194, 20237, 20149 | 0.998 | no regression |
| 64 | 32 | 1 | 67038, 67226, 68380 | 68489, 69846, 68054 | 1.019 | no regression |
| 256 | 2 | 1 | 23948, 24597, 24034 | 24470, 24413, 24366 | 1.016 | no regression |
| 256 | 4 | 1 | 25723, 25790, 25693 | 26087, 25939, 25904 | 1.008 | no regression |
| 256 | 8 | 1 | 31716, 32087, 31467 | 31088, 31099, 31050 | 0.980 | no regression |
| 256 | 16 | 1 | 84038, 84839, 84081 | 83361, 84588, 84391 | 1.004 | no regression |
| 256 | 32 | 1 | 271909, 271992, 272707 | 280816, 284748, 275529 | 1.032 | REGRESSION |
| 64 | 2 | 8 | 5999, 5991, 6018 | 6107, 6060, 6061 | 1.010 | no regression |
| 64 | 4 | 8 | 6462, 6441, 6445 | 6518, 6513, 6513 | 1.011 | no regression |
| 64 | 8 | 8 | 6322, 5967, 6226 | 6262, 6346, 6144 | 1.006 | no regression |
| 64 | 16 | 8 | 7067, 7184, 7230 | 7026, 7044, 7078 | 0.981 | no regression |
| 64 | 32 | 8 | 11043, 10878, 10985 | 10932, 10818, 10981 | 0.995 | no regression |
| 256 | 2 | 8 | 23864, 23833, 23867 | 24336, 24221, 24325 | 1.019 | no regression |
| 256 | 4 | 8 | 25614, 25637, 25472 | 25940, 25919, 25934 | 1.012 | no regression |
| 256 | 8 | 8 | 23162, 23723, 23552 | 23277, 23180, 23151 | 0.984 | no regression |
| 256 | 16 | 8 | 22790, 23177, 23071 | 22680, 22770, 22735 | 0.985 | no regression |
| 256 | 32 | 8 | 39008, 40512, 39080 | 38649, 39184, 38722 | 0.991 | no regression |
