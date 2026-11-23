### Batched vs per-job (candidate ae1bc79, one paired process per width)

| jobs | m=k=n | threads | per-job median (IQR) ns | batched median (IQR) ns | per-job/batched | verdict | allocs/call (per-job, batched) | route (per-job / batched) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 64 | 2 | 1 | 6295 (16) | 1900 (12) | 3.31 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 1 | 6693 (12) | 2637 (16) | 2.54 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 1 | 7997 (41) | 3780 (8) | 2.12 | batched wins | 0, 0 | Serial / Serial |
| 64 | 16 | 1 | 20280 (27) | 15939 (65) | 1.27 | batched wins | 0, 0 | Serial / Serial |
| 64 | 32 | 1 | 67269 (433) | 62178 (488) | 1.08 | batched wins | 0, 0 | Serial / Serial |
| 256 | 2 | 1 | 25191 (75) | 7356 (45) | 3.42 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 1 | 26482 (77) | 10060 (38) | 2.63 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 1 | 32124 (77) | 14584 (23) | 2.20 | batched wins | 0, 0 | Serial / Serial |
| 256 | 16 | 1 | 84102 (215) | 66501 (469) | 1.26 | batched wins | 0, 0 | Serial / Serial |
| 256 | 32 | 1 | 272787 (1734) | 247142 (1167) | 1.10 | batched wins | 0, 0 | Serial / Serial |
| 64 | 2 | 8 | 6338 (58) | 1942 (19) | 3.26 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 8 | 6812 (52) | 2680 (9) | 2.54 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 8 | 6290 (135) | 3919 (22) | 1.61 | batched wins | 1, 0 | Outer { lanes: 3 } / Serial |
| 64 | 16 | 8 | 7969 (225) | 16054 (66) | 0.50 | batched loses | 1, 0 | Outer { lanes: 8 } / Serial |
| 64 | 32 | 8 | 11386 (570) | 62184 (486) | 0.18 | batched loses | 1, 0 | Outer { lanes: 8 } / Serial |
| 256 | 2 | 8 | 25329 (169) | 7316 (88) | 3.46 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 8 | 26772 (189) | 10080 (34) | 2.66 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 8 | 23439 (264) | 14412 (70) | 1.63 | batched wins | 1, 0 | Outer { lanes: 8 } / Serial |
| 256 | 16 | 8 | 24476 (277) | 66631 (235) | 0.37 | batched loses | 1, 0 | Outer { lanes: 8 } / Serial |
| 256 | 32 | 8 | 41076 (1248) | 249473 (1436) | 0.16 | batched loses | 1, 0 | Outer { lanes: 8 } / Serial |

### Per-job non-regression (base ebc74c3 vs cand ae1bc79, per-job arm, 3 process pairs)

| jobs | m=k=n | threads | base process medians ns | cand process medians ns | cand/base (median of medians) | verdict |
|---:|---:|---:|---|---|---:|---|
| 64 | 2 | 1 | 6185, 6115, 6033 | 6303, 6353, 6288 | 1.031 | REGRESSION |
| 64 | 4 | 1 | 6530, 6464, 6465 | 6683, 6802, 6869 | 1.052 | REGRESSION |
| 64 | 8 | 1 | 7770, 7734, 7742 | 8050, 8189, 8198 | 1.058 | REGRESSION |
| 64 | 16 | 1 | 20225, 20187, 20241 | 20290, 20440, 20558 | 1.011 | no regression |
| 64 | 32 | 1 | 67653, 67650, 67824 | 68884, 67020, 68384 | 1.011 | no regression |
| 256 | 2 | 1 | 24491, 24220, 23982 | 25290, 25175, 25143 | 1.039 | REGRESSION |
| 256 | 4 | 1 | 25678, 25688, 25669 | 26334, 26551, 27151 | 1.034 | REGRESSION |
| 256 | 8 | 1 | 31774, 31732, 31333 | 31519, 31408, 31794 | 0.993 | no regression |
| 256 | 16 | 1 | 86346, 85775, 86003 | 85064, 84052, 85100 | 0.989 | no regression |
| 256 | 32 | 1 | 270220, 275384, 272481 | 275011, 270530, 277856 | 1.009 | no regression |
| 64 | 2 | 8 | 6009, 5984, 6028 | 6261, 6280, 6267 | 1.043 | REGRESSION |
| 64 | 4 | 8 | 6499, 6493, 6483 | 6665, 6667, 6680 | 1.027 | no regression |
| 64 | 8 | 8 | 5996, 6332, 6048 | 6361, 6363, 6383 | 1.052 | REGRESSION |
| 64 | 16 | 8 | 7224, 7195, 7131 | 7753, 7809, 8071 | 1.085 | REGRESSION |
| 64 | 32 | 8 | 10904, 10884, 10921 | 10970, 11034, 11145 | 1.012 | no regression |
| 256 | 2 | 8 | 24244, 23920, 23805 | 25070, 25041, 25071 | 1.048 | REGRESSION |
| 256 | 4 | 8 | 25606, 25494, 25647 | 26287, 26307, 26380 | 1.027 | no regression |
| 256 | 8 | 8 | 23730, 23502, 23683 | 23943, 23876, 24111 | 1.011 | no regression |
| 256 | 16 | 8 | 23138, 23141, 23030 | 23476, 23470, 23505 | 1.015 | no regression |
| 256 | 32 | 8 | 38794, 39047, 39129 | 39325, 39848, 39388 | 1.009 | no regression |
