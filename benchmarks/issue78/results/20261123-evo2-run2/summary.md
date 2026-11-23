### Batched vs per-job (cand-d7e0e68, one paired process per width)

| jobs | m=k=n | threads | per-job median (IQR) ns | batched median (IQR) ns | per-job/batched | verdict | allocs/call (per-job, batched) | route (per-job / batched) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 64 | 2 | 1 | 6237 (75) | 1900 (6) | 3.28 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 1 | 6651 (24) | 2634 (18) | 2.53 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 1 | 7953 (18) | 3762 (14) | 2.11 | batched wins | 0, 0 | Serial / Serial |
| 64 | 16 | 1 | 20141 (42) | 16015 (29) | 1.26 | batched wins | 0, 0 | Serial / Serial |
| 64 | 32 | 1 | 66733 (220) | 61503 (169) | 1.09 | batched wins | 0, 0 | Serial / Serial |
| 256 | 2 | 1 | 24754 (72) | 7365 (43) | 3.36 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 1 | 26437 (78) | 10068 (44) | 2.63 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 1 | 31946 (20) | 14468 (22) | 2.21 | batched wins | 0, 0 | Serial / Serial |
| 256 | 16 | 1 | 87560 (1781) | 68978 (1015) | 1.27 | batched wins | 0, 0 | Serial / Serial |
| 256 | 32 | 1 | 274407 (2394) | 250036 (1564) | 1.10 | batched wins | 0, 0 | Serial / Serial |
| 64 | 2 | 8 | 6400 (27) | 1931 (18) | 3.31 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 8 | 6663 (29) | 2673 (9) | 2.49 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 8 | 6454 (75) | 2020 (8) | 3.20 | batched wins | 1, 0 | Outer { lanes: 3 } / Inner { threads: 3 } |
| 64 | 16 | 8 | 7013 (30) | 3316 (38) | 2.11 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 64 | 32 | 8 | 11245 (147) | 64946 (159) | 0.17 | batched loses | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 2 | 8 | 24676 (204) | 7314 (57) | 3.37 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 8 | 26302 (93) | 10054 (113) | 2.62 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 8 | 23440 (86) | 3154 (36) | 7.43 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 16 | 8 | 22597 (313) | 9320 (111) | 2.42 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 32 | 8 | 40727 (1325) | 262606 (1121) | 0.16 | batched loses | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |

### Per-job non-regression (base-ebc74c3 vs cand-d7e0e68, per-job arm, 3 process pairs)

| jobs | m=k=n | threads | base process medians ns | cand process medians ns | cand/base (median of medians) | verdict |
|---:|---:|---:|---|---|---:|---|
| 64 | 2 | 1 | 6042, 6041, 6065 | 6224, 6257, 6225 | 1.030 | REGRESSION |
| 64 | 4 | 1 | 6466, 6464, 6616 | 6787, 6656, 6775 | 1.048 | REGRESSION |
| 64 | 8 | 1 | 7810, 7796, 7896 | 8101, 8107, 8186 | 1.038 | REGRESSION |
| 64 | 16 | 1 | 20269, 20258, 20219 | 20410, 20307, 20336 | 1.004 | no regression |
| 64 | 32 | 1 | 66817, 67819, 67624 | 67232, 67601, 68125 | 1.000 | no regression |
| 256 | 2 | 1 | 24033, 24141, 24050 | 24846, 24874, 24789 | 1.033 | REGRESSION |
| 256 | 4 | 1 | 25838, 25825, 26468 | 27219, 27118, 26451 | 1.050 | no regression |
| 256 | 8 | 1 | 31649, 31797, 31856 | 31693, 31759, 31608 | 0.997 | no regression |
| 256 | 16 | 1 | 84477, 84573, 84018 | 84346, 84121, 83854 | 0.996 | no regression |
| 256 | 32 | 1 | 272531, 272953, 270406 | 278477, 272810, 277627 | 1.019 | no regression |
| 64 | 2 | 8 | 6000, 6000, 6147 | 6391, 6192, 6378 | 1.063 | REGRESSION |
| 64 | 4 | 8 | 6465, 6489, 6491 | 6611, 6617, 6645 | 1.020 | no regression |
| 64 | 8 | 8 | 5946, 6004, 5952 | 6332, 6251, 6352 | 1.064 | REGRESSION |
| 64 | 16 | 8 | 7145, 7214, 7224 | 7082, 7121, 7086 | 0.982 | no regression |
| 64 | 32 | 8 | 11006, 11109, 10970 | 10924, 11044, 11098 | 1.003 | no regression |
| 256 | 2 | 8 | 23889, 23895, 23922 | 24661, 24671, 24605 | 1.032 | REGRESSION |
| 256 | 4 | 8 | 25569, 25774, 25763 | 26635, 26370, 26300 | 1.024 | no regression |
| 256 | 8 | 8 | 23077, 23772, 23670 | 23593, 23597, 23581 | 0.997 | no regression |
| 256 | 16 | 8 | 22939, 23097, 23147 | 22620, 22701, 22656 | 0.981 | no regression |
| 256 | 32 | 8 | 38763, 39381, 39763 | 39371, 39095, 39663 | 1.000 | no regression |
