### Batched vs per-job (cand-1f35056, one paired process per width)

| jobs | m=k=n | threads | per-job median (IQR) ns | batched median (IQR) ns | per-job/batched | verdict | allocs/call (per-job, batched) | route (per-job / batched) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 64 | 2 | 1 | 6163 (68) | 1907 (15) | 3.23 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 1 | 6655 (25) | 2617 (6) | 2.54 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 1 | 7965 (80) | 3823 (12) | 2.08 | batched wins | 0, 0 | Serial / Serial |
| 64 | 16 | 1 | 20256 (38) | 16040 (96) | 1.26 | batched wins | 0, 0 | Serial / Serial |
| 64 | 32 | 1 | 68671 (1042) | 63994 (243) | 1.07 | batched wins | 0, 0 | Serial / Serial |
| 256 | 2 | 1 | 24539 (58) | 7354 (48) | 3.34 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 1 | 26408 (138) | 10076 (26) | 2.62 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 1 | 31533 (77) | 14484 (26) | 2.18 | batched wins | 0, 0 | Serial / Serial |
| 256 | 16 | 1 | 84159 (386) | 66461 (153) | 1.27 | batched wins | 0, 0 | Serial / Serial |
| 256 | 32 | 1 | 278690 (1576) | 257337 (2934) | 1.08 | batched wins | 0, 0 | Serial / Serial |
| 64 | 2 | 8 | 6173 (55) | 1936 (19) | 3.19 | batched wins | 0, 0 | Serial / Serial |
| 64 | 4 | 8 | 6904 (62) | 2682 (8) | 2.57 | batched wins | 0, 0 | Serial / Serial |
| 64 | 8 | 8 | 6116 (94) | 2039 (76) | 3.00 | batched wins | 1, 0 | Outer { lanes: 3 } / Inner { threads: 3 } |
| 64 | 16 | 8 | 7519 (760) | 3460 (120) | 2.17 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 64 | 32 | 8 | 11467 (306) | 65020 (620) | 0.18 | batched loses | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 2 | 8 | 24398 (58) | 7351 (44) | 3.32 | batched wins | 0, 0 | Serial / Serial |
| 256 | 4 | 8 | 26335 (134) | 10064 (50) | 2.62 | batched wins | 0, 0 | Serial / Serial |
| 256 | 8 | 8 | 24226 (380) | 3275 (358) | 7.40 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 16 | 8 | 23567 (397) | 9413 (206) | 2.50 | batched wins | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |
| 256 | 32 | 8 | 41166 (1143) | 259689 (1108) | 0.16 | batched loses | 1, 0 | Outer { lanes: 8 } / Inner { threads: 8 } |

### Per-job non-regression (base-ebc74c3 vs cand-1f35056, per-job arm, 3 process pairs)

| jobs | m=k=n | threads | base process medians ns | cand process medians ns | cand/base (median of medians) | verdict |
|---:|---:|---:|---|---|---:|---|
| 64 | 2 | 1 | 6068, 6032, 6077 | 6095, 6116, 6129 | 1.008 | no regression |
| 64 | 4 | 1 | 6506, 6496, 6548 | 6667, 6685, 6694 | 1.028 | no regression |
| 64 | 8 | 1 | 7840, 7820, 7821 | 7951, 7976, 8049 | 1.020 | no regression |
| 64 | 16 | 1 | 20361, 20283, 20315 | 20187, 20175, 20151 | 0.993 | no regression |
| 64 | 32 | 1 | 69728, 67193, 67619 | 69392, 68440, 68495 | 1.013 | no regression |
| 256 | 2 | 1 | 24084, 24041, 24070 | 24535, 24616, 24665 | 1.023 | no regression |
| 256 | 4 | 1 | 26006, 26177, 26265 | 26415, 26941, 26480 | 1.012 | no regression |
| 256 | 8 | 1 | 31841, 31480, 31527 | 31142, 31232, 31224 | 0.990 | no regression |
| 256 | 16 | 1 | 85244, 85712, 84289 | 83123, 83116, 82889 | 0.975 | no regression |
| 256 | 32 | 1 | 282289, 268763, 277592 | 282308, 276035, 279502 | 1.007 | no regression |
| 64 | 2 | 8 | 6001, 6061, 6018 | 6172, 6062, 6073 | 1.009 | no regression |
| 64 | 4 | 8 | 6532, 6581, 6569 | 6615, 6601, 6616 | 1.007 | no regression |
| 64 | 8 | 8 | 6114, 6066, 6303 | 6465, 6362, 6270 | 1.041 | no regression |
| 64 | 16 | 8 | 7517, 7256, 7323 | 8071, 7956, 8140 | 1.102 | REGRESSION |
| 64 | 32 | 8 | 11021, 10887, 10903 | 11123, 11136, 11117 | 1.020 | no regression |
| 256 | 2 | 8 | 23825, 23952, 23868 | 24497, 24566, 24580 | 1.029 | no regression |
| 256 | 4 | 8 | 25901, 26071, 25792 | 26252, 26345, 26256 | 1.014 | no regression |
| 256 | 8 | 8 | 23890, 23252, 23908 | 24282, 24107, 24321 | 1.016 | no regression |
| 256 | 16 | 8 | 23293, 23016, 23104 | 23414, 23299, 24333 | 1.013 | no regression |
| 256 | 32 | 8 | 38970, 39355, 39084 | 39229, 39481, 39316 | 1.006 | no regression |
