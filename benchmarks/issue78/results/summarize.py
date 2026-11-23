"""Summarize one #78 paired run: python3 summarize.py <run dir> <cand label>."""
import csv, statistics as st, sys
d, cand = sys.argv[1], sys.argv[2]
def load(n): return list(csv.DictReader(open(f"{d}/{n}")))
print(f"### Batched vs per-job ({cand}, one paired process per width)\n")
print("| jobs | m=k=n | threads | per-job median (IQR) ns | batched median (IQR) ns | per-job/batched | verdict | allocs/call (per-job, batched) | route (per-job / batched) |")
print("|---:|---:|---:|---:|---:|---:|---|---|---|")
for t in (1, 8):
    by = {(int(r['jobs']), int(r['m']), r['route']): r for r in load(f"paired-t{t}.csv")}
    for J in (64, 256):
        for s in (2, 4, 8, 16, 32):
            p, b = by[(J, s, 'per-job')], by[(J, s, 'batched')]
            pm, pq1, pq3 = (float(p[k]) for k in ('median_ns', 'q1_ns', 'q3_ns'))
            bm, bq1, bq3 = (float(b[k]) for k in ('median_ns', 'q1_ns', 'q3_ns'))
            v = 'batched wins' if bq3 < pq1 else ('batched loses' if bq1 > pq3 else 'no difference')
            print(f"| {J} | {s} | {t} | {pm:.0f} ({pq3-pq1:.0f}) | {bm:.0f} ({bq3-bq1:.0f}) | {pm/bm:.2f} | {v} | {p['allocs_per_call']}, {b['allocs_per_call']} | {p['route_reported']} / {b['route_reported']} |")
print(f"\n### Per-job non-regression (base-ebc74c3 vs {cand}, per-job arm, 3 process pairs)\n")
print("| jobs | m=k=n | threads | base process medians ns | cand process medians ns | cand/base (median of medians) | verdict |")
print("|---:|---:|---:|---|---|---:|---|")
for t in (1, 8):
    meds = {}
    for who in ('base', 'cand'):
        for i in (1, 2, 3):
            for r in load(f"{who}-t{t}-p{i}.csv"):
                meds.setdefault((who, int(r['jobs']), int(r['m'])), []).append(float(r['median_ns']))
    for J in (64, 256):
        for s in (2, 4, 8, 16, 32):
            b, c = meds[('base', J, s)], meds[('cand', J, s)]
            ratio = st.median(c) / st.median(b)
            v = 'REGRESSION' if (ratio > 1.03 and min(c) > max(b)) else 'no regression'
            print(f"| {J} | {s} | {t} | {', '.join(f'{x:.0f}' for x in b)} | {', '.join(f'{x:.0f}' for x in c)} | {ratio:.3f} | {v} |")
