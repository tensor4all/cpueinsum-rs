#!/usr/bin/env python3
"""Record the predeclared #46 diagnostic; stdlib only. See PROTOCOL.md."""
import argparse
import csv
import datetime
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = [("baseline", s) for s in ("Auto", "Inner", "Split2", "Split4")] + [("candidate", "Auto")]
THREADS = (1, 8, 16, 32, 64)
WORKLOADS = ("A", "B", "C", "large_job", "small_job")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cpu_list(text):
    result = set()
    for item in text.strip().split(","):
        lo, _, hi = item.partition("-")
        result.update(range(int(lo), int(hi or lo) + 1))
    return result


def topology():
    return sorted({tuple(sorted(cpu_list(p.read_text()))) for p in
                   Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cache/index3/shared_cpu_list")})


def command(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return {"args": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def stats():
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        key, *values = line.split()
        if key.startswith("cpu") and key != "cpu":
            result[int(key[3:])] = list(map(int, values[:8]))
    return result


def idle_window(cpus):
    a = stats()
    time.sleep(1)
    b = stats()
    total = sum(sum(b[c][i] - a[c][i] for i in range(8)) for c in cpus)
    idle = sum(b[c][3] - a[c][3] + b[c][4] - a[c][4] for c in cpus)
    return {"before": a, "after": b, "busy_fraction": (total - idle) / total if total else None}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    domains = topology()
    observer = set(domains[-1])
    assert os.sched_getaffinity(0) <= observer, "pin recorder to observer L3 domain first"
    all_cpus = [c for domain in domains for c in domain]
    sets = {n: all_cpus[:n] for n in THREADS}
    assert all(len(sets[n]) == n for n in THREADS)
    assert all(not (set(sets[n]) & observer) for n in (1, 8, 16, 32))
    bins = {arm: Path(getattr(args, arm)).resolve() for arm in ("baseline", "candidate")}
    env = os.environ.copy()
    for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(key, None)
    metadata = {
        "started": now(), "argv": vars(args), "domains": domains,
        "observer": sorted(observer), "cpu_sets": sets, "clock_ticks": os.sysconf("SC_CLK_TCK"),
        "binary_sha256": {arm: sha(p) for arm, p in bins.items()},
        "patch_sha256": sha(HERE / "team.patch"),
        "bench_sha256": sha(HERE.parent / "issue46_grouped.rs"),
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "commands": [command(a) for a in (
            ["git", "rev-parse", "HEAD"], ["git", "status", "--short"],
            ["git", "diff", "--", "Cargo.lock", "crates/cpueinsum/src/grouped.rs"],
            ["rustc", "-Vv"], ["cargo", "-V"], ["lscpu"], ["uname", "-a"],
            ["ps", "-eo", "pid,psr,pcpu,comm", "--sort=-pcpu"],
        )],
        "proc_cpuinfo": Path("/proc/cpuinfo").read_text(),
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "governors": {str(p): p.read_text().strip() for p in
                      Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")},
        "64T_note": "観測側と共有 / shared with observer; reference only",
    }
    (out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    conditions = [(w, n) for w in WORKLOADS for n in THREADS]
    started = time.monotonic()
    next_report = started + 600
    with (out / "raw.jsonl").open("w", buffering=1) as log:
        for rep in range(7):
            guard = idle_window(all_cpus)
            log.write(json.dumps({"event": "round_start", "round": rep, "time": now(), "idle": guard}) + "\n")
            rotated = conditions[rep % len(conditions):] + conditions[:rep % len(conditions)]
            for workload, width in rotated:
                arms = ARMS if rep % 2 == 0 else list(reversed(ARMS))
                for arm, strategy in arms:
                    mask = ",".join(map(str, sets[width]))
                    cmd = ["nice", "-n", "5", "taskset", "-c", mask, str(bins[arm]), workload, str(width), strategy, "250", "250"]
                    record = {"event": "cell", "round": rep, "arm": arm, "workload": workload,
                              "threads": width, "strategy": strategy, "time": now(), "command": cmd,
                              "cpus": sets[width], "before": stats()}
                    try:
                        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
                        record.update(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
                        if p.returncode == 0:
                            rows = list(csv.DictReader(io.StringIO(p.stdout)))
                            assert len(rows) == 1, rows
                            row = rows[0]
                            for k in row.keys() - {"workload", "strategy", "affinity", "build"}:
                                row[k] = int(row[k])
                            assert row["threads"] == width and row["workload"] == workload
                            assert row["strategy"] == strategy and row["build"] == arm
                            assert cpu_list(row["affinity"]) == set(sets[width])
                            assert row["nice"] == 5
                            record["row"] = row
                    except (subprocess.TimeoutExpired, AssertionError, ValueError) as e:
                        record["error"] = repr(e)
                    record["after"] = stats()
                    log.write(json.dumps(record) + "\n")
                    print(f"round={rep} {workload} {width}T {arm}/{strategy}: "
                          f"{record.get('row', {}).get('wall_ns', record.get('error', record.get('returncode')))}", flush=True)
                    if time.monotonic() >= next_report:
                        with (ROOT / "report").open("a") as report:
                            report.write(f"{now()} | 条件: nice5, 7-round interleaved, 観測{sorted(observer)} | 結果: round={rep} {workload}/{width}T/{arm}/{strategy}, 完了セルはraw.jsonlに保持 | 次: 残り全条件→集計（選別再試行なし）\n")
                        next_report += 600
            log.write(json.dumps({"event": "round_end", "round": rep, "time": now(),
                                  "idle": idle_window(all_cpus),
                                  "processes": command(["ps", "-eo", "pid,psr,pcpu,comm", "--sort=-pcpu"])}) + "\n")
    (out / "finished.json").write_text(json.dumps({"finished": now(), "elapsed_seconds": time.monotonic() - started,
        "commands": [command(["git", "diff", "--", "Cargo.lock", "crates/cpueinsum/src/grouped.rs"])],
        "binary_sha256": {arm: sha(p) for arm, p in bins.items()}} , indent=2) + "\n")
    summarize(out)


def milliseconds(ns):
    return f"{ns // 1000000}.{ns % 1000000:06d}"


def summarize(out):
    records = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
    cells = [r for r in records if r["event"] == "cell"]
    environment = json.loads((out / "environment.json").read_text())
    hz = environment["clock_ticks"]
    cpu_note = []
    for width, cpus in environment["cpu_sets"].items():
        domains = sum(bool(set(cpus) & set(d)) for d in environment["domains"])
        mask = f"{cpus[0]}-{cpus[-1]}" if len(cpus) > 1 and cpus == list(range(cpus[0], cpus[-1]+1)) else ",".join(map(str, cpus))
        cpu_note.append(f"{width}T={mask}/{domains} L3")
    lines = ["# Same-library diagnostic (not official performance)", "",
             "All times are warm median ms (7 calls); ratios are exact fractions relative to the fastest of the five arms.",
             "CPU% includes waiting/spinning; it is not a measure of useful arithmetic. See raw windows for tick quantization.",
             "Measured CPUs/L3 domains: " + "; ".join(cpu_note),
             f"Observer={environment['observer']}; **64T: 観測側と共有 / reference only**. Sets are derived from environment.json.", "",
             "| Workload | Threads | Baseline Auto ms (relative) | Inner | Split2 | Split4 | Candidate Auto |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    diagnostics = ["| Workload | T | Arm | Strategy | lanes × item threads | item Auto width | median CPU% | max/min time | lane MAC min–max |", "|---|---:|---|---|---|---:|---:|---:|---|"]
    summary = []
    comparisons = []
    for w in WORKLOADS:
        for n in THREADS:
            samples = {}
            for arm, strategy in ARMS:
                rr = [r for r in cells if (r["workload"], r["threads"], r["arm"], r["strategy"]) == (w, n, arm, strategy)]
                good = [r for r in rr if "row" in r and "error" not in r]
                if len(rr) != 7 or len(good) != 7:
                    samples[arm, strategy] = None
                    continue
                rows = [r["row"] for r in good]
                times = [r["wall_ns"] for r in rows]
                mid = statistics.median(times)
                spread = Fraction(max(times), min(times))
                cpu = statistics.median([Fraction(r["cpu_ticks"] * 100 * 1000000000, hz * r["cpu_wall_ns"]) for r in rows])
                item = {"workload": w, "threads": n, "arm": arm, "strategy": strategy,
                        "median_ns": mid, "min_ns": min(times), "max_ns": max(times),
                        "unstable": spread > Fraction(5, 4), "cpu_percent_exact": str(cpu),
                        "samples_ns": times, "rounds": [r["round"] for r in good],
                        "lanes": sorted({r["lanes"] for r in rows}),
                        "item_threads": sorted({r["item_threads"] for r in rows})}
                summary.append(item)
                samples[arm, strategy] = item
                routes = sorted({(r["lanes"], r["item_threads"]) for r in rows})
                diagnostics.append(f"| {w} | {n} | {arm} | {strategy} | {routes} | {rows[0]['item_auto_width']} | {cpu} | {spread}{' UNSTABLE' if item['unstable'] else ''} | {rows[0]['lane_min_macs']}–{rows[0]['lane_max_macs']} |")
            best = min((v["median_ns"] for v in samples.values() if v), default=None)
            values = []
            for arm in ARMS:
                value = samples[arm]
                if value is None:
                    values.append("FAILED / incomplete (see raw)")
                else:
                    ratio = Fraction(value["median_ns"], best)
                    values.append(f"{milliseconds(value['median_ns'])} ({'1.00' if ratio == 1 else str(ratio)})" + (" UNSTABLE" if value["unstable"] else ""))
            lines.append(f"| {w} | {n}{'†' if n == 64 else ''} | " + " | ".join(values) + " |")
            if all(samples.values()):
                auto, inner, s2, s4, candidate = (samples[arm] for arm in ARMS)
                assert all(v['rounds'] == list(range(7)) for v in samples.values())
                ni = sum(Fraction(a, b) > Fraction(5, 4) for a, b in zip(auto['samples_ns'], inner['samples_ns']))
                ns = sum(Fraction(a, min(b, c)) > Fraction(5, 4) for a, b, c in zip(auto['samples_ns'], s2['samples_ns'], s4['samples_ns']))
                ratio_i = Fraction(auto['median_ns'], inner['median_ns'])
                ratio_s = Fraction(auto['median_ns'], min(s2['median_ns'], s4['median_ns']))
                comparisons.append({'workload': w, 'threads': n, 'auto_over_inner': str(ratio_i),
                    'auto_over_best_split': str(ratio_s), 'auto_slower_inner_pairs': ni,
                    'auto_slower_best_split_pairs': ns,
                    'candidate_over_auto': str(Fraction(candidate['median_ns'], auto['median_ns'])),
                    'candidate_slower_10pct_pairs': sum(Fraction(c, a) > Fraction(11, 10) for c, a in zip(candidate['samples_ns'], auto['samples_ns'])),
                    'candidate_or_auto_unstable': candidate['unstable'] or auto['unstable'],
                    'reproduction_gate': n in (8,16,32) and ratio_i > Fraction(5,4) and ratio_s > Fraction(5,4) and ni >= 6 and ns >= 6})
    failures = [{k: v for k, v in r.items() if k not in ("before", "after")} for r in cells if "row" not in r or "error" in r]
    guards = [{"event": r["event"], "round": r["round"], "busy_fraction": r["idle"]["busy_fraction"]} for r in records if "idle" in r]
    lines += ["", "## Paired direction (thresholds from PROTOCOL.md)", "",
        "| Workload | T | Auto/Inner | Auto/best Split | pairs >1.25 Inner | pairs >1.25 best Split | candidate/Auto | candidate >1.10 Auto pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['workload']} | {r['threads']} | {r['auto_over_inner']} | {r['auto_over_best_split']} | {r['auto_slower_inner_pairs']}/7 | {r['auto_slower_best_split_pairs']}/7 | {r['candidate_over_auto']} | {r['candidate_slower_10pct_pairs']}/7 |" for r in comparisons]
    lines += ["", "## Scheduler / CPU diagnostics (exact fractions)", ""] + diagnostics
    lines += ["", f"Failed/incomplete executions: {len(failures)}", "", "## Idle windows", "", "```json", json.dumps(guards, indent=2), "```", ""]
    (out / "tables.md").write_text("\n".join(lines))
    gates = [r['threads'] for r in comparisons if r['workload'] == 'A' and r['reproduction_gate']]
    (out / "summary.json").write_text(json.dumps({"cells": summary, "comparisons": comparisons,
        "reproduction_widths": gates, "large_job_reproduced": len(gates) >= 2,
        "failures": failures, "idle": guards}, indent=2) + "\n")


def self_test():
    assert cpu_list("0-3,6,8-9") == {0, 1, 2, 3, 6, 8, 9}
    assert milliseconds(1234567) == "1.234567"
    assert milliseconds(0) == "0.000000"
    assert Fraction(statistics.median([3, 1, 4, 5, 2]), 2) == Fraction(3, 2)
    assert len(set(ARMS)) == 5 and len(WORKLOADS) * len(THREADS) * len(ARMS) * 7 == 875
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "environment.json").write_text(json.dumps({"clock_ticks": 100,
            "cpu_sets": {"1": [0]}, "domains": [[0]], "observer": [1]}))
        records = [{"event": "cell", "round": i, "workload": "A", "threads": 1,
            "arm": "baseline", "strategy": "Auto", "row": {"wall_ns": t,
            "cpu_ticks": 1, "cpu_wall_ns": 10000000, "lanes": 1, "item_threads": 1,
            "item_auto_width": 1, "lane_min_macs": 1, "lane_max_macs": 1}}
            for i,t in enumerate([7, 6, 5, 4, 3, 2, 1])]
        (out / "raw.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        summarize(out)
        result = json.loads((out / "summary.json").read_text())["cells"][0]
        assert result["median_ns"] == 4 and result["unstable"]
        assert result["cpu_percent_exact"] == "100"
        records[-1]["error"] = "synthetic failed condition"
        (out / "raw.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        summarize(out)
        result = json.loads((out / "summary.json").read_text())
        assert not result["cells"] and len(result["failures"]) == 1
        assert "FAILED / incomplete" in (out / "tables.md").read_text()
    print("recorder self-check passed (aggregation, exact units, failures)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--baseline", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--output", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("output")
    sub.add_parser("self-test")
    a = p.parse_args()
    if a.action == "run":
        run(a)
    elif a.action == "summarize":
        summarize(Path(a.output))
    else:
        self_test()
