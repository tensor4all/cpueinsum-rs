//! Issue #46 diagnostic; protocol and recorder are in repository-root benchmarks/issue46/.
//! Uses this crate's grouped tests' execution pattern and tcontract-exec's
//! batch.rs:faer_auto rule as the scheduling reference; no kernel is copied.
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::{Algorithm, FaerLimit, Plan, PlanConfig};
use cpueinsum::{
    ArenaProvider, BatchPolicy, BatchStrategy, Exec, GroupedGemmJob, GroupedPlan, GroupedRoute,
};

fn shapes(name: &str) -> Vec<(usize, usize, usize)> {
    match name {
        "A" => vec![(1024, 256, 1024); 2],
        "B" => vec![(32, 32, 32); 16384],
        "C" => {
            let mut v = vec![(1024, 256, 1024)];
            v.extend(vec![(32, 32, 32); 8192]);
            v
        }
        "large_job" => vec![(1024, 256, 1024)],
        "small_job" => vec![(32, 32, 32)],
        _ => panic!("unknown workload {name}"),
    }
}

fn policy(name: &str) -> BatchPolicy {
    BatchPolicy::new(match name {
        "Auto" => BatchStrategy::Auto,
        "Inner" => BatchStrategy::Inner,
        "Split2" => BatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        },
        "Split4" => BatchStrategy::Split {
            lanes: NonZeroUsize::new(4).unwrap(),
        },
        _ => panic!("unknown strategy {name}"),
    })
}

fn widths(route: GroupedRoute) -> (usize, usize) {
    match route {
        GroupedRoute::Serial => (1, 1),
        GroupedRoute::Outer { lanes } => (lanes, 1),
        GroupedRoute::Inner { threads } => (1, threads),
        GroupedRoute::Split {
            lanes,
            item_threads,
        } => (lanes, item_threads),
        _ => panic!("unexpected route {route:?}"),
    }
}

fn row(id: usize, i: usize) -> f64 {
    ((id + i) % 7) as f64 - 3.0
}
fn col(id: usize, j: usize) -> f64 {
    ((id + j) % 5) as f64 - 2.0
}

fn verify(jobs: &[GroupedGemmJob], out: &[f64]) {
    for (id, job) in jobs.iter().enumerate() {
        for j in 0..job.cols {
            for i in 0..job.rows {
                let expected = job.contracted as f64 * row(id, i) * col(id, j) / 64.0;
                let got = out[job.out_offset + i + job.rows * j];
                assert!(
                    got.is_finite() && (got - expected).abs() <= 1e-11 * expected.abs().max(1.0),
                    "job={id} i={i} j={j}: {got} != {expected}"
                );
            }
        }
    }
}

// Linux process CPU ticks include all threads; CPU window is separate from the
// primary single-call time, so tick quantization cannot change that statistic.
fn cpu_ticks() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<_> = stat
        .rsplit_once(") ")
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap()
}

fn assert_faer(m: usize, k: usize, n: usize, cfg: &PlanConfig) {
    let spec = |dims: &[usize], strides: &[isize]| {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
    };
    let p = Problem::from_labels(
        DType::F64,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        CSpec::Output(cpueinsum::tcontract::api::Op::Identity),
        spec(&[m, n], &[1, m as isize]),
        &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
    )
    .unwrap();
    assert_eq!(
        Plan::<f64>::new(&p, cfg).unwrap().report().algorithm,
        Algorithm::Faer
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "workload threads strategy warm_ms cpu_ms");
    let name = &args[1];
    let threads: usize = args[2].parse().unwrap();
    assert!(threads > 0);
    let strategy = &args[3];
    let warm = Duration::from_millis(args[4].parse().unwrap());
    let cpu_window = Duration::from_millis(args[5].parse().unwrap());
    let cfg = PlanConfig {
        faer_limit: FaerLimit::NONE,
        ..PlanConfig::default()
    };
    let shape = shapes(name);
    let macs: usize = shape.iter().map(|&(m, k, n)| m * k * n).sum();
    if matches!(name.as_str(), "A" | "B" | "C") {
        assert_eq!(macs, 536870912);
    }
    let mut geometries = shape.clone();
    geometries.sort_unstable();
    geometries.dedup();
    for &(m, k, n) in &geometries {
        assert_faer(m, k, n, &cfg);
    }
    let mut jobs = Vec::with_capacity(shape.len());
    let (mut al, mut bl, mut dl) = (0, 0, 0);
    for &(m, k, n) in &shape {
        jobs.push(GroupedGemmJob::new(al, bl, dl, m, k, n));
        al += m * k;
        bl += k * n;
        dl += m * n;
    }
    let (mut a, mut b, mut d) = (vec![0.0; al], vec![0.0; bl], vec![f64::NAN; dl]);
    for (id, job) in jobs.iter().enumerate() {
        for p in 0..job.contracted {
            let sign = if p % 2 == 0 { 1.0 } else { -1.0 };
            for i in 0..job.rows {
                a[job.lhs_offset + i + job.rows * p] = row(id, i) * sign / 8.0;
            }
            for j in 0..job.cols {
                b[job.rhs_offset + p + job.contracted * j] = sign * col(id, j) / 8.0;
            }
        }
    }
    let plan = GroupedPlan::<f64>::new(&jobs, &cfg).unwrap();
    let pool = threadteam::Pool::builder()
        .threads(threads)
        .pin_participants(false)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    assert_eq!(pool.width().get(), threads);
    let team = pool
        .into_team(threads)
        .expect("dedicated idle pool must admit team");
    assert_eq!(team.size().get(), threads);
    let arena = ArenaProvider::new();
    let mut exec = Exec::from_token(team.exec(threads), Some(&arena), policy(strategy));
    assert_eq!(exec.budget(), threads);
    assert!(threads == 1 || exec.token_ref().carries_team());
    let largest = shape.iter().map(|&(m, k, n)| m * k * n).max().unwrap();
    let item_auto_width = exec.width_for(
        2.0 * largest as f64 * cpueinsum::tcontract::ns_per_flop(),
        1,
    );
    if name == "A" && threads >= 16 {
        assert!(item_auto_width > 8);
    }
    let mut run = || {
        plan.execute_into(
            &mut exec,
            1.0,
            black_box(&a),
            black_box(&b),
            black_box(&mut d),
        )
        .unwrap()
    };
    let route = run();
    let start = Instant::now();
    while start.elapsed() < warm {
        assert_eq!(run(), route);
    }
    drop(run);
    verify(&jobs, &d);
    // Restore the execution-warm state after the untimed full-output scan.
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &a, &b, &mut d).unwrap(),
        route
    );
    let start = Instant::now();
    let observed = plan
        .execute_into(
            &mut exec,
            1.0,
            black_box(&a),
            black_box(&b),
            black_box(&mut d),
        )
        .unwrap();
    let wall_ns = start.elapsed().as_nanos();
    assert_eq!(observed, route);
    let c0 = cpu_ticks();
    let start = Instant::now();
    let mut calls = 0u64;
    while start.elapsed() < cpu_window {
        assert_eq!(
            plan.execute_into(
                &mut exec,
                1.0,
                black_box(&a),
                black_box(&b),
                black_box(&mut d)
            )
            .unwrap(),
            route
        );
        calls += 1;
    }
    let cpu_wall_ns = start.elapsed().as_nanos();
    let ticks = cpu_ticks() - c0;
    verify(&jobs, black_box(&d));
    let (lanes, item_threads) = widths(route);
    if strategy != "Auto" {
        let expected_lanes = match strategy.as_str() {
            "Inner" => 1,
            "Split2" => 2.min(threads).min(jobs.len()),
            "Split4" => 4.min(threads).min(jobs.len()),
            _ => unreachable!(),
        };
        assert_eq!(
            (lanes, item_threads),
            (expected_lanes, threads / expected_lanes)
        );
    }
    let work: Vec<_> = (0..lanes)
        .map(|lane| {
            let lo = lane * (jobs.len() / lanes) + lane.min(jobs.len() % lanes);
            let hi = lo + jobs.len() / lanes + usize::from(lane < jobs.len() % lanes);
            shape[lo..hi]
                .iter()
                .map(|&(m, k, n)| m * k * n)
                .sum::<usize>()
        })
        .collect();
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let affinity = status
        .lines()
        .find(|l| l.starts_with("Cpus_allowed_list:"))
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap();
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let nice = stat
        .rsplit_once(") ")
        .unwrap()
        .1
        .split_whitespace()
        .nth(16)
        .unwrap();
    println!("workload,threads,strategy,jobs,macs,bytes,wall_ns,cpu_wall_ns,cpu_calls,cpu_ticks,lanes,item_threads,item_auto_width,lane_min_macs,lane_max_macs,affinity,nice,build");
    println!("{name},{threads},{strategy},{},{macs},{},{wall_ns},{cpu_wall_ns},{calls},{ticks},{lanes},{item_threads},{item_auto_width},{},{},{affinity},{nice},{}",
        jobs.len(), (al+bl+dl)*8, work.iter().min().unwrap(), work.iter().max().unwrap(), option_env!("ISSUE46_BUILD").unwrap_or("unlabelled"));
}
