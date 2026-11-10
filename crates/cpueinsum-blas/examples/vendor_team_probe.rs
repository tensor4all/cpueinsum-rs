//! What the vendor GEMM sees of the caller's CPU set, and what it costs.
//!
//! ```text
//! taskset -c 0-3 cargo run --release -p cpueinsum-blas --example vendor_team_probe -- serial 2048 5
//! taskset -c 0-3 cargo run --release -p cpueinsum-blas --example vendor_team_probe -- team 2048 5
//! ```
//!
//! `serial` runs the plan from `Exec::serial()`; `team` builds a pool of four,
//! takes a team of four and runs the plan from its leader. Both print the
//! calling thread's mask before the call, the CPU time the process spent per
//! wall second inside it (a vendor team confined to one CPU shows ~1, a team on
//! four CPUs ~4), the median call, and a checksum of the output.

extern crate blas_src as _;

use std::hint::black_box;
use std::time::{Duration, Instant};

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{BatchPolicy, EinsumPlan, EinsumSpec, Exec, Layout, Scratch};

/// CPUs the calling thread may run on, from `/proc/thread-self/status`.
fn cpus() -> usize {
    let Ok(status) = std::fs::read_to_string("/proc/thread-self/status") else {
        return 0;
    };
    let Some(list) = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
    else {
        return 0;
    };
    list.split(',').fold(0, |count, part| {
        let part = part.trim();
        match part.split_once('-') {
            Some((lo, hi)) => match (lo.parse::<usize>(), hi.parse::<usize>()) {
                (Ok(lo), Ok(hi)) => count + hi.saturating_sub(lo) + 1,
                _ => count,
            },
            None => count + usize::from(part.parse::<usize>().is_ok()),
        }
    })
}

/// CPU time of the whole process in microseconds, from `/proc/self/stat`
/// (fields 14 and 15, user and system, in 10 ms clock ticks).
fn cpu_time_us() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("self stat");
    // The command name may contain spaces; the fields after it are positional.
    let fields: Vec<&str> = stat[stat.rfind(')').expect("comm") + 1..]
        .split_whitespace()
        .collect();
    let ticks: u64 =
        fields[11].parse::<u64>().expect("utime") + fields[12].parse::<u64>().expect("stime");
    ticks * 1_000_000 / 100
}

/// The BLAS provider the process linked, read from its own maps.
fn provider() -> &'static str {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return "unknown";
    };
    if maps.contains("libmkl") {
        "mkl"
    } else if maps.contains("libopenblas") {
        "openblas"
    } else if maps.contains("libblas") {
        "netlib"
    } else {
        "unknown"
    }
}

fn checksum(values: &[f64]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in values.iter().flat_map(|v| v.to_bits().to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "serial".into());
    let n: usize = args
        .next()
        .unwrap_or_else(|| "2048".into())
        .parse()
        .unwrap();
    let reps: usize = args.next().unwrap_or_else(|| "5".into()).parse().unwrap();

    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    let shape = [n, n];
    let strides = [1isize, n as isize];
    let cm = Layout::new(&shape, &strides).unwrap();
    let plan = EinsumPlan::<f64, cpueinsum_blas::Blas>::with_backend(
        cpueinsum_blas::Blas::default(),
        &spec,
        &[cm, cm],
        cm,
    )
    .unwrap();
    assert_eq!(plan.backend_steps(), 1, "the GEMM is the backend step");
    let a = vec![1.0f64; n * n];
    let b = vec![2.0f64; n * n];
    let mut out = vec![0.0f64; n * n];
    let mut scratch = Scratch::new();
    let views: Vec<StridedView<'_, f64>> = [&a, &b]
        .iter()
        .map(|x| StridedView::new(x, &shape, &strides, 0).unwrap())
        .collect();
    let mut out_view = StridedViewMut::new(&mut out, &shape, &strides, 0).unwrap();

    let mut times = Vec::with_capacity(reps);
    let mut ratio = 0.0;
    // The calling thread's mask before the first call and, for a team, still
    // inside it: the leader's mask here is what the last call left behind.
    let (before, after) = match mode.as_str() {
        "serial" => {
            let mut exec = Exec::serial();
            let before = cpus();
            for _ in 0..reps {
                let wall = Instant::now();
                let cpu = cpu_time_us();
                plan.execute_into(&mut exec, &views, &mut out_view, &mut scratch)
                    .unwrap();
                let call = wall.elapsed();
                ratio = (cpu_time_us() - cpu) as f64 / call.as_secs_f64() / 1e6;
                times.push(call);
                black_box(&out_view);
            }
            (before, cpus())
        }
        "team" => {
            let pool = threadteam::Pool::builder()
                .threads(4)
                .acquire_timeout(Duration::from_secs(10))
                .build()
                .unwrap();
            let team = pool.into_team(4).unwrap();
            let mut exec = Exec::from_token(team.exec(4), None, BatchPolicy::default());
            let before = cpus();
            for _ in 0..reps {
                let wall = Instant::now();
                let cpu = cpu_time_us();
                plan.execute_into(&mut exec, &views, &mut out_view, &mut scratch)
                    .unwrap();
                let call = wall.elapsed();
                ratio = (cpu_time_us() - cpu) as f64 / call.as_secs_f64() / 1e6;
                times.push(call);
                black_box(&out_view);
            }
            (before, cpus())
        }
        other => panic!("unknown mode {other}"),
    };
    times.sort_unstable();
    let median = times[times.len() / 2];
    println!(
        "mode={mode} provider={} n={n} reps={reps} mkl_num_threads={:?} \
         cpus_before={before} cpus_after={after} cpu_per_wall={ratio:.2} \
         median_ms={:.1} checksum={:016x}",
        provider(),
        std::env::var("MKL_NUM_THREADS").ok(),
        median.as_secs_f64() * 1e3,
        checksum(&out),
    );
}
