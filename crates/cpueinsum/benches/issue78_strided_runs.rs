//! Issue #78 diagnostic: one strided run of `J` small f64 jobs, executed per job
//! or as one batched child. Protocol and results: the 2026-11-23 worklog.
//!
//! Usage: `issue78_strided_runs <routes> <threads> <reps>`, where `<routes>` is
//! `per-job` or `per-job,batched`. Every case runs its routes interleaved,
//! `<reps>` times, each sample the mean of enough calls to last about 2 ms, and
//! prints one CSV row per (case, route): median and quartiles of the per-call
//! time and the heap allocation requests of one call.
//!
//! To time the per-job route of a commit before #78, replace `build`'s body with
//! `GroupedPlan::<f64>::new(jobs, &PlanConfig::default()).unwrap()` and run
//! `per-job` alone; the rest of the file uses only the pre-#78 API.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{ArenaProvider, Exec, GroupedGemmJob, GroupedPlan};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer came from this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn build(jobs: &[GroupedGemmJob], batched: bool) -> GroupedPlan<f64> {
    let options = cpueinsum::GroupedOptions::default().with_strided_runs(batched);
    let plan = GroupedPlan::<f64>::new_with_options(
        jobs,
        Default::default(),
        &PlanConfig::default(),
        options,
    )
    .unwrap();
    assert_eq!(plan.strided_runs().len(), usize::from(batched));
    plan
}

fn quartiles(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| {
        let x = q * (samples.len() - 1) as f64;
        let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
        samples[lo] + (samples[hi] - samples[lo]) * (x - lo as f64)
    };
    (at(0.25), at(0.5), at(0.75))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() >= 4, "routes threads reps");
    let routes: Vec<bool> = args[1]
        .split(',')
        .map(|r| match r {
            "per-job" => false,
            "batched" => true,
            _ => panic!("unknown route {r}"),
        })
        .collect();
    let threads: usize = args[2].parse().unwrap();
    let reps: usize = args[3].parse().unwrap();

    let pool = threadteam::Pool::builder()
        .threads(threads)
        .pin_participants(false)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let team = pool
        .into_team(threads)
        .expect("dedicated idle pool must admit team");
    let arena = ArenaProvider::new();

    println!("jobs,m,k,n,threads,route,median_ns,q1_ns,q3_ns,iqr_ns,allocs_per_call,calls_per_sample,reps,route_reported,build");
    for count in [64usize, 256] {
        for size in [2usize, 4, 8, 16, 32] {
            let (m, k, n) = (size, size, size);
            let jobs: Vec<_> = (0..count)
                .map(|i| GroupedGemmJob::new(i * m * k, i * k * n, i * m * n, m, k, n))
                .collect();
            let a: Vec<f64> = (0..count * m * k).map(|i| (i % 5) as f64 - 2.0).collect();
            let b: Vec<f64> = (0..count * k * n).map(|i| (i % 3) as f64 - 1.0).collect();
            let plans: Vec<_> = routes.iter().map(|&r| build(&jobs, r)).collect();
            let mut outs: Vec<Vec<f64>> = routes.iter().map(|_| vec![0.0; count * m * n]).collect();
            let mut exec = Exec::from_token(team.exec(threads), Some(&arena), Default::default());
            let mut once = |p: usize, out: &mut [f64]| {
                plans[p]
                    .execute_into(&mut exec, 1.0, black_box(&a), black_box(&b), black_box(out))
                    .unwrap()
            };
            // Warm, check the routes agree, and size one sample to about 2 ms.
            let mut reported = Vec::new();
            for p in 0..plans.len() {
                for _ in 0..50 {
                    once(p, &mut outs[p]);
                }
                reported.push(once(p, &mut outs[p]));
            }
            for out in &outs[1..] {
                assert_eq!(out, &outs[0], "routes disagree");
            }
            let start = Instant::now();
            once(0, &mut outs[0]);
            let one = start.elapsed().as_nanos().max(1) as usize;
            let calls = (2_000_000 / one).clamp(1, 10_000);
            let mut allocs = Vec::new();
            for p in 0..plans.len() {
                let before = ALLOCS.load(Ordering::Relaxed);
                for _ in 0..100 {
                    once(p, &mut outs[p]);
                }
                allocs.push((ALLOCS.load(Ordering::Relaxed) - before) as f64 / 100.0);
            }
            let mut samples: Vec<Vec<f64>> = routes.iter().map(|_| Vec::new()).collect();
            for rep in 0..reps {
                // Alternate which route goes first so drift does not favour one.
                for i in 0..plans.len() {
                    let p = if rep % 2 == 0 { i } else { plans.len() - 1 - i };
                    let start = Instant::now();
                    for _ in 0..calls {
                        once(p, &mut outs[p]);
                    }
                    samples[p].push(start.elapsed().as_nanos() as f64 / calls as f64);
                }
            }
            for (p, &batched) in routes.iter().enumerate() {
                let (q1, med, q3) = quartiles(&mut samples[p]);
                println!(
                    "{count},{m},{k},{n},{threads},{},{med:.0},{q1:.0},{q3:.0},{:.0},{},{calls},{reps},{},{}",
                    if batched { "batched" } else { "per-job" },
                    q3 - q1,
                    allocs[p],
                    format!("{:?}", reported[p]).replace(',', ";"),
                    option_env!("ISSUE78_BUILD").unwrap_or("unlabelled"),
                );
            }
        }
    }
}
