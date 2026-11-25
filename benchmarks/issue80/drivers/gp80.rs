//! cpueinsum-rs#80 benchmark driver (not committed): GroupedPlan under `Auto`,
//! timed per call including the plan construction from cached children, as the
//! #80 table was. Usage: gp80 <workloads.txt> <threads> <prime_ms> <reps>
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpueinsum::tcontract::{Plan, PlanConfig};
use cpueinsum::{ChildKey, Exec, GroupedGemmJob, GroupedPlan, Scalar};
use num_complex::Complex64;

struct W { name: String, dtype: String, alpha: [f64; 2], beta: [f64; 2], jobs: Vec<[usize; 6]> }

fn parse(line: &str) -> W {
    let f: Vec<&str> = line.split('|').collect();
    let pair = |s: &str| { let v: Vec<f64> = s.split(' ').map(|x| x.parse().unwrap()).collect(); [v[0], v[1]] };
    let jobs = f[4].split(';').map(|j| { let v: Vec<usize> = j.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2], v[3], v[4], v[5]] }).collect();
    W { name: f[0].into(), dtype: f[1].into(), alpha: pair(f[2]), beta: pair(f[3]), jobs }
}

fn run<T: Scalar + Copy + PartialEq>(w: &W, of: fn(f64, f64) -> T, exec: &mut Exec<'_>, prime: Duration, reps: usize) -> (Vec<f64>, String) {
    let jobs: Vec<GroupedGemmJob> = w.jobs.iter().map(|j| GroupedGemmJob::new(j[0], j[1], j[2], j[3], j[4], j[5])).collect();
    let mut cache: HashMap<ChildKey, Arc<Plan<T>>> = HashMap::new();
    let mut build = |cache: &mut HashMap<ChildKey, Arc<Plan<T>>>| {
        GroupedPlan::<T>::from_prepared_with(GroupedPlan::<T>::prepare_jobs(&jobs).unwrap(), Default::default(), &PlanConfig::default(), |key| {
            if let Some(p) = cache.get(key) { return Ok(Arc::clone(p)); }
            let p = Arc::new(key.build::<T>()?); cache.insert(key.clone(), Arc::clone(&p)); Ok(p)
        }).unwrap()
    };
    let probe = build(&mut cache);
    let val = |i: usize, s: usize| of(((i * 37 + s) % 101) as f64 / 101.0 - 0.5, ((i * 53 + s) % 89) as f64 / 89.0 - 0.5);
    let lhs: Vec<T> = (0..probe.required_lhs_len()).map(|i| val(i, 1)).collect();
    let rhs: Vec<T> = (0..probe.required_rhs_len()).map(|i| val(i, 2)).collect();
    let mut out: Vec<T> = (0..probe.required_out_len()).map(|i| val(i, 3)).collect();
    let (alpha, beta) = (of(w.alpha[0], w.alpha[1]), of(w.beta[0], w.beta[1]));
    let mut call = |cache: &mut HashMap<ChildKey, Arc<Plan<T>>>, out: &mut [T]| {
        let plan = build(cache);
        plan.execute_into_accum(exec, alpha, &lhs, &rhs, beta, out).unwrap()
    };
    // Checked cell: one call from the initial output against the serial route.
    let initial = out.clone();
    let mut reference = initial.clone();
    probe.execute_into_accum(&mut Exec::serial(), alpha, &lhs, &rhs, beta, &mut reference).unwrap();
    let mut checked = initial.clone();
    call(&mut cache, &mut checked);
    let bitwise = checked == reference; // `==` on values: equal up to the sign of zero
    let start = Instant::now();
    let mut route = call(&mut cache, &mut out);
    while start.elapsed() < prime { route = call(&mut cache, &mut out); }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        route = call(&mut cache, &mut out);
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    (times, format!("{route:?}\tequal={bitwise}"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a[2].parse().unwrap();
    let prime = Duration::from_millis(a[3].parse().unwrap());
    let reps: usize = a[4].parse().unwrap();
    let tp = threadteam::Pool::builder().threads(threads).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&tp);
    let mut exec = Exec::new(&pool, threads).unwrap();
    assert_eq!(exec.budget(), threads, "budget clamped");
    for line in std::fs::read_to_string(&a[1]).unwrap().lines() {
        let w = parse(line);
        let (mut t, route) = match w.dtype.as_str() {
            "F64" => run::<f64>(&w, |x, _| x, &mut exec, prime, reps),
            _ => run::<Complex64>(&w, Complex64::new, &mut exec, prime, reps),
        };
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{}", w.name, w.dtype, threads, t[t.len() / 2], t[0], t[t.len() - 1], route);
    }
}
