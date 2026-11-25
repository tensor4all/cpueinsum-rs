//! The per-job route's work-weighted placement under `Auto` (cpueinsum-rs#80):
//! a dominant job gets a wide lane, and every output is bit-identical to the
//! serial per-job route, because a lane only changes which threads run a job,
//! never the job's own accumulation order or another job's data.

mod common;

use common::pool_with_reference_costs;
use core::mem::MaybeUninit;
use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan, GroupedRoute, Scalar};
use num_complex::Complex64;

/// One dominant `192^3` job and fifteen `16^3` jobs, in that order, packed
/// back to back in every buffer: the `gm-dom` shape of #80, scaled down.
fn skewed() -> Vec<GroupedGemmJob> {
    let mut sizes = vec![192usize];
    sizes.extend([16; 15]);
    let (mut a, mut b, mut d) = (0, 0, 0);
    sizes
        .iter()
        .map(|&n| {
            let job = GroupedGemmJob::new(a, b, d, n, n, n);
            (a, b, d) = (a + n * n, b + n * n, d + n * n);
            job
        })
        .collect()
}

fn data<T: Scalar>(len: usize, seed: usize, of: impl Fn(f64, f64) -> T) -> Vec<T> {
    (0..len)
        .map(|i| {
            let x = ((i * 37 + seed) % 101) as f64 / 17.0 - 2.9;
            of(x, ((i * 53 + seed) % 89) as f64 / 13.0 - 3.1)
        })
        .collect()
}

/// The pool width of these tests.
const THREADS: usize = 8;

/// The widest thread count a route reports.
fn widest(route: GroupedRoute) -> usize {
    match route {
        GroupedRoute::Placed { widest, .. } => widest,
        GroupedRoute::Split { item_threads, .. } => item_threads,
        GroupedRoute::Inner { threads } => threads,
        _ => 1,
    }
}

fn bitwise_against_serial<T: Scalar + PartialEq + core::fmt::Debug>(
    config: PlanConfig,
    of: impl Fn(f64, f64) -> T + Copy,
    alpha: T,
    beta: T,
) {
    let jobs = skewed();
    let plan = GroupedPlan::<T>::new(&jobs, &config).unwrap();
    let lhs = data(plan.required_lhs_len(), 1, of);
    let rhs = data(plan.required_rhs_len(), 2, of);
    let initial = data(plan.required_out_len(), 3, of);

    // The references: each entry point on today's per-job route, serially.
    // Each is compared with itself: the fresh and the initialized contracts
    // need not agree bit for bit with each other (a complex child's fresh
    // output differs from its overwrite in the last bit on one thread too).
    let run = |exec: &mut Exec<'_>, entry: usize| {
        let mut out = initial.clone();
        let route = match entry {
            0 => plan.execute_into(exec, alpha, &lhs, &rhs, &mut out),
            1 => plan.execute_into_accum(exec, alpha, &lhs, &rhs, beta, &mut out),
            _ => {
                let mut fresh = vec![MaybeUninit::<T>::uninit(); out.len()];
                let (written, route) = plan
                    .execute_uninit(exec, alpha, &lhs, &rhs, T::default(), &mut fresh)
                    .unwrap();
                out.copy_from_slice(written);
                Ok(route)
            }
        };
        (out, route.unwrap())
    };
    let thread_pool = pool_with_reference_costs(THREADS);
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let mut exec = Exec::new(&pool, THREADS).unwrap();
    let mut routes = Vec::new();
    for entry in 0..3 {
        let (reference, route) = run(&mut Exec::serial(), entry);
        assert_eq!(route, GroupedRoute::Serial);
        for _ in 0..2 {
            let (out, route) = run(&mut exec, entry);
            assert!(out == reference, "entry {entry} on {route:?}");
            routes.push(route);
        }
    }
    // Deterministic for a fixed job list and context, and the dominant job
    // got more than one thread (the historical route ran it on one).
    assert!(routes.iter().all(|&r| r == routes[0]), "{routes:?}");
    assert!(widest(routes[0]) > 1, "{routes:?}");
}

#[test]
fn placed_f64_outputs_are_bitwise_the_serial_per_job_route() {
    // The packed child: under the default config this host's f64 Faer child
    // keeps one thread at every lane width (tcontract's own decision), so the
    // placement has nothing to widen.
    bitwise_against_serial(PlanConfig::packed(), |x, _| x, 0.75, -1.25);
}

#[test]
fn placed_c64_outputs_are_bitwise_the_serial_per_job_route() {
    bitwise_against_serial(
        PlanConfig::default(),
        Complex64::new,
        Complex64::new(0.75, -0.5),
        Complex64::new(-1.25, 0.25),
    );
}
