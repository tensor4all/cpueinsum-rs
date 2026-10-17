//! A cheap overhead smoke test (#23 comment): ratios measured in one run, not
//! absolute times.
//!
//! Run `OVERHEAD_GATE=1 cargo bench -p cpueinsum --bench overhead_smoke` (or
//! `cargo bench --bench overhead_smoke`) to enforce the gates; without the
//! variable every case just reports ns/op. Sizes are deliberately tiny so the
//! whole binary stays under twenty seconds in release, and a gate fails only
//! when all of its attempts exceed the threshold.

use std::mem::MaybeUninit;
use std::time::Instant;

use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use cpueinsum::tcontract::{OutputContract, PlanConfig, SliceAccumulationSource};
use cpueinsum::{BinaryPlan, Exec, GroupedGemmJob, GroupedPlan, Workspace};

const ROUNDS: usize = 5;
const ATTEMPTS: usize = 3;

fn time_ns(iterations: usize, mut body: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..iterations {
            body();
        }
        let elapsed = start.elapsed();
        best = best.min(elapsed.as_nanos() as f64 / iterations as f64);
    }
    best
}

fn operand(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

/// C1: a native accumulate with beta = 0.5 against the fresh-output path. The
/// gate guards the strided update pass that used to replan per call.
fn c1_ratio(n: usize) -> f64 {
    let problem = Problem::from_labels(
        DType::F64,
        operand(&[n, n], &[1, n as isize]),
        operand(&[n, n], &[1, n as isize]),
        CSpec::Output(Op::Identity),
        operand(&[n, n], &[1, n as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::packed()).unwrap();
    let a: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i % 5) as f64).collect();
    let mut ws =
        Workspace::with_req(&plan.native_workspace_req(&Exec::serial(), OutputContract::Fresh));
    ws.reserve(&plan.native_workspace_req(&Exec::serial(), OutputContract::Initialized));
    let exec = Exec::serial_with_workspace(ws.as_provider());
    let mut accum = vec![1.0; n * n];
    let mut fresh = vec![MaybeUninit::<f64>::uninit(); n * n];
    // Warm both routes and the workspace.
    plan.execute_slices_accum(
        &exec,
        1.0,
        (&a, 0),
        (&b, 0),
        0.5,
        SliceAccumulationSource::Output,
        (&mut accum, 0),
    )
    .unwrap();
    let _ = plan
        .execute_uninit_slices(
            &exec,
            1.0,
            (&a, 0),
            (&b, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut fresh, 0),
        )
        .unwrap();
    let iterations = (1_000_000 / (n * n)).max(20);
    let (mut accum_ns, mut fresh_ns) = (0.0, 0.0);
    for _ in 0..ATTEMPTS {
        accum_ns = time_ns(iterations, || {
            plan.execute_slices_accum(
                &exec,
                1.0,
                (&a, 0),
                (&b, 0),
                0.5,
                SliceAccumulationSource::Output,
                (&mut accum, 0),
            )
            .unwrap();
        });
        fresh_ns = time_ns(iterations, || {
            let _ = plan
                .execute_uninit_slices(
                    &exec,
                    1.0,
                    (&a, 0),
                    (&b, 0),
                    0.0,
                    SliceAccumulationSource::Absent,
                    (&mut fresh, 0),
                )
                .unwrap();
        });
        let ratio = accum_ns / fresh_ns.max(1.0);
        if ratio <= 1.5 {
            return ratio;
        }
    }
    accum_ns / fresh_ns.max(1.0)
}

/// C3: prepared binary plan reuse, ns per step, reported only.
fn c3_ns(n: usize) -> f64 {
    let problem = Problem::from_labels(
        DType::F64,
        operand(&[n, n], &[1, n as isize]),
        operand(&[n, n], &[1, n as isize]),
        CSpec::Absent,
        operand(&[n, n], &[1, n as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let a: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i % 5) as f64).collect();
    let mut d = vec![0.0; n * n];
    let exec = Exec::serial();
    let iterations = (1_000_000 / (n * n)).max(20);
    let _ = time_ns(10, || {
        plan.execute_slices(&exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
            .unwrap();
    });
    time_ns(iterations, || {
        plan.execute_slices(&exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
            .unwrap();
    })
}

/// C4: grouped path against the binary path on a two-operand spec, reported only.
fn c4_ns(n: usize) -> (f64, f64) {
    let binary_problem = Problem::from_labels(
        DType::F64,
        operand(&[n, n], &[1, n as isize]),
        operand(&[n, n], &[1, n as isize]),
        CSpec::Absent,
        operand(&[n, n], &[1, n as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let binary = BinaryPlan::<f64>::new(&binary_problem, &PlanConfig::default()).unwrap();
    let a: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i % 5) as f64).collect();
    let mut d = vec![0.0; n * n];
    let exec = Exec::serial();
    let grouped = GroupedPlan::<f64>::new(
        &[GroupedGemmJob::new(0, 0, 0, n, n, n)],
        &PlanConfig::default(),
    )
    .unwrap();
    let binary_ns = time_ns(200, || {
        binary
            .execute_slices(&exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
            .unwrap();
    });
    let grouped_ns = time_ns(200, || {
        grouped.execute_into(&exec, 1.0, &a, &b, &mut d).unwrap();
    });
    (binary_ns, grouped_ns)
}

fn main() {
    let gate = std::env::var("OVERHEAD_GATE").is_ok_and(|value| value == "1");
    let mut failed = false;

    println!("id  case                    ns/op       gate");
    for n in [2usize, 4, 16] {
        let ratio = c1_ratio(n);
        let ok = ratio <= 1.5;
        println!(
            "C1  accumulate/uninit n={n:<2}  {:>10.1}  ratio {ratio:.3} <= 1.5 {}",
            ratio,
            if ok { "ok" } else { "FAIL" }
        );
        failed |= gate && !ok;
    }
    for n in [2usize, 4] {
        println!("C3  prepared binary n={n:<2}   {:>10.1}  report", c3_ns(n));
    }
    for n in [8usize, 16] {
        let (binary_ns, grouped_ns) = c4_ns(n);
        println!("C4  binary n={n:<2}            {binary_ns:>10.1}  report");
        println!(
            "C4  grouped n={n:<2}           {grouped_ns:>10.1}  report (ratio {:.3})",
            grouped_ns / binary_ns.max(1.0)
        );
    }

    if gate && failed {
        std::process::exit(1);
    }
}
