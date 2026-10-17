//! The BLAS half of the overhead smoke test (#23 comment, case C2).
//!
//! Compares a vendor step with a contiguous operand against one whose operand
//! must be packed, so a regression in plan-time packing shows up as a ratio.
//! Run `OVERHEAD_GATE=1 cargo bench -p cpueinsum-blas --bench overhead_smoke`.

extern crate blas_src as _;

use std::time::Instant;

use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::Exec;
use cpueinsum_blas::{BinaryPlan, ExecutionRoute};

const ROUNDS: usize = 5;
const ATTEMPTS: usize = 3;

fn time_ns(iterations: usize, mut body: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..iterations {
            body();
        }
        best = best.min(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    best
}

fn operand(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

fn problem(b_strides: &[isize]) -> Problem {
    Problem::from_labels(
        DType::F64,
        operand(&[16, 16], &[1, 16]),
        operand(&[16, 16], b_strides),
        CSpec::Absent,
        operand(&[16, 16], &[1, 16]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

fn c2_ns(b_strides: &[isize]) -> f64 {
    let plan = BinaryPlan::<f64>::new(&problem(b_strides), &PlanConfig::default()).unwrap();
    let a: Vec<f64> = (0..256).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..256).map(|i| (i % 5) as f64).collect();
    let mut d = vec![0.0; 256];
    let mut work = vec![0.0; plan.work_len()];
    let exec = Exec::serial();
    let route = plan
        .execute_slices_accum(
            &exec,
            1.0,
            (&a, 0),
            (&b, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut work,
        )
        .unwrap();
    assert_eq!(route, ExecutionRoute::Vendor);
    time_ns(2000, || {
        plan.execute_slices_accum(
            &exec,
            1.0,
            (&a, 0),
            (&b, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut work,
        )
        .unwrap();
    })
}

fn main() {
    let gate = std::env::var("OVERHEAD_GATE").is_ok_and(|value| value == "1");
    let mut passed = false;
    let mut last_ratio = f64::INFINITY;
    for _ in 0..ATTEMPTS {
        let contiguous = c2_ns(&[1, 16]);
        let packed = c2_ns(&[16, 1]);
        let ratio = packed / contiguous.max(1.0);
        last_ratio = ratio;
        println!(
            "C2  packed/contiguous      {:>10.1}  ratio {ratio:.3} <= 2.0 {}",
            packed,
            if ratio <= 2.0 { "ok" } else { "FAIL" }
        );
        if ratio <= 2.0 {
            passed = true;
            break;
        }
    }
    if gate && !passed {
        eprintln!("C2 failed: last ratio {last_ratio:.3}");
        std::process::exit(1);
    }
}
