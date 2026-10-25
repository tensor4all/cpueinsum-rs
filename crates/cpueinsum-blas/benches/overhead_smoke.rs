//! The BLAS half of the overhead smoke test (#23 comment, case C2).
//!
//! C2's gate starts loose and is set from the slowest machine seen: the
//! packed case now runs a genuine strided `CopyPlan` packing pass (#28), so the
//! ratio is inherently above 1. Measured 1.84 on the reference EPYC host and
//! 2.29-2.59 on the GitHub-hosted runner, hence 3.4 (about 1.3x the worst);
//! tighten from CI history.
//!
//! Compares a vendor step with a contiguous operand against one whose operand
//! must be packed, so a regression in plan-time packing shows up as a ratio.
//! Run `OVERHEAD_GATE=1 cargo bench -p cpueinsum-blas --bench overhead_smoke`.

/// C2 gate (packed/contiguous), see the module docs.
const C2_GATE: f64 = 3.4;

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
        operand(&[4, 4], &[1, 4]),
        operand(&[4, 4], b_strides),
        CSpec::Absent,
        operand(&[4, 4], &[1, 4]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// The linked provider's effective worker count, or -1 where it cannot be
/// queried. OpenBLAS reads `OPENBLAS_NUM_THREADS` at initialization; `Exec::serial`
/// does not control it, so the CI gate pins it to one and this reports whether
/// that took effect.
#[cfg(target_os = "linux")]
fn provider_threads() -> i32 {
    extern "C" {
        fn openblas_get_num_threads() -> i32;
    }
    // SAFETY: the linked provider exports this query, which only reads a counter.
    unsafe { openblas_get_num_threads() }
}

#[cfg(not(target_os = "linux"))]
fn provider_threads() -> i32 {
    -1
}

fn c2_ns(b_strides: &[isize], expect_packing: bool) -> f64 {
    let plan = BinaryPlan::<f64>::new(&problem(b_strides), &PlanConfig::default()).unwrap();
    assert_eq!(plan.work_len() > 0, expect_packing, "work_len");
    let a: Vec<f64> = (0..16).map(|i| (i % 7) as f64).collect();
    // B strides [2, 5] are not BLAS-readable in either orientation; its span is
    // 21, so 22 elements.
    let b: Vec<f64> = (0..22).map(|i| (i % 5) as f64).collect();
    let mut d = vec![0.0; 16];
    let mut work = vec![0.0; plan.work_len()];
    let mut exec = Exec::serial();
    let route = plan
        .execute_slices_accum(
            &mut exec,
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
            &mut exec,
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
    let threads = provider_threads();
    println!(
        "C2  provider threads         {threads:>10}  OPENBLAS_NUM_THREADS={:?}",
        std::env::var("OPENBLAS_NUM_THREADS").ok()
    );
    let mut passed = false;
    let mut last_ratio = f64::INFINITY;
    for _ in 0..ATTEMPTS {
        let contiguous = c2_ns(&[1, 4], false);
        let packed = c2_ns(&[2, 5], true);
        let ratio = packed / contiguous.max(1.0);
        last_ratio = ratio;
        println!(
            "C2  packed/contiguous      {:>10.1}  ratio {ratio:.3} <= {C2_GATE} {}",
            packed,
            if ratio <= C2_GATE { "ok" } else { "FAIL" }
        );
        if ratio <= C2_GATE {
            passed = true;
            break;
        }
    }
    if gate && !passed {
        eprintln!("C2 failed: last ratio {last_ratio:.3}");
        std::process::exit(1);
    }
}
