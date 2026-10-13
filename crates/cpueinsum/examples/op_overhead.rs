//! What the view wrapper costs a host per call: the same warm binary plan
//! executed through `StridedView`s built per call, against `execute_slices`.
//!
//! ```text
//! taskset -c 2 cargo run --release -p cpueinsum --example op_overhead
//! ```

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{ArenaProvider, BinaryPlan, Exec};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 500_000;

/// `D[i, l] = sum_j A[i, j] B[j, l]` over column-major 2x2, as the MPS-overlap
/// step that made the view cost visible in the first place.
fn problem() -> Problem {
    let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_labels(
        DType::F64,
        l(&[2, 2], &[1, 2]),
        l(&[2, 2], &[1, 2]),
        CSpec::Absent,
        l(&[2, 2], &[1, 2]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

fn main() {
    let dims = [2usize, 2];
    let strides = [1isize, 2];
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [1.0f64, 0.0, 0.0, 1.0];
    let mut c = [0.0f64; 4];

    let arena = ArenaProvider::new();
    let exec = Exec::serial_with_workspace(&arena); // Explicit 1T backend.
    let plan = BinaryPlan::<f64>::new(&problem(), &PlanConfig::default()).unwrap();

    // Prime the arena so neither arm pays for its scratch.
    plan.execute_slices(&exec, 1.0, (&a, 0), (&b, 0), (&mut c, 0))
        .unwrap();

    let ns = |t: Instant| t.elapsed().as_nanos() as f64 / N as f64;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..N {
            let av = StridedView::<f64>::new(black_box(&a), &dims, &strides, 0).unwrap();
            let bv = StridedView::<f64>::new(black_box(&b), &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(black_box(&mut c), &dims, &strides, 0).unwrap();
            plan.execute_into(&exec, 1.0, &av, &bv, &mut dv).unwrap();
            black_box(&c);
        }
        let views = ns(t);

        let t = Instant::now();
        for _ in 0..N {
            plan.execute_slices(&exec, 1.0, (black_box(&a), 0), (black_box(&b), 0), (black_box(&mut c), 0))
                .unwrap();
            black_box(&c);
        }
        let slices = ns(t);

        // The wrapper alone: three views, no execution.
        let t = Instant::now();
        for _ in 0..N {
            black_box(StridedView::<f64>::new(black_box(&a), &dims, &strides, 0).unwrap());
            black_box(StridedView::<f64>::new(black_box(&b), &dims, &strides, 0).unwrap());
            black_box(StridedViewMut::new(black_box(&mut c), &dims, &strides, 0).unwrap());
        }
        let wrapper = ns(t);

        println!(
            "views {:6.1} ns/call   slices {:6.1} ns/call   delta {:6.1} ns   (3 views alone {:6.1} ns)",
            views,
            slices,
            views - slices,
            wrapper
        );
    }
    assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
}
