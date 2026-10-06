//! [`BinaryPlan`](cpueinsum::BinaryPlan) against the naive reference, its
//! typed error contract, uninitialized output and reusable arenas.

mod common;

use core::mem::MaybeUninit;

use common::{close, naive, Operand};
use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tprims_contract::api::{
    AccumulationSource, CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem,
};
use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::{BinaryPlan, Error, Exec};
use num_complex::Complex64;
use tprims_exec::ArenaProvider;

/// A layout spec with an explicit element offset.
fn spec(dims: &[usize], strides: &[isize], offset: isize) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, offset).unwrap())
}

/// `D[i, j] = sum_k A[i, k] B[k, j]` over column-major 2x3 * 3x2 -> 2x2, with a
/// caller-chosen C mode.
fn matmul(c: CSpec) -> Problem {
    problem(c, spec(&[2, 3], &[1, 2], 0), spec(&[3, 2], &[1, 3], 0))
}

/// `D[i, j] = sum_k A[i, k] B[k, j]` over column-major 2x2 * 2x2 -> 2x2.
fn identity_matmul(c: CSpec) -> Problem {
    problem(c, spec(&[2, 2], &[1, 2], 0), spec(&[2, 2], &[1, 2], 0))
}

fn problem(c: CSpec, a: OperandSpec, b: OperandSpec) -> Problem {
    let labels = Labels::new(&[0, 2], &[2, 1], &[0, 1]);
    let labels = match c {
        CSpec::Separate(_) => labels.with_c(&[0, 1]),
        _ => labels,
    };
    Problem::from_labels(DType::F64, a, b, c, spec(&[2, 2], &[1, 2], 0), &labels).unwrap()
}

/// A = [[1, 2, 3], [4, 5, 6]] and B = [[1, 0], [0, 1], [1, 1]], both
/// column-major; identity-B variants let tests reason about A directly.
const A_COL: [f64; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
const B_COL: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];

fn reference() -> Vec<f64> {
    naive(
        &[
            Operand::new(A_COL.to_vec(), &[2, 3], &[1, 2], 0).labelled(&[0, 2]),
            Operand::new(B_COL.to_vec(), &[3, 2], &[1, 3], 0).labelled(&[2, 1]),
        ],
        &[0, 1],
        &[2, 2],
    )
}

#[test]
fn f64_modes_match_the_reference() {
    let expected = reference();
    let exec = Exec::serial();

    // Absent: plain overwrite.
    let plan = BinaryPlan::<f64>::new(&matmul(CSpec::Absent), &PlanConfig::default()).unwrap();
    let mut d = [0.0; 4];
    let _route = plan
        .execute_slices(&exec, 1.0, (&A_COL, 0), (&B_COL, 0), (&mut d, 0))
        .unwrap();
    close(&d, &expected, 1e-12);

    // Output: read the previous D in place.
    let plan = BinaryPlan::<f64>::new(&matmul(CSpec::Output(Op::Identity)), &PlanConfig::default())
        .unwrap();
    let old = [1.0, -2.0, 0.5, 3.0];
    let mut d = old;
    let _route = plan
        .execute_slices_accum(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            2.0,
            SliceAccumulationSource::Output,
            (&mut d, 0),
        )
        .unwrap();
    let expected_output: Vec<f64> = expected
        .iter()
        .zip(old)
        .map(|(&e, o)| e + 2.0 * o)
        .collect();
    close(&d, &expected_output, 1e-12);

    // Separate: read a distinct C slice.
    let plan = BinaryPlan::<f64>::new(
        &matmul(CSpec::Separate(spec(&[2, 2], &[1, 2], 0))),
        &PlanConfig::default(),
    )
    .unwrap();
    let c = [0.5, 0.25, -1.0, 3.0];
    let mut d = [0.0; 4];
    let _route = plan
        .execute_slices_accum(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            3.0,
            SliceAccumulationSource::Separate((&c, 0)),
            (&mut d, 0),
        )
        .unwrap();
    let expected_separate: Vec<f64> = expected.iter().zip(c).map(|(&e, x)| e + 3.0 * x).collect();
    close(&d, &expected_separate, 1e-12);
}

#[test]
fn complex_conjugated_view() {
    // D[i, j] = sum_k conj(A[i, k]) B[k, j], with B = I so D = conj(A).
    let a = [
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
    ];
    let b = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let expected = [
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
    ];
    let problem = Problem::from_labels(
        DType::C64,
        spec(&[2, 2], &[1, 2], 0).conj(),
        spec(&[2, 2], &[1, 2], 0),
        CSpec::Absent,
        spec(&[2, 2], &[1, 2], 0),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let plan = BinaryPlan::<Complex64>::new(&problem, &PlanConfig::default()).unwrap();

    let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
    let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
    let mut d = [Complex64::new(0.0, 0.0); 4];
    let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
    let _route = plan
        .execute_into(&Exec::serial(), Complex64::new(1.0, 0.0), &av, &bv, &mut dv)
        .unwrap();
    close(&d, &expected, 1e-12);
}

#[test]
fn alpha_zero_and_negative_beta() {
    let exec = Exec::serial();
    let plan = BinaryPlan::<f64>::new(&matmul(CSpec::Absent), &PlanConfig::default()).unwrap();
    let mut d = [7.0; 4];
    let _route = plan
        .execute_slices(&exec, 0.0, (&A_COL, 0), (&B_COL, 0), (&mut d, 0))
        .unwrap();
    // alpha zero reads no A/B value and overwrites D with zero.
    assert_eq!(d, [0.0; 4]);

    let out = BinaryPlan::<f64>::new(&matmul(CSpec::Output(Op::Identity)), &PlanConfig::default())
        .unwrap();
    let old = [1.0, -2.0, 0.5, 3.0];
    let mut d = old;
    let _route = out
        .execute_slices_accum(
            &exec,
            0.0,
            (&A_COL, 0),
            (&B_COL, 0),
            -3.0,
            SliceAccumulationSource::Output,
            (&mut d, 0),
        )
        .unwrap();
    let expected: Vec<f64> = old.iter().map(|&o| -3.0 * o).collect();
    close(&d, &expected, 1e-12);
}

#[test]
fn complex_alpha_beta() {
    // D[i, j] = alpha * A[i, j] + beta * old_D[i, j] with B = I.
    let a = [
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(3.0, 2.0),
    ];
    let b = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let problem = Problem::from_labels(
        DType::C64,
        spec(&[2, 2], &[1, 2], 0),
        spec(&[2, 2], &[1, 2], 0),
        CSpec::Output(Op::Identity),
        spec(&[2, 2], &[1, 2], 0),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let plan = BinaryPlan::<Complex64>::new(&problem, &PlanConfig::default()).unwrap();

    let alpha = Complex64::new(0.0, 1.0);
    let beta = Complex64::new(-1.0, 0.5);
    let old = [
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 3.0),
        Complex64::new(1.0, -1.0),
    ];
    let mut d = old;
    let _route = plan
        .execute_slices_accum(
            &Exec::serial(),
            alpha,
            (&a, 0),
            (&b, 0),
            beta,
            SliceAccumulationSource::Output,
            (&mut d, 0),
        )
        .unwrap();
    let expected: Vec<Complex64> = a
        .iter()
        .zip(old)
        .map(|(&x, o)| alpha * x + beta * o)
        .collect();
    close(&d, &expected, 1e-12);
}

#[test]
fn negative_strides_and_origin() {
    // A = [[1, 2], [3, 4]] stored at strides [1, -2] with the logical origin at
    // physical index 2: [A[0,1], A[1,1], A[0,0], A[1,0]] = [2, 4, 1, 3].
    let a_buf = [2.0, 4.0, 1.0, 3.0];
    let b = [1.0, 0.0, 0.0, 1.0];
    let expected = naive(
        &[
            Operand::new(a_buf.to_vec(), &[2, 2], &[1, -2], 2).labelled(&[0, 2]),
            Operand::new(b.to_vec(), &[2, 2], &[1, 2], 0).labelled(&[2, 1]),
        ],
        &[0, 1],
        &[2, 2],
    );
    assert_eq!(expected, [1.0, 3.0, 2.0, 4.0]);

    let problem = Problem::from_labels(
        DType::F64,
        spec(&[2, 2], &[1, -2], 0),
        spec(&[2, 2], &[1, 2], 0),
        CSpec::Absent,
        spec(&[2, 2], &[1, 2], 0),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let mut d = [0.0; 4];
    let _route = plan
        .execute_slices(&Exec::serial(), 1.0, (&a_buf, 2), (&b, 0), (&mut d, 0))
        .unwrap();
    close(&d, &expected, 1e-12);
}

#[test]
fn uninit_output_and_rejections() {
    let expected = reference();
    let exec = Exec::serial();
    let plan = BinaryPlan::<f64>::new(&matmul(CSpec::Absent), &PlanConfig::default()).unwrap();

    // Dense column-major coverage initializes every slot on success.
    let mut d = [MaybeUninit::<f64>::uninit(); 4];
    let (initialized, _route) = plan
        .execute_uninit_slices(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
        )
        .unwrap();
    close(initialized, &expected, 1e-12);

    // Padding is not exact coverage: rejected before a byte is written.
    let mut padded = [MaybeUninit::<f64>::uninit(); 5];
    let err = plan
        .execute_uninit_slices(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut padded, 0),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Contract { step: 0, .. }));

    // Output-mode accumulation from fresh storage is rejected.
    let out = BinaryPlan::<f64>::new(&matmul(CSpec::Output(Op::Identity)), &PlanConfig::default())
        .unwrap();
    let mut d = [MaybeUninit::<f64>::uninit(); 4];
    let err = out
        .execute_uninit_slices(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            1.0,
            SliceAccumulationSource::Output,
            (&mut d, 0),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Contract { step: 0, .. }));
}

#[test]
fn errors_leave_d_unchanged() {
    let exec = Exec::serial();
    let plan = BinaryPlan::<f64>::new(&matmul(CSpec::Absent), &PlanConfig::default()).unwrap();
    let sentinel = [9.0; 4];

    // Nonzero beta with no accumulation source.
    let mut d = sentinel;
    let err = plan
        .execute_slices_accum(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            1.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Contract { step: 0, .. }));
    assert_eq!(d, sentinel);

    // NaN beta is nonzero for this check as well.
    let mut d = sentinel;
    let err = plan
        .execute_slices_accum(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            f64::NAN,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Contract { step: 0, .. }));
    assert_eq!(d, sentinel);

    // A slice too short for the planned A range.
    let mut d = sentinel;
    let err = plan
        .execute_slices(&exec, 1.0, (&A_COL[..3], 0), (&B_COL, 0), (&mut d, 0))
        .unwrap_err();
    assert!(matches!(err, Error::Contract { step: 0, .. }));
    assert_eq!(d, sentinel);

    // A source that does not match the prepared C mode.
    let out = BinaryPlan::<f64>::new(&matmul(CSpec::Output(Op::Identity)), &PlanConfig::default())
        .unwrap();
    let cv = [1.0; 4];
    let mut d = sentinel;
    let err = out
        .execute_slices_accum(
            &exec,
            1.0,
            (&A_COL, 0),
            (&B_COL, 0),
            2.0,
            SliceAccumulationSource::Separate((&cv, 0)),
            (&mut d, 0),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Contract { step: 0, .. }));
    assert_eq!(d, sentinel);
}

#[test]
fn two_reusable_arenas() {
    let expected = reference();
    // Force packed execution: initialized Faer execution would not exercise
    // these arenas and could make a zero-byte retention assertion vacuous.
    let plan = BinaryPlan::<f64>::new(&matmul(CSpec::Absent), &PlanConfig::packed()).unwrap();
    let arena1 = ArenaProvider::traced();
    let arena2 = ArenaProvider::traced();

    // Alternate providers over warm calls; results stay reference-correct.
    let mut warm = 0usize;
    for round in 0..2 {
        for arena in [&arena1, &arena2] {
            let exec = Exec::serial_with_workspace(arena);
            let mut d = [0.0; 4];
            plan.execute_slices(&exec, 1.0, (&A_COL, 0), (&B_COL, 0), (&mut d, 0))
                .unwrap();
            close(&d, &expected, 1e-12);
        }
        if round == 0 {
            assert!(arena1.retained_bytes() > 0);
            assert!(arena2.retained_bytes() > 0);
            assert!(!arena1.trace_take().is_empty());
            assert!(!arena2.trace_take().is_empty());
        } else {
            warm = arena1.retained_bytes();
            assert!(arena1.trace_take().is_empty());
            assert!(arena2.trace_take().is_empty());
        }
    }

    // A retained arena does not keep growing: the third call reuses its scratch.
    let exec = Exec::serial_with_workspace(&arena1);
    let mut d = [0.0; 4];
    plan.execute_slices(&exec, 1.0, (&A_COL, 0), (&B_COL, 0), (&mut d, 0))
        .unwrap();
    assert_eq!(arena1.retained_bytes(), warm);
    assert_eq!(arena1.stats().leased_bytes, 0);
    assert!(arena1.trace_take().is_empty());
}

#[test]
fn accumulation_view_source_paths() {
    // The view-based accumulation entry points, both Output and Separate.
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [1.0, 0.0, 0.0, 1.0];
    let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
    let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
    let exec = Exec::serial();

    let out = BinaryPlan::<f64>::new(
        &identity_matmul(CSpec::Output(Op::Identity)),
        &PlanConfig::default(),
    )
    .unwrap();
    let mut d = [10.0; 4];
    let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
    out.execute_into_accum(
        &exec,
        1.0,
        &av,
        &bv,
        2.0,
        AccumulationSource::Output,
        &mut dv,
    )
    .unwrap();
    assert_eq!(d, [21.0, 22.0, 23.0, 24.0]);

    let sep = BinaryPlan::<f64>::new(
        &identity_matmul(CSpec::Separate(spec(&[2, 2], &[1, 2], 0))),
        &PlanConfig::default(),
    )
    .unwrap();
    let c = [0.5, 0.25, -1.0, 3.0];
    let cv = StridedView::new(&c, &[2, 2], &[1, 2], 0).unwrap();
    let mut d = [0.0; 4];
    let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
    sep.execute_into_accum(
        &exec,
        1.0,
        &av,
        &bv,
        3.0,
        AccumulationSource::Separate(&cv),
        &mut dv,
    )
    .unwrap();
    assert_eq!(d, [1.0 + 1.5, 2.0 + 0.75, 3.0 - 3.0, 4.0 + 9.0]);
}
