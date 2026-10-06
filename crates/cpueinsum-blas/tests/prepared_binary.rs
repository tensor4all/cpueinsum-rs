//! Prepared BLAS output contracts against the shared naive reference.
extern crate blas_src as _;

#[path = "../../cpueinsum/tests/common/mod.rs"]
mod common;

use common::{close, naive, Elem, Operand};
use cpueinsum::tprims_contract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem};
use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::Exec;
use cpueinsum_blas::{BinaryPlan, BlasError, BlasScalar, ExecutionRoute};
use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::mem::MaybeUninit;

fn spec(dims: &[usize], strides: &[isize], conjugate: bool) -> OperandSpec {
    // Nonzero planned offsets deliberately differ from execution origins.
    OperandSpec::new(LayoutSpec::new(dims, strides, 9).unwrap()).with_op(if conjugate {
        Op::Conjugate
    } else {
        Op::Identity
    })
}

fn problem<T: BlasScalar>(ops: [bool; 3], c: CSpec, ds: &[isize]) -> Problem {
    let labels = Labels::new(&[0, 1], &[1, 2], &[0, 2]);
    let labels = if matches!(c, CSpec::Separate(_)) {
        labels.with_c(&[0, 2])
    } else {
        labels
    };
    Problem::from_labels(
        T::STORAGE,
        spec(&[2, 3], &[1, 2], ops[0]),
        spec(&[3, 2], &[1, 3], ops[1]),
        c,
        spec(&[2, 2], ds, ops[2]),
        &labels,
    )
    .unwrap()
}

fn reference<T: Elem + BlasScalar>(a: &[T], b: &[T], ops: [bool; 3]) -> Vec<T> {
    let convert = |values: &[T], conj: bool| {
        values
            .iter()
            .copied()
            .map(|x| if conj { x.conjugate() } else { x })
            .collect()
    };
    let a = Operand::new(convert(a, ops[0]), &[2, 3], &[1, 2], 0).labelled(&[0, 1]);
    let b = Operand::new(convert(b, ops[1]), &[3, 2], &[1, 3], 0).labelled(&[1, 2]);
    naive(&[a, b], &[0, 2], &[2, 2])
}

fn all_modes<T: Elem + BlasScalar>(tolerance: f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(61);
    let a: Vec<T> = (0..6).map(|_| T::random(&mut rng)).collect();
    let b: Vec<T> = (0..6).map(|_| T::random(&mut rng)).collect();
    let c: Vec<T> = (0..4).map(|_| T::random(&mut rng)).collect();
    let alpha = T::random(&mut rng);
    let beta = T::random(&mut rng);
    for mask in 0..16 {
        let ops = [mask & 1 != 0, mask & 2 != 0, mask & 8 != 0];
        let conj_c = mask & 4 != 0;
        let product = reference(&a, &b, ops);
        for mode in 0..3 {
            let c_spec = match mode {
                0 => CSpec::Absent,
                1 => CSpec::Output(if conj_c { Op::Conjugate } else { Op::Identity }),
                _ => CSpec::Separate(spec(&[2, 2], &[1, 2], conj_c)),
            };
            let plan =
                BinaryPlan::<T>::new(&problem::<T>(ops, c_spec, &[1, 2]), &PlanConfig::default())
                    .unwrap();
            let mut work = vec![T::default(); plan.work_len()];
            let beta = if mode == 0 { T::default() } else { beta };
            let source = match mode {
                0 => SliceAccumulationSource::Absent,
                1 => SliceAccumulationSource::Output,
                _ => SliceAccumulationSource::Separate((&c, 0)),
            };
            let expected: Vec<T> = product
                .iter()
                .zip(&c)
                .map(|(&p, &old)| {
                    let old = if conj_c { old.conjugate() } else { old };
                    let value = alpha * p + beta * old;
                    if ops[2] {
                        value.conjugate()
                    } else {
                        value
                    }
                })
                .collect();
            let mut out = c.clone();
            for _ in 0..2 {
                out.copy_from_slice(&c);
                let route = plan
                    .execute_slices_accum(
                        &Exec::serial(),
                        alpha,
                        (&a, 0),
                        (&b, 0),
                        beta,
                        source,
                        (&mut out, 0),
                        &mut work,
                    )
                    .unwrap();
                let vendor = mode == 0 || (mode == 1 && (!T::COMPLEX || conj_c == ops[2]));
                assert_eq!(matches!(route, ExecutionRoute::Vendor), vendor);
                close(&out, &expected, tolerance);
            }
            if mode == 0 || mode == 2 {
                let mut fresh = vec![MaybeUninit::uninit(); 4];
                let (out, route) = plan
                    .execute_uninit_slices(
                        &Exec::serial(),
                        alpha,
                        (&a, 0),
                        (&b, 0),
                        beta,
                        source,
                        (&mut fresh, 0),
                        &mut work,
                    )
                    .unwrap();
                assert_eq!(matches!(route, ExecutionRoute::Vendor), mode == 0);
                close(out, &expected, tolerance);
            }
        }
    }
}

#[test]
fn all_four_dtypes_scales_and_conjugation_masks() {
    all_modes::<f32>(2e-5);
    all_modes::<f64>(1e-12);
    all_modes::<Complex32>(2e-5);
    all_modes::<Complex64>(1e-12);
}

#[test]
fn bounds_workspace_and_accumulation_fail_before_output_writes() {
    let plan = BinaryPlan::<Complex64>::new(
        &problem::<Complex64>([true, false, false], CSpec::Absent, &[1, 2]),
        &PlanConfig::default(),
    )
    .unwrap();
    assert!(plan.work_len() > 0);
    let a = [Complex64::new(1.0, 1.0); 6];
    let b = [Complex64::new(1.0, -1.0); 6];
    let marker = Complex64::new(91.0, 2.0);
    let mut d = [marker; 4];
    let mut work = vec![Complex64::default(); plan.work_len()];
    assert!(matches!(
        plan.execute_slices_accum(
            &Exec::serial(),
            Complex64::new(1.0, 0.0),
            (&a[..5], 0),
            (&b, 0),
            Complex64::default(),
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut work
        ),
        Err(BlasError::OutOfBounds { .. })
    ));
    assert_eq!(d, [marker; 4]);
    assert!(matches!(
        plan.execute_slices_accum(
            &Exec::serial(),
            Complex64::new(1.0, 0.0),
            (&a, 0),
            (&b, 0),
            Complex64::default(),
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut []
        ),
        Err(BlasError::Work { .. })
    ));
    assert_eq!(d, [marker; 4]);
    assert!(matches!(
        plan.execute_slices_accum(
            &Exec::serial(),
            Complex64::new(1.0, 0.0),
            (&a, 0),
            (&b, 0),
            Complex64::new(1.0, 0.0),
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut work
        ),
        Err(BlasError::Accumulation)
    ));
    assert_eq!(d, [marker; 4]);
}

#[test]
fn fresh_rejects_padding_and_gaps_and_output_reads() {
    let a = [1.0; 6];
    let b = [2.0; 6];
    for (strides, length) in [([1, 2], 5), ([1, 3], 5)] {
        let plan = BinaryPlan::<f64>::new(
            &problem::<f64>([false; 3], CSpec::Absent, &strides),
            &PlanConfig::default(),
        )
        .unwrap();
        let mut fresh = vec![MaybeUninit::new(91.0); length];
        assert!(matches!(
            plan.execute_uninit_slices(
                &Exec::serial(),
                1.0,
                (&a, 0),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Absent,
                (&mut fresh, 0),
                &mut []
            ),
            Err(BlasError::FreshCoverage)
        ));
        // Every slot was initialized by this test before the refused call.
        for slot in fresh {
            assert_eq!(unsafe { slot.assume_init() }, 91.0);
        }
    }
    let plan = BinaryPlan::<f64>::new(
        &problem::<f64>([false; 3], CSpec::Output(Op::Identity), &[1, 2]),
        &PlanConfig::default(),
    )
    .unwrap();
    let mut fresh = [MaybeUninit::uninit(); 4];
    assert!(matches!(
        plan.execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&a, 0),
            (&b, 0),
            1.0,
            SliceAccumulationSource::Output,
            (&mut fresh, 0),
            &mut []
        ),
        Err(BlasError::Accumulation)
    ));
}

#[test]
fn zero_alpha_uses_prepared_native_no_read_route() {
    let plan = BinaryPlan::<f64>::new(
        &problem::<f64>([false; 3], CSpec::Absent, &[1, 2]),
        &PlanConfig::default(),
    )
    .unwrap();
    let input = [f64::NAN; 6];
    let mut out = [f64::NAN; 4];
    let route = plan
        .execute_slices_accum(
            &Exec::serial(),
            0.0,
            (&input, 0),
            (&input, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut out, 0),
            &mut [],
        )
        .unwrap();
    assert!(matches!(route, ExecutionRoute::Native(_)));
    assert_eq!(out, [0.0; 4]);
    let mut fresh = [MaybeUninit::uninit(); 4];
    let (out, route) = plan
        .execute_uninit_slices(
            &Exec::serial(),
            0.0,
            (&input, 0),
            (&input, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut fresh, 0),
            &mut [],
        )
        .unwrap();
    assert!(matches!(route, ExecutionRoute::Native(_)));
    assert_eq!(out, &[0.0; 4]);
}
