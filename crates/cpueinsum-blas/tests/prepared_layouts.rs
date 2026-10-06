extern crate blas_src as _;

#[path = "../../cpueinsum/tests/common/mod.rs"]
mod common;

use common::{close, naive, Operand};
use cpueinsum::tprims_contract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::Exec;
use cpueinsum_blas::{BinaryPlan, ExecutionRoute};
use std::mem::MaybeUninit;

fn spec(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 9).unwrap())
}

#[test]
fn reversed_gapped_broadcast_inputs_and_beta_zero_ignore_old_nan_output() {
    for (strides, origin, data) in [
        ([-1, 2], 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        (
            [2, 4],
            0,
            vec![1.0, 91.0, 2.0, 91.0, 3.0, 91.0, 4.0, 91.0, 5.0, 91.0, 6.0],
        ),
        ([0, 0], 0, vec![3.0]),
    ] {
        let problem = Problem::from_labels(
            DType::F64,
            spec(&[2, 3], &strides),
            spec(&[3, 2], &[1, 3]),
            CSpec::Output(Op::Identity),
            spec(&[2, 2], &[1, 2]),
            &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
        )
        .unwrap();
        let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
        let b = [1.0; 6];
        let aa = Operand::new(data.clone(), &[2, 3], &strides, origin).labelled(&[0, 1]);
        let bb = Operand::new(b.to_vec(), &[3, 2], &[1, 3], 0).labelled(&[1, 2]);
        let expected = naive(&[aa, bb], &[0, 2], &[2, 2]);
        let mut work = vec![f64::NAN; plan.work_len()];
        let mut out = [f64::NAN; 4];
        let route = plan
            .execute_slices_accum(
                &Exec::serial(),
                1.0,
                (&data, origin as isize),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Output,
                (&mut out, 0),
                &mut work,
            )
            .unwrap();
        assert_eq!(route, ExecutionRoute::Vendor);
        close(&out, &expected, 1e-12);
        let mut fresh = [MaybeUninit::uninit(); 4];
        let (initialized, route) = plan
            .execute_uninit_slices(
                &Exec::serial(),
                1.0,
                (&data, origin as isize),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Output,
                (&mut fresh, 0),
                &mut work,
            )
            .unwrap();
        assert_eq!(route, ExecutionRoute::Vendor);
        close(initialized, &expected, 1e-12);
    }
}

#[test]
fn unsupported_direct_output_is_native_without_writing_padding() {
    let problem = Problem::from_labels(
        DType::F64,
        spec(&[2, 3], &[1, 2]),
        spec(&[3, 2], &[1, 3]),
        CSpec::Absent,
        spec(&[2, 2], &[2, 5]),
        &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
    )
    .unwrap();
    let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let mut out = [91.0; 8];
    let route = plan
        .execute_slices_accum(
            &Exec::serial(),
            1.0,
            (&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0),
            (&[1.0; 6], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut out, 0),
            &mut [],
        )
        .unwrap();
    assert!(matches!(route, ExecutionRoute::Native(_)));
    assert_eq!(out, [9.0, 91.0, 12.0, 91.0, 91.0, 9.0, 91.0, 12.0]);
}

#[test]
fn no_materialize_preserves_direct_vendor_but_refuses_whole_input_packing() {
    let config = PlanConfig {
        no_materialize: true,
        ..PlanConfig::default()
    };
    for (strides, origin, vendor) in [([1, 2], 0, true), ([-1, 2], 1, false)] {
        let problem = Problem::from_labels(
            DType::F64,
            spec(&[2, 3], &strides),
            spec(&[3, 2], &[1, 3]),
            CSpec::Absent,
            spec(&[2, 2], &[1, 2]),
            &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
        )
        .unwrap();
        let plan = BinaryPlan::<f64>::new(&problem, &config).unwrap();
        assert_eq!(plan.work_len(), 0);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0; 6];
        let expected = naive(
            &[
                Operand::new(a.to_vec(), &[2, 3], &strides, origin).labelled(&[0, 1]),
                Operand::new(b.to_vec(), &[3, 2], &[1, 3], 0).labelled(&[1, 2]),
            ],
            &[0, 2],
            &[2, 2],
        );
        let mut out = [91.0; 4];
        let route = plan
            .execute_slices_accum(
                &Exec::serial(),
                1.0,
                (&a, origin as isize),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Absent,
                (&mut out, 0),
                &mut [],
            )
            .unwrap();
        assert_eq!(matches!(route, ExecutionRoute::Vendor), vendor);
        close(&out, &expected, 1e-12);
    }
}

#[test]
fn fresh_coverage_counts_reduced_output_roles_not_original_axes() {
    let problem = Problem::from_labels(
        DType::F64,
        spec(&[2, 3], &[1, 2]),
        spec(&[3, 2], &[1, 3]),
        CSpec::Absent,
        spec(&[2, 2, 2], &[2, 4, 1]),
        &Labels::new(&[0, 1], &[1, 2], &[0, 0, 2]),
    )
    .unwrap();
    let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let mut work = vec![0.0; plan.work_len()];
    let mut out = [91.0; 8];
    assert_eq!(
        plan.execute_slices_accum(
            &Exec::serial(),
            1.0,
            (&[1.0; 6], 0),
            (&[1.0; 6], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut out, 0),
            &mut work,
        )
        .unwrap(),
        ExecutionRoute::Vendor
    );
    assert_eq!(out, [3.0, 3.0, 91.0, 91.0, 91.0, 91.0, 3.0, 3.0]);
    let mut fresh = [MaybeUninit::new(91.0); 8];
    assert!(matches!(
        plan.execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[1.0; 6], 0),
            (&[1.0; 6], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut fresh, 0),
            &mut work,
        ),
        Err(cpueinsum_blas::BlasError::FreshCoverage)
    ));
    for slot in fresh {
        // SAFETY: every slot was initialized before the call.
        assert_eq!(unsafe { slot.assume_init() }, 91.0);
    }
}
