extern crate blas_src as _;

#[path = "../../cpueinsum/tests/common/mod.rs"]
mod common;

use common::{close, naive, Elem, Operand};
use cpueinsum::tprims_contract::api::Op;
use cpueinsum::tprims_contract::PlanConfig;
use cpueinsum::{Exec, GroupedGemmJob};
use cpueinsum_blas::{BlasError, BlasScalar, GroupedPlan, GroupedRoute};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn modes<T: Elem + BlasScalar>(tolerance: f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(83);
    let a: Vec<T> = (0..6).map(|_| T::random(&mut rng)).collect();
    let b: Vec<T> = (0..6).map(|_| T::random(&mut rng)).collect();
    let alpha = T::random(&mut rng);
    let beta = T::random(&mut rng);
    let marker = T::random(&mut rng);
    let jobs = [
        GroupedGemmJob::new(0, 0, 8, 2, 3, 2),
        GroupedGemmJob::new(0, 0, 0, 3, 2, 1),
    ];
    for mask in 0..8 {
        let ops = [0, 1, 2].map(|bit| {
            if mask & (1 << bit) != 0 {
                Op::Conjugate
            } else {
                Op::Identity
            }
        });
        let plan = GroupedPlan::<T>::new_with_ops(&jobs, ops, &PlanConfig::default()).unwrap();
        let mut work = vec![T::default(); plan.work_len()];
        let mut out = vec![marker; 12];
        let mut expected = out.clone();
        for job in jobs {
            let convert = |data: &[T], op: Op| {
                data.iter()
                    .copied()
                    .map(|x| if op.is_conj() { x.conjugate() } else { x })
                    .collect()
            };
            let aa = Operand::new(
                convert(&a[..job.rows * job.contracted], ops[0]),
                &[job.rows, job.contracted],
                &[1, job.rows as isize],
                0,
            )
            .labelled(&[0, 1]);
            let bb = Operand::new(
                convert(&b[..job.contracted * job.cols], ops[1]),
                &[job.contracted, job.cols],
                &[1, job.contracted as isize],
                0,
            )
            .labelled(&[1, 2]);
            let product = naive(&[aa, bb], &[0, 2], &[job.rows, job.cols]);
            for (i, value) in product.into_iter().enumerate() {
                let c = if ops[2].is_conj() {
                    marker.conjugate()
                } else {
                    marker
                };
                let value = alpha * value + beta * c;
                expected[job.out_offset + i] = if ops[2].is_conj() {
                    value.conjugate()
                } else {
                    value
                };
            }
        }
        for _ in 0..2 {
            out.fill(marker);
            assert_eq!(
                plan.execute_into_accum(&Exec::serial(), alpha, &a, &b, beta, &mut out, &mut work)
                    .unwrap(),
                GroupedRoute::Vendor { jobs: 2 }
            );
            close(&out, &expected, tolerance);
        }
    }
}

#[test]
fn heterogeneous_jobs_shared_inputs_gaps_and_all_four_dtype_conjugations() {
    modes::<f32>(2e-5);
    modes::<f64>(1e-12);
    modes::<num_complex::Complex32>(2e-5);
    modes::<num_complex::Complex64>(1e-12);
}

#[test]
fn late_vendor_bounds_and_workspace_errors_precede_every_output_write() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
        GroupedGemmJob::new(20, 0, 4, 2, 3, 2),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let mut out = [91.0; 8];
    assert!(matches!(
        plan.execute_into(
            &Exec::serial(),
            1.0,
            &[1.0; 6],
            &[1.0; 6],
            &mut out,
            &mut []
        ),
        Err(BlasError::Grouped { job: 1, .. })
    ));
    assert_eq!(out, [91.0; 8]);
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
        GroupedGemmJob::new(0, 0, 4, 2, 3, 2),
    ];
    let plan = GroupedPlan::<num_complex::Complex64>::new_with_ops(
        &jobs,
        [Op::Conjugate, Op::Identity, Op::Identity],
        &PlanConfig::default(),
    )
    .unwrap();
    assert!(plan.work_len() > 0);
    let value = num_complex::Complex64::new(1.0, 2.0);
    let mut out = [value; 8];
    assert!(matches!(
        plan.execute_into(
            &Exec::serial(),
            value,
            &[value; 6],
            &[value; 6],
            &mut out,
            &mut []
        ),
        Err(BlasError::Grouped { job: 0, .. })
    ));
    assert_eq!(out, [value; 8]);
}

#[test]
fn overlap_and_checked_byte_overflow_are_rejected_at_preparation() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 4, 2, 3, 2),
        GroupedGemmJob::new(0, 0, 6, 2, 3, 2),
    ];
    assert!(matches!(
        GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()),
        Err(BlasError::Prepare {
            source: cpueinsum::Error::Grouped(cpueinsum::GroupedError::Overlap { .. })
        })
    ));
    let jobs = [GroupedGemmJob::new(usize::MAX, 0, 0, 2, 3, 2)];
    assert!(matches!(
        GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()),
        Err(BlasError::Prepare {
            source: cpueinsum::Error::Grouped(cpueinsum::GroupedError::Overflow { .. })
        })
    ));
}

#[test]
fn one_degenerate_job_prepares_whole_group_native_and_zero_alpha_reads_no_inputs() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
        GroupedGemmJob::new(0, 0, 4, 2, 0, 2),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let mut out = [91.0; 8];
    let route = plan
        .execute_into(
            &Exec::serial(),
            1.0,
            &[1.0; 6],
            &[2.0; 6],
            &mut out,
            &mut [],
        )
        .unwrap();
    assert!(matches!(route, GroupedRoute::Native(_)));
    assert_eq!(out, [6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0]);
    let jobs = [GroupedGemmJob::new(0, 0, 1, 2, 3, 2)];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let mut out = [91.0; 6];
    let route = plan
        .execute_into(
            &Exec::serial(),
            0.0,
            &[f64::NAN; 6],
            &[f64::NAN; 6],
            &mut out,
            &mut [],
        )
        .unwrap();
    assert!(matches!(route, GroupedRoute::Native(_)));
    assert_eq!(out, [91.0, 0.0, 0.0, 0.0, 0.0, 91.0]);
}

#[test]
fn no_materialize_selects_whole_group_native_for_conjugate_input_packing() {
    use num_complex::Complex64;
    let jobs = [
        GroupedGemmJob::new(0, 0, 4, 2, 3, 2),
        GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
    ];
    let config = PlanConfig {
        no_materialize: true,
        ..PlanConfig::default()
    };
    let plan = GroupedPlan::<Complex64>::new_with_ops(
        &jobs,
        [Op::Conjugate, Op::Identity, Op::Identity],
        &config,
    )
    .unwrap();
    assert_eq!(plan.work_len(), 0);
    let a = [Complex64::new(1.0, 2.0); 6];
    let b = [Complex64::new(2.0, 1.0); 6];
    let mut out = [Complex64::new(91.0, 2.0); 8];
    let route = plan
        .execute_into(
            &Exec::serial(),
            Complex64::new(1.0, 0.0),
            &a,
            &b,
            &mut out,
            &mut [],
        )
        .unwrap();
    assert!(matches!(route, GroupedRoute::Native(_)));
    assert_eq!(out, [Complex64::new(12.0, -9.0); 8]);
}
