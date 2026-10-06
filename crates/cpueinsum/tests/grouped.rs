mod common;

use common::{close, naive, Elem, Operand};
use cpueinsum::tprims_contract::{api::Op, PlanConfig};
use cpueinsum::{Error, Exec, GroupedError, GroupedGemmJob, GroupedPlan, GroupedRoute};
use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn gemm(a: &[f64], b: &[f64], rows: usize, contracted: usize, cols: usize) -> Vec<f64> {
    (0..cols)
        .flat_map(|j| {
            (0..rows).map(move |i| {
                (0..contracted)
                    .map(|k| a[i + rows * k] * b[k + contracted * j])
                    .sum()
            })
        })
        .collect()
}

#[test]
fn heterogeneous_shared_inputs_unsorted_outputs_and_holes() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 5, 2, 2, 1),
        GroupedGemmJob::new(4, 2, 0, 1, 2, 2),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    assert_eq!(plan.required_lhs_len(), 6);
    assert_eq!(plan.required_rhs_len(), 6);
    assert_eq!(plan.required_out_len(), 7);

    let lhs = [1., 2., 3., 4., 5., 6., 7., 8.];
    let rhs = [1., 0., 0., 1., 2., 3.];
    let mut out = vec![-9.; 8];
    let route = plan
        .execute_into(&Exec::serial(), 1.0, &lhs, &rhs, &mut out)
        .unwrap();
    assert_eq!(route, GroupedRoute::Serial);
    let mut expected = vec![-9.; 8];
    expected[0..2].copy_from_slice(&gemm(&lhs[4..6], &rhs[2..6], 1, 2, 2));
    expected[5..7].copy_from_slice(&gemm(&lhs[0..4], &rhs[0..2], 2, 2, 1));
    assert_eq!(out, expected);
}

#[test]
fn alpha_beta_accumulation_uses_previous_output() {
    let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let lhs = [1., 2., 3., 4.];
    let rhs = [2., 0., 0., 2.];
    let mut out = [10., 20., 30., 40.];
    plan.execute_into_accum(&Exec::serial(), 2.0, &lhs, &rhs, 0.5, &mut out)
        .unwrap();
    let product = gemm(&lhs, &rhs, 2, 2, 2);
    let expected: Vec<_> = product
        .iter()
        .zip([10., 20., 30., 40.])
        .map(|(x, old)| 2.0 * x + 0.5 * old)
        .collect();
    assert_eq!(out, expected.as_slice());
}

#[test]
fn all_bounds_are_checked_before_any_write() {
    for offsets in [[0, 1], [1, 0]] {
        let jobs = [
            GroupedGemmJob::new(0, 0, offsets[0], 1, 1, 1),
            GroupedGemmJob::new(99, 0, offsets[1], 1, 1, 1),
        ];
        let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
        let mut out = [77.; 2];
        let err = plan
            .execute_into(&Exec::serial(), 1.0, &[3.], &[4.], &mut out)
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Grouped(GroupedError::Bounds {
                job: 1,
                operand: cpueinsum::GroupedOperand::Lhs,
                ..
            })
        ));
        assert_eq!(out, [77., 77.]);
    }
}

#[test]
fn overlap_and_checked_overflow_are_plan_errors() {
    let overlap = [
        GroupedGemmJob::new(0, 0, 2, 2, 1, 2),
        GroupedGemmJob::new(0, 0, 3, 1, 1, 2),
    ];
    assert!(matches!(
        GroupedPlan::<f64>::new(&overlap, &PlanConfig::default()),
        Err(Error::Grouped(GroupedError::Overlap {
            first: 0,
            second: 1
        }))
    ));

    let overflow = [GroupedGemmJob::new(0, 0, 0, usize::MAX, 2, 1)];
    assert!(matches!(
        GroupedPlan::<f64>::new(&overflow, &PlanConfig::default()),
        Err(Error::Grouped(GroupedError::Overflow { job: 0, .. }))
    ));
    // Element arithmetic fits, but no Rust f64 slice can cover this byte reach.
    let offset = isize::MAX as usize / core::mem::size_of::<f64>() + 1;
    let byte_overflow = [GroupedGemmJob::new(offset, 0, 0, 1, 1, 1)];
    assert!(matches!(
        GroupedPlan::<f64>::new(&byte_overflow, &PlanConfig::default()),
        Err(Error::Grouped(GroupedError::Overflow {
            job: 0,
            what: "lhs byte span"
        }))
    ));
}

#[test]
fn empty_jobs_are_validated_and_zero_jobs_are_empty() {
    let empty = [GroupedGemmJob::new(4, 3, 9, 0, 5, 2)];
    let plan = GroupedPlan::<f64>::new(&empty, &PlanConfig::default()).unwrap();
    let mut out = [11.; 9];
    assert_eq!(
        plan.execute_into(&Exec::serial(), 1.0, &[0.; 4], &[0.; 13], &mut out)
            .unwrap(),
        GroupedRoute::Serial
    );
    assert_eq!(out, [11.; 9]);

    let none: [GroupedGemmJob; 0] = [];
    let plan = GroupedPlan::<f64>::new(&none, &PlanConfig::default()).unwrap();
    assert_eq!(
        plan.execute_into(&Exec::serial(), 1.0, &[], &[], &mut [])
            .unwrap(),
        GroupedRoute::Empty
    );
}

#[test]
fn bounded_pool_uses_outer_lanes_without_nested_entries() {
    let jobs: Vec<_> = (0..4)
        .map(|i| GroupedGemmJob::new(i * 4096, i * 4096, i * 4096, 64, 64, 64))
        .collect();
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let lhs = vec![1.; 4 * 4096];
    let rhs = vec![1.; 4 * 4096];
    let mut out = vec![1.; 4 * 4096];
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let exec = Exec::rayon(&pool).with_budget(2).unwrap();
    let route = plan
        .execute_into_accum(&exec, 1.0, &lhs, &rhs, 1.0, &mut out)
        .unwrap();
    assert!(matches!(route, GroupedRoute::Outer { lanes: 2 }));
    // Accumulation detects duplicate job execution, unlike idempotent overwrite.
    assert!(out.iter().all(|&x| x == 65.));
    assert_eq!(pool.stats().broadcasts, 0);
    assert_eq!(pool.stats().entries, 1);

    for budget in [1, 4] {
        pool.reset_stats();
        out.fill(1.0);
        let bounded = Exec::rayon(&pool).with_budget(budget).unwrap();
        let route = plan
            .execute_into_accum(&bounded, 1.0, &lhs, &rhs, 1.0, &mut out)
            .unwrap();
        if budget == 1 {
            assert_eq!(route, GroupedRoute::Serial);
            assert_eq!(pool.stats().entries, 0);
        } else {
            assert!(matches!(route, GroupedRoute::Outer { lanes } if lanes > 1 && lanes <= budget));
            assert_eq!(pool.stats().entries, 1);
        }
        assert_eq!(pool.stats().broadcasts, 0);
        assert!(out.iter().all(|&x| x == 65.));
    }
    pool.reset_stats();
    thread_pool.install(|| {
        plan.execute_into(&exec, 1.0, &lhs, &rhs, &mut out).unwrap();
    });
    assert_eq!(pool.stats().entries, 0);
    assert!(pool.stats().inline_runs > 0);
}

fn scalar_group<T: Elem>(conj: impl Fn(T) -> T + Copy, tol: f64) {
    let jobs = [
        GroupedGemmJob::new(0, 0, 12, 2, 3, 2),
        GroupedGemmJob::new(4, 3, 0, 3, 2, 4),
        GroupedGemmJob::new(10, 11, 18, 2, 0, 1),
        GroupedGemmJob::new(10, 11, 30, 0, 3, 2),
    ];
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let lhs: Vec<T> = (0..10).map(|_| T::random(&mut rng)).collect();
    let rhs: Vec<T> = (0..17).map(|_| T::random(&mut rng)).collect();
    let old: Vec<T> = (0..30).map(|_| T::random(&mut rng)).collect();
    let alpha = T::random(&mut rng);
    let beta = T::random(&mut rng);
    let arena = cpueinsum::ArenaProvider::traced();
    let exec = Exec::serial_with_workspace(&arena); // Explicit 1T backend.
    for conjugate in [false, true] {
        let apply = |x| if conjugate { conj(x) } else { x };
        let ops = [if conjugate {
            Op::Conjugate
        } else {
            Op::Identity
        }; 3];
        let plan = GroupedPlan::<T>::new_with_ops(&jobs, ops, &PlanConfig::packed()).unwrap();
        for scale in [T::default(), beta] {
            let mut expected = old.clone();
            for job in jobs {
                let a = lhs[job.lhs_offset..job.lhs_offset + job.rows * job.contracted]
                    .iter()
                    .copied()
                    .map(apply)
                    .collect();
                let b = rhs[job.rhs_offset..job.rhs_offset + job.contracted * job.cols]
                    .iter()
                    .copied()
                    .map(apply)
                    .collect();
                let product = naive(
                    &[
                        Operand::new(
                            a,
                            &[job.rows, job.contracted],
                            &[1, job.rows.max(1) as isize],
                            0,
                        )
                        .labelled(&[0, 1]),
                        Operand::new(
                            b,
                            &[job.contracted, job.cols],
                            &[1, job.contracted.max(1) as isize],
                            0,
                        )
                        .labelled(&[1, 2]),
                    ],
                    &[0, 2],
                    &[job.rows, job.cols],
                );
                for (i, value) in product.iter().enumerate() {
                    let mut result = alpha * *value;
                    if scale != T::default() {
                        result += scale * apply(old[job.out_offset + i]);
                    }
                    expected[job.out_offset + i] = apply(result);
                }
            }
            let mut out = old.clone();
            let route = plan
                .execute_into_accum(&exec, alpha, &lhs, &rhs, scale, &mut out)
                .unwrap();
            assert_eq!(route, GroupedRoute::Serial);
            close(&out, &expected, tol);
            assert_eq!(arena.stats().leased_bytes, 0);
        }
    }
    assert!(arena.retained_bytes() > 0);
    let warm = arena.retained_bytes();
    arena.trace_take();
    let plan = GroupedPlan::<T>::new(&jobs, &PlanConfig::packed()).unwrap();
    plan.execute_into(&exec, alpha, &lhs, &rhs, &mut old.clone())
        .unwrap();
    assert_eq!(arena.retained_bytes(), warm);
    assert!(arena.trace_take().is_empty());
}

#[test]
fn grouped_all_dtypes_conjugation_empty_k_and_warm_workspace() {
    scalar_group::<f32>(|x| x, 2e-5);
    scalar_group::<f64>(|x| x, 1e-12);
    scalar_group::<Complex32>(|x| x.conj(), 3e-5);
    scalar_group::<Complex64>(|x| x.conj(), 1e-12);
}
