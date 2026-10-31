extern crate blas_src as _;

#[path = "../../cpueinsum/tests/common/mod.rs"]
mod common;

use common::{close, naive, Elem, Operand};
use core::mem::MaybeUninit;
use cpueinsum::tcontract::api::Op;
use cpueinsum::tcontract::PlanConfig;
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
                plan.execute_into_accum(
                    &mut Exec::serial(),
                    alpha,
                    &a,
                    &b,
                    beta,
                    &mut out,
                    &mut work
                )
                .unwrap(),
                GroupedRoute::Vendor { jobs: 2 }
            );
            close(&out, &expected, tolerance);
        }
        // The same heterogeneous, unsorted jobs with compact output exercise
        // fresh vendor publication for all four dtypes and all conjugations.
        let compact = [
            GroupedGemmJob {
                out_offset: 3,
                ..jobs[0]
            },
            jobs[1],
        ];
        let fresh_plan =
            GroupedPlan::<T>::new_with_ops(&compact, ops, &PlanConfig::default()).unwrap();
        let mut expected = vec![T::default(); 7];
        fresh_plan
            .execute_into(&mut Exec::serial(), alpha, &a, &b, &mut expected, &mut work)
            .unwrap();
        let mut fresh = vec![MaybeUninit::uninit(); 7];
        let (out, route) = fresh_plan
            .execute_uninit(
                &mut Exec::serial(),
                alpha,
                &a,
                &b,
                T::default(),
                &mut fresh,
                &mut work,
            )
            .unwrap();
        assert_eq!(route, GroupedRoute::Vendor { jobs: 2 });
        close(out, &expected, tolerance);
    }
}

#[test]
fn auto_keeps_the_vendor_loop_and_forced_lanes_use_the_native_route() {
    use core::num::NonZeroUsize;
    use cpueinsum::{BatchPolicy, BatchStrategy};
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 16, 16, 16),
        GroupedGemmJob::new(0, 0, 256, 16, 16, 16),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let a: Vec<f64> = (0..256).map(|i| ((i % 7) as f64) - 3.0).collect();
    let b: Vec<f64> = (0..256).map(|i| ((i % 5) as f64) - 2.0).collect();
    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);

    // Auto uses the coordinator vendor loop.
    let mut vendor = vec![0.0; 512];
    let mut exec = cpueinsum::Exec::new(&pool, pool.size()).unwrap();
    assert_eq!(
        plan.execute_into(
            &mut exec,
            1.0,
            &a,
            &b,
            &mut vendor,
            &mut vec![0.0; plan.work_len()]
        )
        .unwrap(),
        GroupedRoute::Vendor { jobs: 2 }
    );

    // Forced lanes cannot be served by the serial loop, so the native
    // alternative takes them; the arithmetic route changes, so compare.
    for strategy in [
        BatchStrategy::Outer,
        BatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        },
        BatchStrategy::Inner,
    ] {
        let mut native = vec![0.0; 512];
        let mut exec = cpueinsum::Exec::new(&pool, 4)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy));
        let route = plan
            .execute_into(
                &mut exec,
                1.0,
                &a,
                &b,
                &mut native,
                &mut vec![0.0; plan.work_len()],
            )
            .unwrap();
        assert!(
            matches!(route, GroupedRoute::Native(_)),
            "{strategy:?} -> {route:?}"
        );
        close(&native, &vendor, 1e-12);
        let mut fresh = vec![MaybeUninit::uninit(); 512];
        let (out, fresh_route) = plan
            .execute_uninit(
                &mut exec,
                1.0,
                &a,
                &b,
                0.0,
                &mut fresh,
                &mut vec![0.0; plan.work_len()],
            )
            .unwrap();
        assert_eq!(fresh_route, route);
        close(out, &vendor, 1e-12);
    }

    // Forced Sequential is exactly one serial coordinator, which the loop
    // already is, so it stays on the vendor route (and is bitwise equal).
    let mut sequential = vec![0.0; 512];
    let mut exec = cpueinsum::Exec::new(&pool, pool.size())
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Sequential));
    assert_eq!(
        plan.execute_into(
            &mut exec,
            1.0,
            &a,
            &b,
            &mut sequential,
            &mut vec![0.0; plan.work_len()]
        )
        .unwrap(),
        GroupedRoute::Vendor { jobs: 2 }
    );
    assert_eq!(sequential, vendor);
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
            &mut Exec::serial(),
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
            &mut Exec::serial(),
            value,
            &[value; 6],
            &[value; 6],
            &mut out,
            &mut []
        ),
        Err(BlasError::Grouped { job: 0, .. })
    ));
    assert_eq!(out, [value; 8]);
    let mut fresh = [MaybeUninit::new(value); 8];
    assert!(matches!(
        plan.execute_uninit(
            &mut Exec::serial(),
            value,
            &[value; 6],
            &[value; 6],
            Default::default(),
            &mut fresh,
            &mut []
        ),
        Err(BlasError::Grouped { job: 0, .. })
    ));
    // SAFETY: every slot began initialized; workspace rejection precedes writes.
    assert!(fresh
        .into_iter()
        .all(|slot| unsafe { slot.assume_init() } == value));
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
            &mut Exec::serial(),
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
            &mut Exec::serial(),
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
            &mut Exec::serial(),
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

// ---- fresh (uninitialized) grouped vendor output ----

/// Every slot of a refused fresh call must still hold this value.
const SENTINEL: f64 = 91.0;

fn fresh_pair() -> ([f64; 4], [f64; 4]) {
    ([1.0, 2.0, 3.0, 4.0], [1.0, -1.0, 0.5, 2.0])
}

#[test]
fn fresh_full_coverage_overwrites_the_sentinel_and_matches_vendor_reference() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 2, 2),
        GroupedGemmJob::new(0, 0, 4, 2, 2, 1),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let (a, b) = fresh_pair();
    let mut work = vec![0.0; plan.work_len()];
    let mut reference = [0.0; 6];
    assert_eq!(
        plan.execute_into(&mut Exec::serial(), 1.5, &a, &b, &mut reference, &mut work)
            .unwrap(),
        GroupedRoute::Vendor { jobs: 2 }
    );
    let mut fresh = [MaybeUninit::new(SENTINEL); 6];
    let (out, route) = plan
        .execute_uninit(&mut Exec::serial(), 1.5, &a, &b, 0.0, &mut fresh, &mut work)
        .unwrap();
    assert_eq!(route, GroupedRoute::Vendor { jobs: 2 });
    assert_eq!(out, reference);
    assert!(
        out.iter().all(|&v| v != SENTINEL),
        "an untouched sentinel survived a full fresh write: {out:?}"
    );
}

#[test]
fn fresh_uncovered_element_errors_and_leaves_every_sentinel() {
    let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let (a, b) = fresh_pair();
    let mut work = vec![0.0; plan.work_len()];
    for len in [3usize, 5] {
        let mut fresh = vec![MaybeUninit::new(SENTINEL); len];
        let error = plan
            .execute_uninit(&mut Exec::serial(), 1.0, &a, &b, 0.0, &mut fresh, &mut work)
            .unwrap_err();
        assert!(
            matches!(error, BlasError::FreshCoverage),
            "len {len}: {error:?}"
        );
        for slot in fresh {
            assert_eq!(unsafe { slot.assume_init() }, SENTINEL);
        }
    }
}

#[test]
fn fresh_nonzero_beta_errors_before_writes() {
    let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let (a, b) = fresh_pair();
    let mut work = vec![0.0; plan.work_len()];
    for (beta, alpha) in [(1.0, 1.0), (f64::NAN, 1.0), (1.0, 0.0)] {
        let mut fresh = [MaybeUninit::new(SENTINEL); 4];
        let error = plan
            .execute_uninit(
                &mut Exec::serial(),
                alpha,
                &a,
                &b,
                beta,
                &mut fresh,
                &mut work,
            )
            .unwrap_err();
        assert!(
            matches!(error, BlasError::Accumulation),
            "beta {beta}, alpha {alpha}: {error:?}"
        );
        for slot in fresh {
            assert_eq!(unsafe { slot.assume_init() }, SENTINEL);
        }
    }
    let mut empty: [MaybeUninit<f64>; 0] = [];
    assert!(matches!(
        plan.execute_uninit(
            &mut Exec::serial(),
            0.0,
            &[],
            &[],
            1.0,
            &mut empty,
            &mut work
        ),
        Err(BlasError::Accumulation)
    ));
}

#[test]
fn fresh_zero_alpha_uses_the_native_route_and_late_bounds_precede_writes() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 2, 2),
        GroupedGemmJob::new(0, 0, 4, 2, 2, 1),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let (a, b) = fresh_pair();
    let mut work = vec![0.0; plan.work_len()];
    // Alpha zero reads no operand and takes the native fresh alternative.
    let mut reference = [0.0; 6];
    plan.execute_into(&mut Exec::serial(), 0.0, &a, &b, &mut reference, &mut work)
        .unwrap();
    let mut fresh = [MaybeUninit::uninit(); 6];
    let (out, route) = plan
        .execute_uninit(
            &mut Exec::serial(),
            0.0,
            &[f64::NAN; 4],
            &[f64::NAN; 4],
            0.0,
            &mut fresh,
            &mut work,
        )
        .unwrap();
    assert!(matches!(route, GroupedRoute::Native(_)), "{route:?}");
    assert_eq!(out, reference.as_slice());
    assert_eq!(out, [0.0; 6]);

    // Late vendor bounds are preflighted before any write.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 2, 2),
        GroupedGemmJob::new(64, 0, 4, 2, 2, 1),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let mut work = vec![0.0; plan.work_len()];
    let mut fresh = [MaybeUninit::new(SENTINEL); 6];
    let error = plan
        .execute_uninit(&mut Exec::serial(), 1.0, &a, &b, 0.0, &mut fresh, &mut work)
        .unwrap_err();
    assert!(
        matches!(error, BlasError::Grouped { job: 1, .. }),
        "{error:?}"
    );
    for slot in fresh {
        assert_eq!(unsafe { slot.assume_init() }, SENTINEL);
    }

    // An empty group publishes an empty destination.
    let plan = GroupedPlan::<f64>::new(&[], &PlanConfig::default()).unwrap();
    let mut none: [MaybeUninit<f64>; 0] = [];
    let (out, route) = plan
        .execute_uninit(&mut Exec::serial(), 1.0, &[], &[], 0.0, &mut none, &mut [])
        .unwrap();
    assert!(out.is_empty());
    assert_eq!(route, GroupedRoute::Empty);
}
