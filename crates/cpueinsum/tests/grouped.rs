mod common;

use common::{close, naive, Elem, Operand};
use cpueinsum::tcontract::{api::Op, PlanConfig};
use cpueinsum::{BatchPolicy, BatchStrategy, Exec as BatchExec};
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
fn tiny_scalar_jobs_stay_serial_under_auto() {
    // The shared generic Auto would fan two one-multiply-add jobs onto two
    // lanes; this route keeps today's width_for decision, which stays serial.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(0, 0, 1, 1, 1, 1),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let thread_pool = strided_exec::Pool::builder().threads(2).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let mut exec = Exec::new(&pool, 2).unwrap();
    let route = plan
        .execute_into(&mut exec, 1.0, &[3.0], &[4.0], &mut [0.0; 2])
        .unwrap();
    assert_eq!(route, GroupedRoute::Serial);
}

#[test]
fn forced_strategies_are_honored_or_refused() {
    use core::num::NonZeroUsize;
    // Four 64x64x64 items: big enough that the Auto width_for decision leaves
    // the serial threshold, small enough for a fast test. Offsets keep every
    // job's ranges disjoint.
    let jobs: Vec<_> = (0..4)
        .map(|i| GroupedGemmJob::new(i * 4096, i * 4096, i * 4096, 64, 64, 64))
        .collect();
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let lhs: Vec<f64> = (0..4 * 4096).map(|i| ((i % 7) as f64) - 3.0).collect();
    let rhs: Vec<f64> = (0..4 * 4096).map(|i| ((i % 5) as f64) - 2.0).collect();
    let thread_pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);

    // The reference is the serial route (all item kernels serial, one lane).
    let mut reference = vec![0.0; 4 * 4096];
    assert_eq!(
        plan.execute_into(&mut Exec::serial(), 1.0, &lhs, &rhs, &mut reference)
            .unwrap(),
        GroupedRoute::Serial
    );

    // Auto under four threads: outer lanes.
    let mut auto = vec![0.0; 4 * 4096];
    let mut exec = Exec::new(&pool, 4).unwrap();
    let route = plan
        .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut auto)
        .unwrap();
    assert!(matches!(route, GroupedRoute::Outer { lanes } if lanes > 1));
    close(&auto, &reference, 1e-12);

    // Forced Outer: the per-item kernels are unchanged, so bitwise equal.
    let mut outer = vec![0.0; 4 * 4096];
    let mut exec = Exec::new(&pool, pool.size())
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut outer)
            .unwrap(),
        GroupedRoute::Outer { lanes: 4 }
    );
    assert_eq!(outer, reference);

    // Forced Sequential ignores the budget and is bitwise equal too.
    let mut sequential = vec![0.0; 4 * 4096];
    let mut exec = Exec::new(&pool, pool.size())
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Sequential));
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut sequential)
            .unwrap(),
        GroupedRoute::Serial
    );
    assert_eq!(sequential, reference);

    // Forced Split on the original budget: two lanes, two item threads each.
    let mut split = vec![0.0; 4 * 4096];
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        }));
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut split)
            .unwrap(),
        GroupedRoute::Split {
            lanes: 2,
            item_threads: 2
        }
    );
    // Item threads change the inner decomposition, so compare, do not require bits.
    close(&split, &reference, 1e-12);

    // Forced Inner: one lane, the whole budget per item.
    let mut inner = vec![0.0; 4 * 4096];
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Inner));
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut inner)
            .unwrap(),
        GroupedRoute::Inner { threads: 4 }
    );
    close(&inner, &reference, 1e-12);

    // The serial context carries a policy too, so a forced route at one thread
    // resolves there as well.
    let mut exec = BatchExec::serial().with_batch_policy(BatchPolicy::new(BatchStrategy::Inner));
    let mut one = vec![0.0; 4 * 4096];
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut one)
            .unwrap(),
        GroupedRoute::Serial
    );
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
        .execute_into(&mut Exec::serial(), 1.0, &lhs, &rhs, &mut out)
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
    plan.execute_into_accum(&mut Exec::serial(), 2.0, &lhs, &rhs, 0.5, &mut out)
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
            .execute_into(&mut Exec::serial(), 1.0, &[3.], &[4.], &mut out)
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
        plan.execute_into(&mut Exec::serial(), 1.0, &[0.; 4], &[0.; 13], &mut out)
            .unwrap(),
        GroupedRoute::Serial
    );
    assert_eq!(out, [11.; 9]);

    let none: [GroupedGemmJob; 0] = [];
    let plan = GroupedPlan::<f64>::new(&none, &PlanConfig::default()).unwrap();
    assert_eq!(
        plan.execute_into(&mut Exec::serial(), 1.0, &[], &[], &mut [])
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
    let thread_pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let mut exec = Exec::new(&pool, 2).unwrap();
    let root = exec
        .token()
        .ledger_root()
        .expect("a pooled token is a root");
    let route = plan
        .execute_into_accum(&mut exec, 1.0, &lhs, &rhs, 1.0, &mut out)
        .unwrap();
    assert!(matches!(route, GroupedRoute::Outer { lanes: 2 }));
    // Accumulation detects duplicate job execution, unlike idempotent overwrite.
    assert!(out.iter().all(|&x| x == 65.));
    assert_eq!(strided_exec::ledger::installs(root), 1);

    for budget in [1, 4] {
        out.fill(1.0);
        let mut bounded = Exec::new(&pool, budget).unwrap();
        let root = bounded
            .token()
            .ledger_root()
            .expect("a pooled token is a root");
        let route = plan
            .execute_into_accum(&mut bounded, 1.0, &lhs, &rhs, 1.0, &mut out)
            .unwrap();
        if budget == 1 {
            assert_eq!(route, GroupedRoute::Serial);
            assert_eq!(strided_exec::ledger::installs(root), 0);
        } else {
            assert!(matches!(route, GroupedRoute::Outer { lanes } if lanes > 1 && lanes <= budget));
            assert_eq!(strided_exec::ledger::installs(root), 1);
        }
        assert!(out.iter().all(|&x| x == 65.));
    }
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
    let mut exec = Exec::serial_with_workspace(&arena); // Explicit 1T backend.
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
                .execute_into_accum(&mut exec, alpha, &lhs, &rhs, scale, &mut out)
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
    plan.execute_into(&mut exec, alpha, &lhs, &rhs, &mut old.clone())
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

#[test]
fn empty_jobs_do_not_move_parallel_chunk_boundaries_backwards() {
    use cpueinsum::{BatchPolicy, BatchStrategy};
    // A scalar output at offset 0, then an empty job whose (0, 0) descriptor
    // sorts last. Under forced Outer the first lane takes the scalar and the
    // second the empty job; before the fix the empty job's offset became the
    // first lane's chunk end, so the scalar had no room and the lane panicked.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(0, 0, 0, 0, 1, 1),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let thread_pool = strided_exec::Pool::builder().threads(2).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let mut exec = Exec::new(&pool, 2)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    let mut out = [0.0; 1];
    let route = plan
        .execute_into(&mut exec, 1.0, &[3.0], &[4.0], &mut out)
        .unwrap();
    assert_eq!(route, GroupedRoute::Outer { lanes: 2 });
    assert_eq!(out, [12.0]);
}

#[test]
fn forced_inner_runs_the_child_with_the_granted_width() {
    use cpueinsum::{BatchPolicy, BatchStrategy};
    // A 64^3 packed item is large enough for a team when the whole budget is
    // forced, but its ~26 us of work falls below Auto's 50 us serial threshold:
    // the pre-fix Auto child stayed serial while the route reported four
    // threads. The test isolates the child by comparing against forced
    // Sequential at the same budget.
    let n = 64;
    let jobs = [GroupedGemmJob::new(0, 0, 0, n, n, n)];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::packed()).unwrap();
    let lhs = vec![1.0; n * n];
    let rhs = vec![1.0; n * n];
    let mut out = vec![0.0; n * n];
    let thread_pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);

    // Forced Sequential: one lane, serial child, the pool is never entered.
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Sequential));
    plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out)
        .unwrap();
    let sequential_root = exec
        .token()
        .ledger_root()
        .expect("a pooled token is a root");
    let sequential_uses = strided_exec::ledger::installs(sequential_root);

    // Forced Inner: the item is granted the whole budget, and the child must
    // actually execute at that width. The pre-fix Auto child fell below Auto's
    // 50 us threshold and stayed serial while the route reported four threads.
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Inner));
    let route = plan
        .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out)
        .unwrap();
    assert_eq!(route, GroupedRoute::Inner { threads: 4 });
    let inner_root = exec
        .token()
        .ledger_root()
        .expect("a pooled token is a root");
    let inner_uses = strided_exec::ledger::installs(inner_root);
    assert!(
        inner_uses > sequential_uses,
        "forced Inner did not exercise its granted width: sequential={sequential_uses}, inner={inner_uses}"
    );
}

#[test]
fn split_runs_spmd_routed_children_barrier_free_on_workers() {
    use core::num::NonZeroUsize;
    use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    use cpueinsum::tcontract::{ExecutionRoute, OutputContract, PackedRoute, Partition, Plan};
    use cpueinsum::{BatchPolicy, BatchStrategy};
    // Two packed 64^3 jobs whose budget-2 child wants an SPMD team. That team is
    // available from the coordinator but not from the pool worker each Split lane
    // runs on; the worker's token takes tcontract's barrier-free route instead.
    //
    // The child's route is host- and profile-dependent: the planner's own cost
    // model may prefer a non-SPMD cells grid on some ISAs. Pin the child's
    // static team grid to (2, 1) so the precondition is guaranteed on every
    // host. The planner's default family reads B packed (never direct) on every
    // host, so the pinned grid is not collapsed into a cells grid.
    let config = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: Some((2, 1)),
            align_c_lines: false,
        }),
        ..PlanConfig::packed()
    };
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 64, 64, 64),
        GroupedGemmJob::new(0, 0, 4096, 64, 64, 64),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &config).unwrap();
    let lhs = vec![1.0; 2 * 4096];
    let rhs = vec![1.0; 2 * 4096];
    let mut out = vec![77.0; 2 * 4096];
    let thread_pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        }));

    // Precondition: at budget two the child route really is an SPMD team, so a
    // future route change fails here with a clear message instead of the
    // production check silently never triggering. Rebuild the child plan the
    // grouped plan prepares (the same problem and config, fresh_output cleared)
    // and resolve it on the child context a Split lane runs on.
    let mut child_config = config.clone();
    child_config.fresh_output = false;
    let layout = |dims: &[usize], strides: &[isize]| {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).expect("valid layout"))
            .with_op(Op::Identity)
    };
    let problem = Problem::from_labels(
        DType::F64,
        layout(&[64, 64], &[1, 64]),
        layout(&[64, 64], &[1, 64]),
        CSpec::Output(Op::Identity),
        layout(&[64, 64], &[1, 64]),
        &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
    )
    .expect("valid 64^3 problem");
    let child = Plan::<f64>::new(&problem, &child_config).expect("child plan");
    let child_exec = Exec::new(&pool, 2)
        .expect("budget two")
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Inner));
    match child
        .execution_route(&child_exec, 1.0, OutputContract::Initialized)
        .expect("the child route resolves")
    {
        ExecutionRoute::Packed {
            kind: PackedRoute::Spmd,
            width,
        } => assert_eq!(
            width, 2,
            "the SPMD team must use the budget-two child width"
        ),
        other => panic!("the child route must be an SPMD team at budget 2, got {other:?}"),
    }

    // Each output entry is a 64-term sum of ones, exact in f64. The children run on
    // pool workers, whose token cannot co-schedule a barrier-bearing team, so tcontract
    // takes the barrier-free route for them and the result is still exact.
    let route = plan
        .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out)
        .expect("a forced Split runs an SPMD-routed child on its worker");
    assert!(
        matches!(route, GroupedRoute::Split { lanes: 2, .. }),
        "expected two lanes, got {route:?}"
    );
    assert!(
        out.iter().all(|&x| x == 64.0),
        "the lanes must write every output entry: first {:?}",
        &out[..4]
    );
}
