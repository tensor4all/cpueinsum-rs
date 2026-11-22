//! Caller-supplied grouped child plans (cpueinsum-rs#79).
//!
//! Inputs are small integers, so every f64 product is exact and results are
//! compared bit for bit against a naive column-major GEMM and against the
//! built-in (rebuilt-child) path.

use std::collections::HashMap;
use std::sync::Arc;

use cpueinsum::tcontract::api::Op;
use cpueinsum::tcontract::{Plan, PlanConfig};
use cpueinsum::{
    BatchPolicy, BatchStrategy, ChildKey, Error, Exec, GroupedError, GroupedGemmJob, GroupedPlan,
    GroupedRoute,
};

type Cache = HashMap<ChildKey, Arc<Plan<f64>>>;

fn gemm(a: &[f64], b: &[f64], rows: usize, contracted: usize, cols: usize) -> Vec<f64> {
    let mut d = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            d[i + rows * j] =
                (0..contracted).fold(0.0, |acc, l| acc + a[i + rows * l] * b[l + contracted * j]);
        }
    }
    d
}

/// `count` jobs over `geometries` distinct shapes, outputs packed in reverse
/// job order so stored (output) order differs from job order.
fn jobs(count: usize, geometries: usize) -> Vec<GroupedGemmJob> {
    let shape = |id: usize| {
        let g = id % geometries;
        (g + 1, 2 + g % 2, 2)
    };
    let total: usize = (0..count).map(|id| shape(id).0 * shape(id).2).sum();
    let mut out = total;
    (0..count)
        .map(|id| {
            let (m, k, n) = shape(id);
            out -= m * n;
            GroupedGemmJob::new(id % 5, 2 * (id % 3), out, m, k, n)
        })
        .collect()
}

fn inputs() -> (Vec<f64>, Vec<f64>) {
    let lhs = (0..64).map(|i| (i % 7) as f64 - 3.0).collect();
    let rhs = (0..64).map(|i| (i % 5) as f64 - 2.0).collect();
    (lhs, rhs)
}

fn cached(
    jobs: &[GroupedGemmJob],
    cache: &mut Cache,
    keys: &mut Vec<ChildKey>,
    misses: &mut usize,
) -> GroupedPlan<f64> {
    GroupedPlan::from_prepared_with(
        GroupedPlan::<f64>::prepare_jobs(jobs).unwrap(),
        [Op::Identity; 3],
        &PlanConfig::default(),
        |key| {
            keys.push(key.clone());
            if let Some(plan) = cache.get(key) {
                return Ok(Arc::clone(plan));
            }
            *misses += 1;
            let plan = Arc::new(key.build::<f64>()?);
            cache.insert(key.clone(), Arc::clone(&plan));
            Ok(plan)
        },
    )
    .unwrap()
}

fn run(plan: &GroupedPlan<f64>, exec: &mut Exec<'_>) -> (Vec<f64>, GroupedRoute) {
    let (lhs, rhs) = inputs();
    let mut out = vec![f64::NAN; plan.required_out_len()];
    let route = plan.execute_into(exec, 1.0, &lhs, &rhs, &mut out).unwrap();
    (out, route)
}

fn naive(jobs: &[GroupedGemmJob]) -> Vec<f64> {
    let (lhs, rhs) = inputs();
    let len = jobs
        .iter()
        .map(|j| j.out_offset + j.rows * j.cols)
        .max()
        .unwrap_or(0);
    let mut out = vec![f64::NAN; len];
    for j in jobs {
        let a = &lhs[j.lhs_offset..j.lhs_offset + j.rows * j.contracted];
        let b = &rhs[j.rhs_offset..j.rhs_offset + j.contracted * j.cols];
        let d = gemm(a, b, j.rows, j.contracted, j.cols);
        out[j.out_offset..j.out_offset + d.len()].copy_from_slice(&d);
    }
    out
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn closure_is_called_once_per_distinct_geometry_in_child_index_order() {
    let jobs = jobs(200, 3);
    let (mut cache, mut keys, mut misses) = (Cache::new(), Vec::new(), 0);
    let plan = cached(&jobs, &mut cache, &mut keys, &mut misses);
    assert_eq!((keys.len(), misses), (3, 3));
    let shapes: Vec<_> = keys.iter().map(ChildKey::shape).collect();
    let mut sorted = shapes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(shapes, sorted, "ascending geometry order, no repeats");
    for key in &keys {
        let (m, k, _) = key.shape();
        assert_eq!(key.leading_dims(), (m.max(1), k.max(1), m.max(1)));
        assert_eq!(key.ops(), [Op::Identity; 3]);
        assert!(!key.config().fresh_output);
    }
    assert_eq!(
        bits(&run(&plan, &mut Exec::serial()).0),
        bits(&naive(&jobs))
    );
}

#[test]
fn supplied_children_match_the_rebuilt_path_bitwise() {
    let jobs = jobs(40, 4);
    let rebuilt = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let (mut cache, mut keys, mut misses) = (Cache::new(), Vec::new(), 0);
    let supplied = cached(&jobs, &mut cache, &mut keys, &mut misses);
    let reference = naive(&jobs);

    let (serial_rebuilt, _) = run(&rebuilt, &mut Exec::serial());
    let (serial_supplied, route) = run(&supplied, &mut Exec::serial());
    assert_eq!(route, GroupedRoute::Serial);
    assert_eq!(bits(&serial_rebuilt), bits(&reference));
    assert_eq!(bits(&serial_supplied), bits(&reference));

    // Outer lanes run the same item kernels, so the bits do not depend on the
    // thread count either.
    let thread_pool = threadteam::Pool::builder().threads(3).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    for threads in [2, 3] {
        let mut exec = Exec::new(&pool, threads)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
        let (lanes, route) = run(&supplied, &mut exec);
        assert_eq!(route, GroupedRoute::Outer { lanes: threads });
        assert_eq!(bits(&lanes), bits(&reference), "{threads} lanes");
    }
}

#[test]
fn one_cached_child_serves_two_plans_and_repeated_calls() {
    let first_jobs = jobs(12, 3);
    // A different job list (other offsets and counts) over the same shapes.
    let second_jobs = jobs(7, 3);
    let (mut cache, mut keys, mut misses) = (Cache::new(), Vec::new(), 0);
    let first = cached(&first_jobs, &mut cache, &mut keys, &mut misses);
    let built = misses;
    let held: Vec<_> = cache.values().map(Arc::clone).collect();
    let second = cached(&second_jobs, &mut cache, &mut keys, &mut misses);
    assert_eq!(built, 3);
    assert_eq!(misses, built, "the second plan builds no child");
    // Each cached child is now held by the cache, `held`, and both plans.
    for plan in &held {
        assert_eq!(Arc::strong_count(plan), 4);
    }

    for (plan, jobs) in [(&first, &first_jobs), (&second, &second_jobs)] {
        let reference = bits(&naive(jobs));
        let (once, _) = run(plan, &mut Exec::serial());
        let (twice, _) = run(plan, &mut Exec::serial());
        assert_eq!(bits(&once), reference);
        assert_eq!(bits(&twice), reference);
    }
    drop(first);
    drop(second);
    for plan in &held {
        assert_eq!(Arc::strong_count(plan), 2, "plans release their children");
    }
}

#[test]
fn same_shape_set_gives_the_same_keys_and_assignment() {
    let jobs = jobs(30, 3);
    // Same jobs, reversed and rotated: the shape set is unchanged.
    let mut reversed = jobs.clone();
    reversed.reverse();
    let mut rotated = jobs.clone();
    rotated.rotate_left(11);
    let mut sequences = Vec::new();
    for list in [&jobs, &jobs, &reversed, &rotated] {
        let (mut cache, mut keys, mut misses) = (Cache::new(), Vec::new(), 0);
        let plan = cached(list, &mut cache, &mut keys, &mut misses);
        // Same assignment: every job lands on its own geometry's child, so the
        // output equals the naive per-job result whatever the list order.
        assert_eq!(bits(&run(&plan, &mut Exec::serial()).0), bits(&naive(list)));
        sequences.push(keys);
    }
    for keys in &sequences[1..] {
        assert_eq!(keys, &sequences[0]);
    }
}

/// The child the built-in path makes for one job's geometry under `ops`.
fn child_of(job: GroupedGemmJob, ops: [Op; 3]) -> Arc<Plan<f64>> {
    let mut child = None;
    GroupedPlan::<f64>::from_prepared_with(
        GroupedPlan::<f64>::prepare_jobs(&[job]).unwrap(),
        ops,
        &PlanConfig::default(),
        |key| {
            let plan = Arc::new(key.build::<f64>()?);
            child = Some(Arc::clone(&plan));
            Ok(plan)
        },
    )
    .unwrap();
    child.unwrap()
}

#[test]
fn a_child_for_another_problem_is_rejected_with_the_lowest_job_id() {
    // Geometry (2, 2, 2) is used by jobs 3 and 1; job 1 is the lowest.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(0, 0, 1, 2, 2, 2),
        GroupedGemmJob::new(0, 0, 5, 1, 1, 1),
        GroupedGemmJob::new(0, 0, 9, 2, 2, 2),
    ];
    let identity = [Op::Identity; 3];
    let other_shape = child_of(GroupedGemmJob::new(0, 0, 0, 3, 2, 2), identity);
    let other_ops = child_of(
        GroupedGemmJob::new(0, 0, 0, 2, 2, 2),
        [Op::Conjugate, Op::Identity, Op::Identity],
    );
    for (label, wrong) in [("other shape", other_shape), ("other ops", other_ops)] {
        let mut written = vec![7.0; 13];
        let result = GroupedPlan::<f64>::from_prepared_with(
            GroupedPlan::<f64>::prepare_jobs(&jobs).unwrap(),
            identity,
            &PlanConfig::default(),
            |key| {
                if key.shape() == (2, 2, 2) {
                    Ok(Arc::clone(&wrong))
                } else {
                    key.build::<f64>().map(Arc::new)
                }
            },
        )
        .and_then(|plan| {
            plan.execute_into(&mut Exec::serial(), 1.0, &[1.0; 4], &[1.0; 4], &mut written)
        });
        assert!(
            matches!(
                result,
                Err(Error::Grouped(GroupedError::ChildMismatch { job: 1 }))
            ),
            "{label}: {result:?}"
        );
        assert_eq!(written, [7.0; 13], "{label}: nothing ran");
    }
}

#[test]
fn a_closure_error_names_the_lowest_job_id_and_stops_construction() {
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(0, 0, 9, 3, 1, 1),
        GroupedGemmJob::new(0, 0, 1, 2, 2, 2),
        GroupedGemmJob::new(0, 0, 5, 2, 2, 2),
    ];
    let mut calls = Vec::new();
    let result = GroupedPlan::<f64>::from_prepared_with(
        GroupedPlan::<f64>::prepare_jobs(&jobs).unwrap(),
        [Op::Identity; 3],
        &PlanConfig::default(),
        |key| {
            calls.push(key.shape());
            if key.shape() == (2, 2, 2) {
                // A dtype mismatch is a real tcontract error to propagate.
                key.build::<f32>()
                    .map(|_| unreachable!("f32 plan for an f64 key"))
            } else {
                key.build::<f64>().map(Arc::new)
            }
        },
    );
    assert!(
        matches!(result, Err(Error::Contract { step: 2, .. })),
        "{result:?}"
    );
    // Child-index order is (1,1,1), (2,2,2), (3,1,1): the third is never asked.
    assert_eq!(calls, [(1, 1, 1), (2, 2, 2)]);
}

#[test]
fn keys_hash_and_compare_by_everything_the_child_depends_on() {
    use std::collections::HashSet;
    let collect = |ops: [Op; 3], config: &PlanConfig| {
        let mut keys = Vec::new();
        GroupedPlan::<f64>::from_prepared_with(
            GroupedPlan::<f64>::prepare_jobs(&[
                GroupedGemmJob::new(0, 0, 0, 2, 0, 2),
                GroupedGemmJob::new(0, 0, 4, 2, 1, 2),
            ])
            .unwrap(),
            ops,
            config,
            |key| {
                keys.push(key.clone());
                key.build::<f64>().map(Arc::new)
            },
        )
        .unwrap();
        keys
    };
    let base = collect([Op::Identity; 3], &PlanConfig::default());
    // Zero and one contracted extents are different children.
    assert_ne!(base[0], base[1]);
    let fresh = PlanConfig {
        fresh_output: true,
        ..PlanConfig::default()
    };
    // `fresh_output` does not affect the child, so it is normalized away.
    assert_eq!(collect([Op::Identity; 3], &fresh), base);
    assert_ne!(collect([Op::Identity; 3], &PlanConfig::packed()), base);
    assert_ne!(
        collect(
            [Op::Identity, Op::Identity, Op::Conjugate],
            &PlanConfig::default()
        ),
        base
    );
    let set: HashSet<_> = base
        .iter()
        .cloned()
        .chain(collect([Op::Identity; 3], &fresh))
        .collect();
    assert_eq!(set.len(), 2);
}
