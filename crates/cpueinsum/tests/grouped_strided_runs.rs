//! Batched strided runs in `GroupedPlan` (cpueinsum-rs#78).
//!
//! The partition is asserted directly on hand-written job lists, without
//! timing. Execution compares the run route with the per-job route on small
//! integer inputs, so every dtype's result is exact and the comparison is `==`.

use core::fmt::Debug;
use core::mem::MaybeUninit;

use cpueinsum::t4a_desc::gemm::GemmRun;
use cpueinsum::tcontract::api::Op;
use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{
    BatchPolicy, BatchStrategy, Error, Exec, GroupedError, GroupedGemmJob, GroupedOptions,
    GroupedPlan, Scalar,
};
use num_complex::{Complex32, Complex64};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn on() -> GroupedOptions {
    GroupedOptions::default().with_strided_runs(true)
}

fn job(offsets: [usize; 3], rows: usize, contracted: usize, cols: usize) -> GroupedGemmJob {
    GroupedGemmJob::new(offsets[0], offsets[1], offsets[2], rows, contracted, cols)
}

/// `count` jobs of one geometry, offsets `base + i * stride`.
fn strided(
    base: [usize; 3],
    stride: [usize; 3],
    count: usize,
    shape: (usize, usize, usize),
) -> Vec<GroupedGemmJob> {
    (0..count)
        .map(|i| {
            job(
                [0, 1, 2].map(|o| base[o] + i * stride[o]),
                shape.0,
                shape.1,
                shape.2,
            )
        })
        .collect()
}

fn runs(jobs: &[GroupedGemmJob]) -> Vec<GemmRun> {
    GroupedPlan::<f64>::prepare_jobs_with(jobs, on())
        .unwrap()
        .strided_runs()
        .to_vec()
}

fn run(first: usize, len: usize, stride: [isize; 3]) -> GemmRun {
    GemmRun { first, len, stride }
}

// ---------------------------------------------------------------- partition

#[test]
fn partition_of_empty_and_singleton_lists_is_empty() {
    assert_eq!(runs(&[]), []);
    assert_eq!(runs(&[job([0, 0, 0], 2, 2, 2)]), []);
}

#[test]
fn partition_of_one_long_run_is_one_run() {
    let jobs = strided([0, 0, 0], [6, 9, 6], 64, (2, 3, 3));
    assert_eq!(runs(&jobs), [run(0, 64, [6, 9, 6])]);
}

#[test]
fn partition_is_off_by_default() {
    let jobs = strided([0, 0, 0], [4, 4, 4], 8, (2, 2, 2));
    let prepared = GroupedPlan::<f64>::prepare_jobs(&jobs).unwrap();
    assert_eq!(prepared.strided_runs(), []);
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    assert_eq!(plan.strided_runs(), []);
}

#[test]
fn a_stride_break_ends_the_run() {
    // Four jobs advancing by 4, then the lhs stride changes to 8.
    let mut jobs = strided([0, 0, 0], [4, 4, 4], 4, (2, 2, 2));
    jobs.extend(strided([20, 16, 16], [8, 4, 4], 3, (2, 2, 2)));
    // Job 4's lhs step from job 3 is 8, not 4: the first run stops at job 3,
    // and jobs 4..7 advance by [8, 4, 4] from there.
    assert_eq!(runs(&jobs), [run(0, 4, [4, 4, 4]), run(4, 3, [8, 4, 4])]);
}

#[test]
fn a_shape_break_ends_the_run() {
    let mut jobs = strided([0, 0, 0], [4, 4, 4], 3, (2, 2, 2));
    jobs.push(job([12, 12, 12], 2, 2, 1));
    jobs.extend(strided([16, 16, 16], [4, 4, 4], 2, (2, 2, 2)));
    assert_eq!(runs(&jobs), [run(0, 3, [4, 4, 4]), run(4, 2, [4, 4, 4])]);
}

#[test]
fn zero_or_short_output_strides_never_form_a_run() {
    // A zero output stride of non-empty outputs, or one below rows * cols, is
    // an overlap: the plan refuses it before any run is considered.
    for c_stride in [0, 2, 3] {
        let jobs = strided([0, 0, 0], [4, 4, c_stride], 3, (2, 2, 2));
        let error = GroupedPlan::<f64>::prepare_jobs_with(&jobs, on()).unwrap_err();
        assert!(
            matches!(error, Error::Grouped(GroupedError::Overlap { .. })),
            "c stride {c_stride}: {error:?}"
        );
    }
    // Empty outputs may share an offset; they are never batched.
    let empty = strided([0, 0, 0], [4, 4, 0], 3, (0, 2, 2));
    assert_eq!(runs(&empty), []);
    // Descending outputs have a negative output stride: no run.
    let descending: Vec<_> = (0..3)
        .map(|i| job([4 * i, 4 * i, 8 - 4 * i], 2, 2, 2))
        .collect();
    assert_eq!(runs(&descending), []);
}

#[test]
fn interleaved_runs_are_not_batched() {
    // Two strided families interleaved in the job list: no two neighbours
    // advance by the same step twice, so nothing is a run.
    let mut jobs = Vec::new();
    for i in 0..4 {
        jobs.push(job([4 * i, 0, 8 * i], 2, 2, 2));
        jobs.push(job([4 * i, 0, 8 * i + 4], 2, 1, 2));
    }
    assert_eq!(runs(&jobs), []);

    // A run whose output gaps hold another run's outputs: each run's span holds
    // the other's jobs, so neither can own one output slice.
    let mut jobs = strided([0, 0, 0], [4, 0, 8], 3, (2, 2, 2));
    jobs.extend(strided([0, 0, 4], [4, 0, 8], 3, (2, 2, 2)));
    assert_eq!(runs(&jobs), []);
}

#[test]
fn ties_go_to_the_earlier_run_and_the_partition_is_deterministic() {
    // Job 3 could end the stride-1 run or start the stride-8 one; the
    // greedy-from-the-left rule gives it to the earlier run.
    let jobs: Vec<_> = [0, 1, 2, 3, 11, 19]
        .map(|o| job([o, o, o], 1, 1, 1))
        .to_vec();
    let first = runs(&jobs);
    assert_eq!(first, [run(0, 4, [1, 1, 1]), run(4, 2, [8, 8, 8])]);
    assert_eq!(runs(&jobs), first);
}

#[test]
fn shared_and_descending_inputs_form_runs() {
    // B shared by every job (stride 0) and A read at descending offsets.
    let jobs: Vec<_> = (0..4)
        .map(|i| job([12 - 4 * i, 0, 4 * i], 2, 2, 2))
        .collect();
    assert_eq!(runs(&jobs), [run(0, 4, [-4, 0, 4])]);
}

// ---------------------------------------------------------------- execution

trait Num: Scalar + PartialEq + Debug + Copy {
    fn int(re: i32, im: i32) -> Self;
}
impl Num for f32 {
    fn int(re: i32, _: i32) -> Self {
        re as f32
    }
}
impl Num for f64 {
    fn int(re: i32, _: i32) -> Self {
        re as f64
    }
}
impl Num for Complex32 {
    fn int(re: i32, im: i32) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}
impl Num for Complex64 {
    fn int(re: i32, im: i32) -> Self {
        Complex64::new(re as f64, im as f64)
    }
}

fn ints<T: Num>(len: usize, seed: usize) -> Vec<T> {
    (0..len)
        .map(|i| {
            let x = (i * 7 + seed * 3) as i32;
            T::int(x % 5 - 2, x % 3 - 1)
        })
        .collect()
}

/// Run the per-job and the run route on every entry and width, and assert the
/// results equal. Returns the number of batched runs.
fn same_results<T: Num>(jobs: &[GroupedGemmJob], ops: [Op; 3]) -> usize {
    let config = PlanConfig::default();
    let per_job =
        GroupedPlan::<T>::new_with_options(jobs, ops, &config, GroupedOptions::default()).unwrap();
    let runs = GroupedPlan::<T>::new_with_options(jobs, ops, &config, on()).unwrap();
    assert_eq!(per_job.strided_runs(), []);
    let lhs = ints::<T>(per_job.required_lhs_len(), 1);
    let rhs = ints::<T>(per_job.required_rhs_len(), 2);
    let out_len = per_job.required_out_len() + 3;
    let initial = ints::<T>(out_len, 3);

    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let policies = [
        BatchPolicy::new(BatchStrategy::Auto),
        BatchPolicy::new(BatchStrategy::Outer),
    ];
    for (alpha, beta) in [(T::int(2, 1), T::int(0, 0)), (T::int(1, -1), T::int(-1, 2))] {
        // The reference: the per-job route, serial.
        let mut reference = initial.clone();
        per_job
            .execute_into_accum(&mut Exec::serial(), alpha, &lhs, &rhs, beta, &mut reference)
            .unwrap();
        let mut overwrite = initial.clone();
        per_job
            .execute_into(&mut Exec::serial(), alpha, &lhs, &rhs, &mut overwrite)
            .unwrap();
        for width in [1, 4] {
            for policy in policies {
                for plan in [&per_job, &runs] {
                    let mut exec = if width == 1 {
                        Exec::serial()
                    } else {
                        Exec::new(&pool, width).unwrap()
                    }
                    .with_batch_policy(policy);
                    let mut out = initial.clone();
                    plan.execute_into_accum(&mut exec, alpha, &lhs, &rhs, beta, &mut out)
                        .unwrap();
                    assert_eq!(out, reference, "accum, width {width}, {policy:?}");
                    let mut out = initial.clone();
                    plan.execute_into(&mut exec, alpha, &lhs, &rhs, &mut out)
                        .unwrap();
                    assert_eq!(out, overwrite, "overwrite, width {width}, {policy:?}");
                }
            }
        }
        // The uninitialized-output entry, where the jobs cover the output.
        let fresh_len = per_job.required_out_len();
        if per_job.validate_uninit(T::default(), fresh_len).is_ok() {
            for plan in [&per_job, &runs] {
                let mut fresh = vec![MaybeUninit::<T>::uninit(); fresh_len];
                let (got, _) = plan
                    .execute_uninit(
                        &mut Exec::serial(),
                        alpha,
                        &lhs,
                        &rhs,
                        T::default(),
                        &mut fresh,
                    )
                    .unwrap();
                assert_eq!(got, &overwrite[..fresh_len]);
            }
        }
    }
    runs.strided_runs().len()
}

fn all_dtypes(jobs: &[GroupedGemmJob]) -> usize {
    let n = same_results::<f64>(jobs, [Op::Identity; 3]);
    assert_eq!(same_results::<f32>(jobs, [Op::Identity; 3]), n);
    let conj = [Op::Conjugate, Op::Identity, Op::Conjugate];
    assert_eq!(same_results::<Complex32>(jobs, conj), n);
    assert_eq!(same_results::<Complex64>(jobs, [Op::Identity; 3]), n);
    assert_eq!(same_results::<Complex64>(jobs, conj), n);
    n
}

#[test]
fn empty_list_runs_nothing() {
    assert_eq!(all_dtypes(&[]), 0);
}

#[test]
fn one_long_run_matches_the_per_job_route() {
    // Compact outputs: the fresh entry is exercised too.
    let jobs = strided([0, 0, 0], [6, 9, 6], 64, (2, 3, 3));
    assert_eq!(all_dtypes(&jobs), 1);
}

#[test]
fn a_run_split_by_one_irregular_job_matches_the_per_job_route() {
    // Jobs 0..9 advance by 4 except job 4, whose lhs and output sit elsewhere:
    // its step to job 5 has a negative output stride, so it pairs with neither
    // neighbour and keeps the per-job route.
    let mut jobs = strided([0, 0, 0], [4, 4, 4], 9, (2, 2, 2));
    jobs[4].lhs_offset = 100;
    jobs[4].out_offset = 100;
    assert_eq!(
        runs(&jobs),
        [run(0, 4, [4, 4, 4]), run(5, 4, [4, 4, 4])],
        "the irregular job splits the run"
    );
    assert_eq!(all_dtypes(&jobs), 2);
}

#[test]
fn a_constant_stride_run_with_one_geometry_off_by_one_matches() {
    // Offsets advance by a constant stride, but job 3 has one more column.
    let mut jobs = strided([0, 0, 0], [9, 9, 9], 7, (3, 3, 3));
    jobs[3].cols = 2;
    assert_eq!(runs(&jobs), [run(0, 3, [9, 9, 9]), run(4, 3, [9, 9, 9])]);
    assert_eq!(all_dtypes(&jobs), 2);
}

#[test]
fn a_shuffled_job_list_matches_the_per_job_route() {
    let mut rng = ChaCha8Rng::seed_from_u64(78);
    let mut jobs = strided([0, 0, 0], [4, 4, 4], 24, (2, 2, 2));
    jobs.extend(strided([96, 96, 96], [9, 6, 6], 16, (3, 3, 2)));
    jobs.push(job([0, 0, 192], 1, 4, 1));
    for _ in 0..4 {
        jobs.shuffle(&mut rng);
        // A shuffle breaks most runs; the result is the same either way.
        all_dtypes(&jobs);
    }
    // Sorting the shuffled list back restores the two runs.
    jobs.sort_by_key(|job| (job.rows, job.out_offset));
    assert_eq!(all_dtypes(&jobs), 2);
}

#[test]
fn shared_descending_inputs_gapped_outputs_and_empty_reductions_match() {
    // B shared, A descending, outputs with gaps the run leaves untouched.
    let jobs: Vec<_> = (0..5)
        .map(|i| job([16 - 4 * i, 0, 7 * i], 2, 2, 2))
        .collect();
    assert_eq!(runs(&jobs), [run(0, 5, [-4, 0, 7])]);
    assert_eq!(all_dtypes(&jobs), 1);
    // k = 0: the run only scales its outputs.
    let jobs = strided([0, 0, 0], [0, 0, 4], 4, (2, 0, 2));
    assert_eq!(all_dtypes(&jobs), 1);
    // A run next to empty jobs and a singleton.
    let mut jobs = strided([0, 0, 0], [4, 4, 4], 3, (2, 2, 2));
    jobs.push(job([0, 0, 2], 0, 3, 2));
    jobs.push(job([0, 0, 12], 1, 1, 1));
    assert_eq!(all_dtypes(&jobs), 1);
}

#[test]
fn without_runs_the_route_is_bitwise_the_per_job_route() {
    // Random non-integer inputs: with no batched run, enabling the route must
    // not change a single bit, at any width.
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut jobs: Vec<_> = (0..12)
        .map(|i| job([0, 0, 64 * i], 2 + i % 3, 3 + i % 4, 2 + i % 5))
        .collect();
    jobs.shuffle(&mut rng);
    let per_job = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let options = on();
    let routed = GroupedPlan::<f64>::new_with_options(
        &jobs,
        [Op::Identity; 3],
        &PlanConfig::default(),
        options,
    )
    .unwrap();
    assert_eq!(routed.strided_runs(), []);
    let lhs: Vec<f64> = (0..per_job.required_lhs_len()).map(|_| rng.gen()).collect();
    let rhs: Vec<f64> = (0..per_job.required_rhs_len()).map(|_| rng.gen()).collect();
    let initial: Vec<f64> = (0..per_job.required_out_len()).map(|_| rng.gen()).collect();
    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    for width in [1, 4] {
        let mut outs = Vec::new();
        for plan in [&per_job, &routed] {
            let mut exec = Exec::new(&pool, width)
                .unwrap()
                .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
            let mut out = initial.clone();
            plan.execute_into_accum(&mut exec, 0.7, &lhs, &rhs, 0.3, &mut out)
                .unwrap();
            outs.push(out.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
        assert_eq!(outs[0], outs[1], "width {width}");
    }
}

#[test]
fn a_short_buffer_is_refused_before_any_write_with_the_lowest_id() {
    let jobs = strided([0, 0, 0], [4, 4, 4], 4, (2, 2, 2));
    let plan = GroupedPlan::<f64>::new_with_options(
        &jobs,
        [Op::Identity; 3],
        &PlanConfig::default(),
        on(),
    )
    .unwrap();
    assert_eq!(plan.strided_runs().len(), 1);
    let mut out = vec![7.0; 16];
    let error = plan
        .execute_into(&mut Exec::serial(), 1.0, &[1.0; 15], &[1.0; 16], &mut out)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Grouped(GroupedError::Bounds { job: 3, .. })
    ));
    assert_eq!(out, [7.0; 16]);
}

#[test]
fn single_jobs_next_to_batched_runs_are_bitwise_the_per_job_route() {
    // Two batched runs of 2x2x2 jobs around one large single job (three items,
    // so `Auto` on a wide budget grants the runs more than one thread while the
    // single must stay serial), random non-integer inputs: the job outside the
    // runs must give the bits the per-job route gives, at every width; run jobs
    // agree to rounding. (On the Zen 5 development host, width 8 under `Auto`
    // resolves to `Split { lanes: 3, item_threads: 2 }` here.)
    let mut rng = ChaCha8Rng::seed_from_u64(89);
    let mut jobs = strided([0, 0, 0], [4, 4, 4], 4, (2, 2, 2));
    jobs.push(job([0, 0, 16], 128, 128, 128));
    jobs.extend(strided([16, 16, 16 + 128 * 128], [4, 4, 4], 4, (2, 2, 2)));
    let config = PlanConfig::default();
    let per_job = GroupedPlan::<f64>::new(&jobs, &config).unwrap();
    let routed =
        GroupedPlan::<f64>::new_with_options(&jobs, [Op::Identity; 3], &config, on()).unwrap();
    assert_eq!(
        routed.strided_runs(),
        [run(0, 4, [4, 4, 4]), run(5, 4, [4, 4, 4])]
    );
    let lhs: Vec<f64> = (0..per_job.required_lhs_len()).map(|_| rng.gen()).collect();
    let rhs: Vec<f64> = (0..per_job.required_rhs_len()).map(|_| rng.gen()).collect();
    let initial: Vec<f64> = (0..per_job.required_out_len()).map(|_| rng.gen()).collect();
    let singles = [4usize];
    let thread_pool = threadteam::Pool::builder().threads(8).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    for width in [1, 4, 8] {
        for strategy in [BatchStrategy::Auto, BatchStrategy::Outer] {
            let mut outs = Vec::new();
            for plan in [&per_job, &routed] {
                let mut exec = Exec::new(&pool, width)
                    .unwrap()
                    .with_batch_policy(BatchPolicy::new(strategy));
                let mut out = initial.clone();
                plan.execute_into_accum(&mut exec, 0.7, &lhs, &rhs, 0.3, &mut out)
                    .unwrap();
                outs.push(out);
            }
            for &id in &singles {
                let j = jobs[id];
                let range = j.out_offset..j.out_offset + j.rows * j.cols;
                let bits = |out: &[f64]| {
                    out[range.clone()]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&outs[0]),
                    bits(&outs[1]),
                    "job {id}, width {width}, {strategy:?}"
                );
            }
            for (a, b) in outs[0].iter().zip(&outs[1]) {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                    "width {width}, {strategy:?}"
                );
            }
        }
    }
}
