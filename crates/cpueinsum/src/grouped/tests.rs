use super::*;
use std::cell::Cell;

thread_local! {
    pub(super) static PLAN_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
    pub(super) static BOUNDS_VISITS: Cell<usize> = const { Cell::new(0) };
    pub(super) static PLACEMENTS: Cell<usize> = const { Cell::new(0) };
}

fn bounds(job: usize) -> Error {
    Error::Grouped(GroupedError::Bounds {
        job,
        operand: GroupedOperand::Lhs,
        end: job + 1,
        available: job,
    })
}

#[test]
fn lowest_job_id_wins_in_any_arrival_order() {
    // Lane children fail in scheduling order, so the reduction must not
    // depend on it.
    for order in [[5usize, 2, 9], [2, 5, 9], [9, 5, 2], [9, 2, 5]] {
        let mut slot = None;
        for job in order {
            keep_lowest(&mut slot, job, bounds(job));
        }
        match slot {
            Some((job, error)) => {
                assert_eq!(job, 2, "arrival order {order:?}");
                assert!(
                    matches!(error, Error::Grouped(GroupedError::Bounds { job: 2, .. })),
                    "arrival order {order:?}"
                );
            }
            None => panic!("a failure was recorded for order {order:?}"),
        }
    }
}

#[test]
fn two_hundred_jobs_construct_exactly_three_children() {
    for (count, geometries) in [(0, 0), (1, 1), (17, 17), (200, 3)] {
        let jobs: Vec<_> = (0..count)
            .map(|id| GroupedGemmJob::new(0, 0, (count - 1 - id) * 64, id % geometries + 2, 2, 2))
            .collect();
        PLAN_CONSTRUCTIONS.with(|counter| counter.set(0));
        let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
        let constructed = PLAN_CONSTRUCTIONS.with(Cell::get);
        println!(
            "jobs={count}, geometries={geometries}, constructed={constructed}, child_capacity={}",
            plan.plans.capacity()
        );
        assert_eq!(constructed, geometries);
        assert_eq!(plan.plans.len(), geometries);
        assert_eq!(plan.plans.capacity(), geometries);
        assert_eq!(plan.meta.jobs.len(), count);
        for a in &plan.meta.jobs {
            for b in &plan.meta.jobs {
                assert_eq!(
                    a.plan == b.plan,
                    jobs[a.meta.id].rows == jobs[b.meta.id].rows
                );
            }
        }
    }
}

#[test]
fn empty_and_unit_reduction_have_distinct_children_in_both_orders() {
    for reverse in [false, true] {
        let mut jobs = [
            GroupedGemmJob::new(0, 0, 0, 2, 0, 2),
            GroupedGemmJob::new(0, 0, 4, 2, 1, 2),
        ];
        if reverse {
            jobs[0].out_offset = 4;
            jobs[1].out_offset = 0;
        }
        let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
        assert_eq!(plan.plans.len(), 2);
        assert_ne!(plan.meta.jobs[0].plan, plan.meta.jobs[1].plan);
        let mut out = [f64::NAN; 8];
        plan.execute_into(&mut Exec::serial(), 1.0, &[1.0, 2.0], &[3.0, 4.0], &mut out)
            .unwrap();
        assert_eq!(&out[jobs[0].out_offset..jobs[0].out_offset + 4], &[0.0; 4]);
        assert_eq!(
            &out[jobs[1].out_offset..jobs[1].out_offset + 4],
            &[3.0, 6.0, 4.0, 8.0]
        );
    }
}

#[test]
fn aggregate_bounds_skip_scan_but_short_buffers_keep_lowest_id() {
    let jobs = [
        GroupedGemmJob::new(2, 2, 2, 1, 1, 1),
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let data = [2.0; 4];
    for len in [3, 4] {
        let mut out = [7.0; 4];
        BOUNDS_VISITS.with(|counter| counter.set(0));
        plan.execute_into(
            &mut Exec::serial(),
            1.0,
            &data[..len],
            &data[..len],
            &mut out[..len],
        )
        .unwrap();
        assert_eq!(BOUNDS_VISITS.with(Cell::get), 0);
        assert_eq!(out, [4.0, 7.0, 4.0, 7.0]);
    }
    for (lhs_len, rhs_len, out_len, operand) in [
        (2, 4, 4, GroupedOperand::Lhs),
        (4, 2, 4, GroupedOperand::Rhs),
        (4, 4, 2, GroupedOperand::Output),
        (0, 0, 0, GroupedOperand::Lhs),
    ] {
        let mut out = [7.0; 4];
        BOUNDS_VISITS.with(|counter| counter.set(0));
        let error = plan
            .execute_into(
                &mut Exec::serial(),
                1.0,
                &data[..lhs_len],
                &data[..rhs_len],
                &mut out[..out_len],
            )
            .unwrap_err();
        assert!(
            matches!(error, Error::Grouped(GroupedError::Bounds {job: 0, operand: got, ..}) if got == operand)
        );
        assert_eq!(BOUNDS_VISITS.with(Cell::get), jobs.len());
        assert_eq!(out, [7.0; 4]);
    }
}

#[test]
fn child_index_conversion_is_checked() {
    assert_eq!(child_index(7, 0).unwrap(), 0);
    assert_eq!(child_index(7, u32::MAX as usize).unwrap(), u32::MAX);
    #[cfg(target_pointer_width = "64")]
    assert!(matches!(
        child_index(7, u32::MAX as usize + 1),
        Err(Error::Grouped(GroupedError::Overflow {
            job: 7,
            what: "child index"
        }))
    ));
}

fn runs_on() -> GroupedOptions {
    GroupedOptions::default().with_strided_runs(true)
}

#[test]
fn a_batched_run_is_one_item_with_one_batched_child() {
    // 64 jobs of one geometry in one run: one geometry child, one run child,
    // and one execution item instead of 64.
    let jobs: Vec<_> = (0..64)
        .map(|i| GroupedGemmJob::new(4 * i, 4 * i, 4 * i, 2, 2, 2))
        .collect();
    PLAN_CONSTRUCTIONS.with(|counter| counter.set(0));
    let plan = GroupedPlan::<f64>::new_with_options(
        &jobs,
        [Op::Identity; 3],
        &PlanConfig::default(),
        runs_on(),
    )
    .unwrap();
    // The geometry child goes through `ChildKey::build`; the run child is
    // planned from its own problem.
    assert_eq!(PLAN_CONSTRUCTIONS.with(Cell::get), 1);
    assert_eq!(plan.plans.len(), 2);
    assert_eq!(plan.meta.item_count(), 1);
    let item = plan.meta.item(0);
    assert_eq!((item.id, item.plan), (0, 1));
    assert_eq!((item.out.start, item.out.len, item.out.origin), (0, 256, 0));
    let problem = plan.plans[1].problem();
    assert_eq!(problem.a().layout().dims(), &[2, 2, 64]);
    assert_eq!(problem.d().layout().strides(), &[1, 2, 4]);
    assert_eq!(plan.meta.max_item_work, 64 * 8);
}

#[test]
fn equal_runs_share_one_child() {
    // Two runs of three 2x2x2 jobs with the same strides, and a third of two:
    // two run children, ordered by key, each naming its lowest first id.
    let family = |base: usize, count: usize| {
        (0..count)
            .map(move |i| GroupedGemmJob::new(base + 4 * i, base + 4 * i, base + 4 * i, 2, 2, 2))
    };
    let separator = |out| GroupedGemmJob::new(0, 0, out, 1, 1, 1);
    let mut jobs: Vec<_> = family(0, 3).collect();
    jobs.push(separator(12));
    jobs.extend(family(16, 3));
    jobs.push(separator(28));
    jobs.extend(family(32, 2));
    let prepared = GroupedPlan::<f64>::prepare_jobs_with(&jobs, runs_on()).unwrap();
    assert_eq!(prepared.strided_runs().len(), 3);
    let children: Vec<_> = prepared
        .run_children
        .iter()
        .map(|c| (c.len, c.stride, c.id))
        .collect();
    assert_eq!(children, [(2, [4, 4, 4], 8), (3, [4, 4, 4], 0)]);
    // Items: run, separator, run, separator, run.
    assert_eq!(prepared.item_count(), 5);
    let plans: Vec<_> = (0..5).map(|i| prepared.item(i).plan).collect();
    let (run2, run3) = (
        prepared.children.len() as u32,
        prepared.children.len() as u32 + 1,
    );
    assert_eq!(plans[0], run3);
    assert_eq!(plans[2], run3);
    assert_eq!(plans[4], run2);
}

#[test]
fn run_windows_cover_descending_and_shared_inputs() {
    assert_eq!(
        {
            let w = Window::run(12, 4, -4, 4);
            (w.start, w.len, w.origin)
        },
        (0, 16, 12)
    );
    let w = Window::run(5, 4, 0, 3);
    assert_eq!((w.start, w.len, w.origin), (5, 4, 0));
    let w = Window::run(7, 0, 9, 3);
    assert_eq!((w.start, w.len, w.origin), (7, 0, 0));
}

#[test]
fn a_forced_run_grant_keeps_single_jobs_serial_in_both_loops() {
    // The width `Auto` grants batched runs depends on the host's cost model,
    // so force it here: two lanes or one, two threads per item, the run grant
    // policy. The single job between the runs must give the per-job route's
    // bits in the serial loop and in the lane loop; run jobs agree to rounding.
    let run = |base: usize| {
        (0..4).map(move |i| GroupedGemmJob::new(base + 4 * i, base + 4 * i, base + 4 * i, 2, 2, 2))
    };
    let mut jobs: Vec<_> = run(0).collect();
    jobs.push(GroupedGemmJob::new(0, 0, 16, 6, 5, 4));
    jobs.extend(run(40));
    let per_job = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let routed = GroupedPlan::<f64>::new_with_options(
        &jobs,
        [Op::Identity; 3],
        &PlanConfig::default(),
        runs_on(),
    )
    .unwrap();
    assert_eq!(routed.strided_runs().len(), 2);
    let value = |i: usize| ((i * 37 % 101) as f64).sqrt() - 4.3;
    let lhs: Vec<f64> = (0..per_job.required_lhs_len()).map(value).collect();
    let rhs: Vec<f64> = (0..per_job.required_rhs_len())
        .map(|i| value(i + 7))
        .collect();
    let initial: Vec<f64> = (0..per_job.required_out_len())
        .map(|i| value(i + 3))
        .collect();
    let mut reference = initial.clone();
    per_job
        .execute_into_accum(&mut Exec::serial(), 0.7, &lhs, &rhs, 0.3, &mut reference)
        .unwrap();
    let single = 16..16 + 24;
    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = crate::Pool::borrow(&thread_pool);
    for lanes in [1, 2] {
        let widths = Widths {
            lanes,
            item_threads: 2,
            item_policy: Some(BatchPolicy::default()),
            placed: None,
        };
        assert!(widths.single_context(&Exec::serial()).is_some());
        let mut exec = Exec::new(&pool, 4).unwrap();
        let mut out = initial.clone();
        let (route, _) = routed
            .run_items(
                &mut exec,
                widths,
                (0.7, OutputContract::Initialized),
                &lhs,
                &rhs,
                &mut out,
                |item, exec| run_item(item, exec, 0.7, 0.3, SliceAccumulationSource::Output),
            )
            .unwrap();
        assert_eq!(route, grouped_route(lanes, 2));
        let bits = |v: &[f64]| {
            v[single.clone()]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&out), bits(&reference), "{lanes} lanes");
        for (a, b) in out.iter().zip(&reference) {
            assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{lanes} lanes");
        }
    }
}

#[test]
fn a_warm_call_reuses_its_placement_until_the_context_changes() {
    // One dominant job and three small ones: wide enough that `Auto` places.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 192, 192, 192),
        GroupedGemmJob::new(0, 0, 192 * 192, 8, 8, 8),
        GroupedGemmJob::new(0, 0, 192 * 192 + 64, 8, 8, 8),
        GroupedGemmJob::new(0, 0, 192 * 192 + 128, 8, 8, 8),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::packed()).unwrap();
    let (lhs, rhs) = (vec![1.0; 192 * 192], vec![1.0; 192 * 192]);
    let mut out = vec![0.0; plan.required_out_len()];
    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = crate::Pool::borrow(&thread_pool);
    let placements = |exec: &mut Exec<'_>, out: &mut [f64], alpha: f64| {
        PLACEMENTS.with(|count| count.set(0));
        plan.execute_into(exec, alpha, &lhs, &rhs, out).unwrap();
        PLACEMENTS.with(Cell::get)
    };
    let mut exec = Exec::new(&pool, 4).unwrap();
    assert_eq!(placements(&mut exec, &mut out, 1.0), 1);
    assert_eq!(placements(&mut exec, &mut out, 1.0), 0);
    assert_eq!(placements(&mut exec, &mut out, 2.0), 0);
    // A zero alpha makes every child output-only: another placement input.
    assert_eq!(placements(&mut exec, &mut out, 0.0), 1);
    let mut narrower = Exec::new(&pool, 3).unwrap();
    assert_eq!(placements(&mut narrower, &mut out, 1.0), 1);
    assert_eq!(placements(&mut narrower, &mut out, 1.0), 0);
}
