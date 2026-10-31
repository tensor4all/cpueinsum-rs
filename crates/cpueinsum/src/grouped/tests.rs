use super::*;
use std::cell::Cell;

thread_local! {
    pub(super) static PLAN_CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
    pub(super) static BOUNDS_VISITS: Cell<usize> = const { Cell::new(0) };
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
        assert_eq!(plan.jobs.len(), count);
        for a in &plan.jobs {
            for b in &plan.jobs {
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
        assert_ne!(plan.jobs[0].plan, plan.jobs[1].plan);
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
