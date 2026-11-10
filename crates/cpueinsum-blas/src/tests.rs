//! Unit tests for the crate root: the vendor call region and its placement.

use std::time::Duration;

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{BatchPolicy, EinsumPlan, EinsumSpec, Exec, Layout, Scratch};

/// CPUs the calling thread may run on, read from `/proc` (0 where the platform
/// has no CPU set). `thread-self` is this thread, not the process's leader.
pub(crate) fn cpus_allowed() -> usize {
    let Ok(status) = std::fs::read_to_string("/proc/thread-self/status") else {
        return 0;
    };
    let Some(list) = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
    else {
        return 0;
    };
    list.split(',').fold(0, |count, part| {
        let part = part.trim();
        match part.split_once('-') {
            Some((lo, hi)) => match (lo.parse::<usize>(), hi.parse::<usize>()) {
                (Ok(lo), Ok(hi)) => count + hi.saturating_sub(lo) + 1,
                _ => count,
            },
            None => count + usize::from(part.parse::<usize>().is_ok()),
        }
    })
}

/// Run one vendor route and assert that every GEMM it issued saw more than one
/// CPU, i.e. ran inside the widened region of [`crate::vendor`].
fn assert_gemms_wide(route: &str, call: impl FnOnce()) {
    crate::ffi::tests::watch_cpus();
    call();
    let seen = crate::ffi::tests::take_cpus();
    assert!(!seen.is_empty(), "{route}: no GEMM was recorded");
    assert!(
        seen.iter().all(|&cpus| cpus > 1),
        "{route}: a GEMM saw {seen:?} CPUs, not the whole set"
    );
}

/// All three vendor loops — the N-ary backend step, the prepared `BinaryPlan`
/// and the grouped `GroupedPlan` — run on the whole CPU set of a placed team
/// leader (`TeamExec::external`), and each leaves the leader on `{S[0]}` again.
///
/// Skipped where the platform has no CPU set or the process has fewer than four
/// CPUs, which is where threadteam places nothing and every call is plain; the
/// skip prints a line, because a skipped run is not evidence.
#[test]
fn gemm_from_a_team_leader_sees_the_whole_set() {
    let set = cpus_allowed();
    if set < 4 {
        eprintln!("skipped: {set} CPUs available, a team of four is not placed");
        return;
    }
    let pool = threadteam::Pool::builder()
        .threads(4)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .expect("a pool of four");
    let team = pool.into_team(4).expect("four idle members");
    let mut exec = Exec::from_token(team.exec(4), None, BatchPolicy::default());
    assert_eq!(
        cpus_allowed(),
        1,
        "the team leader is placed before the calls"
    );

    // The N-ary plan's backend step (`run`).
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    let cm = Layout::new(&[4, 4], &[1, 4]).unwrap();
    let plan = EinsumPlan::<f64, crate::Blas>::with_backend(
        crate::Blas::with_min_macs(0),
        &spec,
        &[cm, cm],
        cm,
    )
    .unwrap();
    assert_eq!(plan.backend_steps(), 1, "the GEMM is the backend step");
    let (a, b) = ([2.0f64; 16], [3.0f64; 16]);
    let mut out = [0.0f64; 16];
    let mut scratch = Scratch::new();
    let views: Vec<StridedView<'_, f64>> = [&a, &b]
        .iter()
        .map(|x| StridedView::new(x.as_slice(), &[4, 4], &[1, 4], 0).unwrap())
        .collect();
    let mut out_view = StridedViewMut::new(&mut out, &[4, 4], &[1, 4], 0).unwrap();
    assert_gemms_wide("backend step", || {
        plan.execute_into(&mut exec, &views, &mut out_view, &mut scratch)
            .unwrap()
    });
    assert_eq!(cpus_allowed(), 1, "the leader is placed again after `run`");
    assert_eq!(out[0], 4.0 * 2.0 * 3.0);

    // The prepared binary adapter, which reaches `VendorPlan::run_vendor`.
    assert_gemms_wide("prepared binary", || {
        crate::ffi::tests::check_inline::<f64>(&mut exec)
    });
    assert_eq!(
        cpus_allowed(),
        1,
        "the leader is placed again after the prepared call"
    );

    // The grouped adapter, whose jobs reach the same `VendorPlan::run_vendor`.
    let jobs = [
        cpueinsum::GroupedGemmJob::new(0, 0, 4, 2, 3, 2),
        cpueinsum::GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
    ];
    let grouped =
        crate::GroupedPlan::<f64>::new(&jobs, &cpueinsum::tcontract::PlanConfig::default())
            .unwrap();
    let mut out = [0.0f64; 8];
    assert_gemms_wide("grouped", || {
        assert_eq!(
            grouped
                .execute_into(&mut exec, 1.0, &[1.0; 6], &[1.0; 6], &mut out, &mut [])
                .unwrap(),
            crate::GroupedRoute::Vendor { jobs: 2 }
        )
    });
    assert_eq!(
        cpus_allowed(),
        1,
        "the leader is placed again after the grouped call"
    );
    assert_eq!(out, [3.0; 8]);
}
