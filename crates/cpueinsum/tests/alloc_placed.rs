//! Allocation bound of the per-job route's work-weighted placement (#80).
//!
//! A warm call on the same context reuses the placement cached in the plan,
//! its tables and the parked allocations of its two lane tables, so the route
//! itself allocates nothing: a placement whose lanes are all one thread wide
//! makes no allocation at all. A child granted a wider lane runs on a team,
//! which may allocate inside tcontract and threadteam (measured: one or two
//! reallocations, varying run to run); that case keeps the per-lane bound
//! `alloc_free_parallel.rs` asserts for the historical `Outer` route. The
//! counter is process-global, and this is the only test in the binary.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan, GroupedRoute};

struct Counting;

static ARMED: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        // SAFETY: the pointer came from this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn measure(body: impl FnOnce()) -> usize {
    ARMED.store(1, Ordering::Relaxed);
    let before = ALLOCS.load(Ordering::Relaxed);
    body();
    let after = ALLOCS.load(Ordering::Relaxed);
    ARMED.store(0, Ordering::Relaxed);
    after - before
}

/// 256 jobs: one `big^3` and 255 `24^3`, on a warm 4-thread context.
/// Returns the warm call's allocations and its route.
fn warm_call(big: usize) -> (usize, GroupedRoute) {
    let mut jobs = vec![GroupedGemmJob::new(0, 0, 0, big, big, big)];
    let base = big * big;
    jobs.extend((0..255).map(|i| GroupedGemmJob::new(0, 0, base + i * 576, 24, 24, 24)));
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::packed()).unwrap();
    let lhs = vec![1.0; plan.required_lhs_len()];
    let rhs = vec![1.0; plan.required_rhs_len()];
    let mut out = vec![0.0; plan.required_out_len()];
    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let mut exec = Exec::new(&pool, 4).unwrap();
    // Warm the pool, the arena, this thread's slot and the plan's placement.
    let mut route = GroupedRoute::Empty;
    for _ in 0..3 {
        route = plan
            .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out)
            .unwrap();
    }
    let allocations = measure(|| {
        let again = plan
            .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out)
            .unwrap();
        assert_eq!(again, route);
    });
    (allocations, route)
}

#[test]
fn a_warm_placed_call_allocates_nothing_of_its_own() {
    // Every job one thread wide: the placed route allocates nothing at all
    // (the historical `Outer` route allocates its lane-chunk vector).
    let (allocations, route) = warm_call(24);
    assert!(
        matches!(route, GroupedRoute::Outer { .. } | GroupedRoute::Serial),
        "{route:?}"
    );
    assert_eq!(allocations, 0, "{route:?}");

    // A dominant job on a wider lane: its child's team may allocate.
    let (allocations, route) = warm_call(192);
    // The exact lanes follow the host class's serial rate; on this route
    // they are uneven or a single wide lane.
    let lanes = match route {
        GroupedRoute::Placed { lanes, .. } | GroupedRoute::Split { lanes, .. } => lanes,
        GroupedRoute::Inner { .. } => 1,
        other => panic!("expected a widened lane, got {other:?}"),
    };
    assert!(
        allocations <= 8 + 2 * lanes,
        "{allocations} allocations for {route:?}"
    );
}
