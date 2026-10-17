//! Parallel grouped allocation bound (#23 G7 rayon variant).
//!
//! The bound is parameterized by the decided lane count, not a universal
//! constant: this dispatcher spawns one heap-backed rayon scope task per lane,
//! and cold worker storage adds more on the first call. The test warms the pool
//! and the arena before the counted window, then asserts a bound that scales
//! with `lanes`. The counter is process-global, and this is the only test in
//! the binary, so no other test's allocations fall inside the window.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

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

#[test]
fn parallel_grouped_allocation_bound_scales_with_lanes() {
    use cpueinsum::tcontract::PlanConfig;
    // A large job count at a fixed lane budget: an allocation that grew with the
    // batch (one per job) would blow the lane-parameterized bound at 256 jobs
    // even though it passed at four.
    let jobs: Vec<_> = (0..256)
        .map(|i| GroupedGemmJob::new(i * 256, i * 256, i * 256, 16, 16, 16))
        .collect();
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::packed()).unwrap();
    let lhs = vec![1.0; 256 * 256];
    let rhs = vec![1.0; 256 * 256];
    let mut out = vec![0.0; 256 * 256];
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let exec = Exec::rayon(&pool).with_budget(4).unwrap();

    // Warm the pool, the arena and this thread's slot.
    let route = plan.execute_into(&exec, 1.0, &lhs, &rhs, &mut out).unwrap();
    let lanes = match route {
        GroupedRoute::Outer { lanes } | GroupedRoute::Split { lanes, .. } => lanes,
        other => panic!("expected outer lanes, got {other:?}"),
    };

    let allocations = measure(|| {
        plan.execute_into(&exec, 1.0, &lhs, &rhs, &mut out).unwrap();
    });
    // One heap-backed rayon scope task per lane, plus the lane-chunk vector and
    // a few dispatch bookkeeping allocations. The bound does not grow with the
    // 256 jobs.
    assert!(
        allocations <= 8 + 2 * lanes,
        "{allocations} allocations for {lanes} lanes and {} jobs",
        jobs.len()
    );
}
