//! Preparation-cost gate for grouped plans (#51).
//!
//! `GroupedPlan::new` keeps one numerical child per distinct local geometry and
//! one compact descriptor per job, so its cost is one child plus small per-job
//! metadata rather than one child per job. The counter below is per thread, so
//! work another thread does inside the window is not charged to this call; the
//! measured calls are serial.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use cpueinsum::tcontract::Plan;
use cpueinsum::{GroupedGemmJob, GroupedPlan};

struct Counting;

thread_local! {
    // `const` init keeps the counter itself out of the allocator path that reads it.
    static ARMED: Cell<u32> = const { Cell::new(0) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn count(bytes: usize) {
    // `try_with`: a thread already past TLS teardown can still free memory.
    let _ = ARMED.try_with(|armed| {
        if armed.get() == 1 {
            let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
            let _ = BYTES.try_with(|total| total.set(total.get() + bytes));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer came from this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocation requests and requested bytes of one preparation.
fn prepare(jobs: &[GroupedGemmJob]) -> (usize, usize) {
    let calls = CALLS.with(Cell::get);
    let bytes = BYTES.with(Cell::get);
    ARMED.with(|armed| armed.set(1));
    let plan = GroupedPlan::<f64>::new(jobs, &Default::default()).unwrap();
    ARMED.with(|armed| armed.set(0));
    assert!(plan.required_out_len() >= jobs.iter().map(|j| j.rows * j.cols).max().unwrap_or(0));
    (CALLS.with(Cell::get) - calls, BYTES.with(Cell::get) - bytes)
}

#[test]
fn one_child_per_distinct_geometry_takes_two_allocation_requests() {
    let child = std::mem::size_of::<Plan<f64>>();
    // Compact per-job descriptor: seven words plus the child index.
    let per_job = std::mem::size_of::<usize>() * 8;
    let mut shared = Vec::new();
    let mut distinct = Vec::new();
    for count in [1usize, 5, 17] {
        // Same local geometry, distinct global offsets.
        let jobs: Vec<_> = (0..count)
            .map(|i| GroupedGemmJob::new(0, 0, 4 * i, 2, 2, 2))
            .collect();
        let measured = prepare(&jobs);
        shared.push((count, measured));

        // Every job a different local geometry.
        let mut out = 0;
        let jobs: Vec<_> = (0..count)
            .map(|i| {
                let job = GroupedGemmJob::new(0, 0, out, 2 + i, 2, 2 + i);
                out += (2 + i) * (2 + i);
                job
            })
            .collect();
        let measured = prepare(&jobs);
        distinct.push((count, measured));
    }
    println!("shared geometry: {shared:?}");
    println!("distinct geometry: {distinct:?}");

    let (_, (_, shared_one)) = shared[0];
    for (count, (calls, bytes)) in &shared {
        assert_eq!(*calls, 2, "shared geometry, {count} jobs");
        assert!(
            *bytes <= shared_one + per_job * (count - 1),
            "one child plus compact metadata, {count} jobs: {bytes} bytes"
        );
        // The pre-#51 cost was one full child per job, so the saving grows with
        // J: at 17 shared jobs the request is an order of magnitude smaller.
        if *count >= 17 {
            assert!(
                bytes * 10 < 7_320 * count,
                "an order of magnitude below one child per job, {count} jobs: {bytes} bytes"
            );
        }
    }
    for (count, (calls, bytes)) in &distinct {
        assert_eq!(*calls, 2, "distinct geometry, {count} jobs");
        assert!(
            *bytes >= child * count,
            "a distinct geometry still owns its child, {count} jobs: {bytes} bytes"
        );
    }
    // J = 17 sharing one geometry stays within one extra child of J = 1.
    assert!(
        shared[2].1 .1 < shared[0].1 .1 + child,
        "17 shared jobs: {:?}",
        shared[2].1
    );
}
