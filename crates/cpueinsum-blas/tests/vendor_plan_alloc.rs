//! Preparation-cost gate for the vendor grouped and binary plans (#52).
//!
//! A vendor plan prepares its own jobs and keeps the checked job metadata, and
//! builds the tcontract children only when a call actually executes natively.
//! The counter is per thread, so work another thread happens to do inside the
//! window is not charged to the measured call; the measured calls are serial.

extern crate blas_src as _;

use core::mem::MaybeUninit;
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;

use cpueinsum::tcontract::Plan;
use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{BatchPolicy, BatchStrategy, Exec, GroupedGemmJob};
use cpueinsum_blas::{GroupedPlan, GroupedRoute};

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
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        count(layout.size());
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        // SAFETY: the pointer came from this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocation requests and requested bytes of one measured call.
fn measure<T>(body: impl FnOnce() -> T) -> (usize, usize, T) {
    let calls = CALLS.with(Cell::get);
    let bytes = BYTES.with(Cell::get);
    ARMED.with(|armed| armed.set(1));
    let out = body();
    ARMED.with(|armed| armed.set(0));
    (
        CALLS.with(Cell::get) - calls,
        BYTES.with(Cell::get) - bytes,
        out,
    )
}

fn jobs(count: usize) -> Vec<GroupedGemmJob> {
    (0..count)
        .map(|i| GroupedGemmJob::new(0, 0, 16 * i, 4, 4, 4))
        .collect()
}

// #52: the vendor route prepares its own jobs and keeps no native child plan.
#[test]
fn vendor_group_preparation_builds_no_native_child() {
    // Deferring the children adds a `OnceLock`; a prepared plan must stay
    // shareable across threads as it was.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GroupedPlan<f64>>();
    assert_send_sync::<cpueinsum::GroupedPlan<f64>>();

    let child = std::mem::size_of::<Plan<f64>>();
    let config = PlanConfig::default();
    let mut measured = Vec::new();
    for count in [1usize, 5, 32] {
        let jobs = jobs(count);
        let (calls, bytes, plan) = measure(|| GroupedPlan::<f64>::new(&jobs, &config).unwrap());
        println!("vendor grouped jobs={count} calls={calls} bytes={bytes}");
        measured.push((count, calls, bytes));
        std::hint::black_box(&plan);
    }
    let (_, _, one) = measured[0];
    for (count, calls, bytes) in &measured {
        // One request per metadata vector: the job vector, the distinct
        // geometry table and the vendor job vector. Nothing grows per job.
        assert!(*calls <= 3, "{count} jobs: {calls} requests");
        if *count == 1 {
            // One job's whole metadata is far below one native child, so this
            // fails the moment the vendor route prepares numerical children.
            assert!(
                *bytes < child,
                "no native child on the vendor route: {bytes} bytes"
            );
        }
        // One prepared GEMM per job: three operand sides, the batch extents and
        // the caller's spans. A native child is 16x larger.
        assert!(
            *bytes <= one + 512 * (count - 1),
            "compact per-job vendor metadata, {count} jobs: {bytes} bytes"
        );
    }
}

// #52: the native alternative keeps its route and is prepared once.
#[test]
fn native_alternative_is_prepared_once_and_stays_allocation_free() {
    let jobs = jobs(2);
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let a: Vec<f64> = (0..32).map(|i| ((i % 7) as f64) - 3.0).collect();
    let b: Vec<f64> = (0..32).map(|i| ((i % 5) as f64) - 2.0).collect();
    let mut work = vec![0.0; plan.work_len()];

    // A forced lane strategy is one the serial vendor loop cannot honor, so the
    // native alternative runs. The first such call prepares it.
    let mut forced = Exec::serial().with_batch_policy(BatchPolicy::new(BatchStrategy::Inner));
    let mut out = vec![0.0; 32];
    let (first_calls, first_bytes, route) = measure(|| {
        plan.execute_into(&mut forced, 1.0, &a, &b, &mut out, &mut work)
            .unwrap()
    });
    assert!(matches!(route, GroupedRoute::Native(_)), "{route:?}");
    assert!(
        first_calls > 0,
        "the first native call prepares the children"
    );
    assert!(
        first_bytes >= std::mem::size_of::<Plan<f64>>(),
        "that call builds a child plan: {first_bytes} bytes"
    );

    let mut reference = vec![0.0; 32];
    let (warm_calls, _, route) = measure(|| {
        plan.execute_into(&mut forced, 1.0, &a, &b, &mut reference, &mut work)
            .unwrap()
    });
    assert!(matches!(route, GroupedRoute::Native(_)), "{route:?}");
    assert_eq!(warm_calls, 0, "a warm native call prepares nothing");
    assert_eq!(reference, out, "the prepared native route is deterministic");

    // The same plan still serves the vendor loop, bitwise the same.
    let mut vendor = vec![0.0; 32];
    let (vendor_calls, _, route) = measure(|| {
        plan.execute_into(&mut Exec::serial(), 1.0, &a, &b, &mut vendor, &mut work)
            .unwrap()
    });
    assert_eq!(route, GroupedRoute::Vendor { jobs: 2 });
    assert_eq!(vendor_calls, 0, "the vendor loop allocates nothing");
    assert_eq!(vendor, out);

    // Forced `Sequential` is the serial coordinator itself: no native plan.
    let mut sequential =
        Exec::serial().with_batch_policy(BatchPolicy::new(BatchStrategy::Sequential));
    let (sequential_calls, _, route) = measure(|| {
        plan.execute_into(&mut sequential, 1.0, &a, &b, &mut vendor, &mut work)
            .unwrap()
    });
    assert_eq!(route, GroupedRoute::Vendor { jobs: 2 });
    assert_eq!(sequential_calls, 0);

    // A fresh call through the prepared native route reports native and needs no
    // plan either. It is not an allocation gate here: the first fresh native call
    // in a process also pays a one-time host calibration in the faer backend,
    // which is outside the vendor plan's preparation cost.
    let mut fresh = vec![MaybeUninit::uninit(); 32];
    let (fresh, route) = plan
        .execute_uninit(&mut forced, 1.0, &a, &b, 0.0, &mut fresh, &mut work)
        .unwrap();
    assert!(matches!(route, GroupedRoute::Native(_)), "{route:?}");
    assert_eq!(fresh, out);

    // `alpha = 0` reads no input and also takes the native route.
    let mut zeroed = vec![0.0; 32];
    let (zero_calls, _, route) = measure(|| {
        plan.execute_into(&mut Exec::serial(), 0.0, &a, &b, &mut zeroed, &mut work)
            .unwrap()
    });
    assert!(matches!(route, GroupedRoute::Native(_)), "{route:?}");
    assert_eq!(zero_calls, 0);
    assert_eq!(zeroed, vec![0.0; 32]);
}
