//! Rust-owned allocation gates for the BLAS vendor steady state (#23, G3/G4/G8).
//!
//! The counter sees Rust's global allocator only; the linked CBLAS provider's
//! own allocations are outside it and are deliberately not promised here. The
//! measured calls are serial.

extern crate blas_src as _;

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::{Exec, GroupedGemmJob};
use cpueinsum_blas::{BinaryPlan, GroupedPlan, GroupedRoute};

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

fn operand(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

fn problem(b_strides: &[isize]) -> Problem {
    Problem::from_labels(
        DType::F64,
        operand(&[16, 16], &[1, 16]),
        operand(&[16, 16], b_strides),
        CSpec::Absent,
        operand(&[16, 16], &[1, 16]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

#[test]
fn blas_vendor_steady_state_allocates_no_rust_memory() {
    let a: Vec<f64> = (0..256).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..256).map(|i| (i % 5) as f64).collect();

    // G3 contiguous, G4 a packed (row-major) B whose copy-back-free step packs.
    for (name, b_strides) in [("contiguous", &[1isize, 16][..]), ("packed", &[16, 1][..])] {
        let plan = BinaryPlan::<f64>::new(&problem(b_strides), &PlanConfig::default()).unwrap();
        let mut d = vec![0.0; 256];
        let mut work = vec![0.0; plan.work_len()];
        let exec = Exec::serial();
        // Warm the plan's own (exec-independent) state.
        plan.execute_slices_accum(
            &exec,
            1.0,
            (&a, 0),
            (&b, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut work,
        )
        .unwrap();
        let allocations = measure(|| {
            plan.execute_slices_accum(
                &exec,
                1.0,
                (&a, 0),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Absent,
                (&mut d, 0),
                &mut work,
            )
            .unwrap();
        });
        assert_eq!(allocations, 0, "binary vendor {name}");
    }

    // G8: the grouped vendor coordinator loop.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 16, 16, 16),
        GroupedGemmJob::new(0, 0, 256, 16, 16, 16),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let mut out = vec![0.0; 512];
    let mut work = vec![0.0; plan.work_len()];
    let exec = Exec::serial();
    assert_eq!(
        plan.execute_into(&exec, 1.0, &a, &b, &mut out, &mut work)
            .unwrap(),
        GroupedRoute::Vendor { jobs: 2 }
    );
    let allocations = measure(|| {
        plan.execute_into(&exec, 1.0, &a, &b, &mut out, &mut work)
            .unwrap();
    });
    assert_eq!(allocations, 0, "grouped vendor");
}
