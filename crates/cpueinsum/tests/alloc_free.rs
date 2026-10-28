//! Rust-owned allocation gates for the raw steady-state paths (#23).
//!
//! The counter arms only around the measured calls and counts Rust's global
//! allocator. A linked C library's own allocations (a BLAS provider, faer's
//! thread pool) are outside it by construction, so every assertion here is
//! "no Rust-owned allocation", never "no allocation anywhere". The measured
//! calls are all serial, so no pool task can be attributed to them.
//!
//! One test drives every gate: the counter is process-global, so a second
//! concurrent test would be counted too.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::{OutputContract, PlanConfig, SliceAccumulationSource};
use cpueinsum::{
    ArenaProvider, BinaryPlan, EinsumPlan, EinsumSpec, Exec, GroupedGemmJob, GroupedPlan, Layout,
    Scratch, Workspace,
};

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

/// `count` disjoint 2x2x2 `f64` jobs: planning one of their child plans
/// allocates nothing, so the grouped container's own allocation shows through.
fn tiny_grouped_jobs(count: usize) -> Vec<GroupedGemmJob> {
    (0..count)
        .map(|i| GroupedGemmJob::new(i * 4, i * 4, i * 4, 2, 2, 2))
        .collect()
}

#[test]
fn raw_steady_state_paths_allocate_no_rust_memory() {
    // G1/G2: a native prepared binary plan through a caller workspace.
    let problem = Problem::from_labels(
        DType::F64,
        operand(&[8, 8], &[1, 8]),
        operand(&[8, 8], &[1, 8]),
        CSpec::Absent,
        operand(&[8, 8], &[1, 8]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let binary = BinaryPlan::<f64>::new(&problem, &PlanConfig::packed()).unwrap();
    assert_eq!(binary.workspace_len(), 0);
    let a: Vec<f64> = (0..64).map(|i| (i % 5) as f64).collect();
    let b: Vec<f64> = (0..64).map(|i| (i % 3) as f64).collect();
    let req = binary.native_workspace_req(&Exec::serial(), OutputContract::Fresh);
    let workspace = Workspace::with_req(&req);
    let mut exec = Exec::serial_with_workspace(workspace.as_provider());
    let mut d = [MaybeUninit::<f64>::uninit(); 64];
    let (_, _) = binary
        .execute_uninit_slices(
            &mut exec,
            1.0,
            (&a, 0),
            (&b, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
        )
        .unwrap();
    let mut d = [MaybeUninit::<f64>::uninit(); 64];
    let allocations = measure(|| {
        let (initialized, _) = binary
            .execute_uninit_slices(
                &mut exec,
                1.0,
                (&a, 0),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Absent,
                (&mut d, 0),
            )
            .unwrap();
        std::hint::black_box(initialized);
    });
    assert_eq!(allocations, 0, "native execute_uninit_slices");

    // G5: an einsum plan from a warm intermediate arena and a prepared native
    // workspace.
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    let cm = Layout::new(&[8, 8], &[1, 8]).unwrap();
    let einsum = EinsumPlan::<f64>::new(&spec, &[cm, cm], cm).unwrap();
    assert_eq!(einsum.workspace_len(), einsum.scratch_len());
    let req = einsum.native_workspace_req(&Exec::serial(), OutputContract::Initialized);
    let workspace = Workspace::with_req(&req);
    let mut exec = Exec::serial_with_workspace(workspace.as_provider());
    let mut scratch = Scratch::with_len(einsum.workspace_len());
    let mut out = vec![0.0; 64];
    {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &[8, 8], &[1, 8], 0).unwrap();
        einsum
            .execute_into(&mut exec, &[av, bv], &mut dv, &mut scratch)
            .unwrap();
    }
    let allocations = measure(|| {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &[8, 8], &[1, 8], 0).unwrap();
        einsum
            .execute_into(&mut exec, &[av, bv], &mut dv, &mut scratch)
            .unwrap();
    });
    assert_eq!(allocations, 0, "einsum execute_into");

    // G7: the grouped serial route collects nothing per call.
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 8, 8, 8),
        GroupedGemmJob::new(0, 0, 64, 8, 8, 8),
    ];
    let grouped = GroupedPlan::<f64>::new(&jobs, &PlanConfig::packed()).unwrap();
    assert_eq!(grouped.workspace_len(), 0);
    let lhs = vec![1.0; 128];
    let rhs = vec![1.0; 128];
    let mut grouped_out = vec![0.0; 128];
    let arena = ArenaProvider::new();
    let mut exec = Exec::serial_with_workspace(&arena);
    grouped
        .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut grouped_out)
        .unwrap();
    let allocations = measure(|| {
        grouped
            .execute_into(&mut exec, 1.0, &lhs, &rhs, &mut grouped_out)
            .unwrap();
    });
    assert_eq!(allocations, 0, "grouped execute_into serial");

    // G8: preparing a grouped plan stays at O(1) allocations in the job count.
    // Each child plan of these tiny f64 jobs allocates nothing, so what is left
    // is the prepared-job container itself; collecting its entries through a
    // `Result<Vec<_>>` instead of pre-sizing regrows it about log2(J) times.
    let mut plan_allocations = Vec::new();
    for job_count in [2usize, 8, 64] {
        let jobs = tiny_grouped_jobs(job_count);
        let allocations = measure(|| {
            let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
            std::hint::black_box(&plan);
        });
        eprintln!("grouped plan allocations at J={job_count}: {allocations}");
        plan_allocations.push(allocations);
    }
    assert_eq!(
        plan_allocations[1], plan_allocations[0],
        "grouped plan allocations: J=8 vs J=2"
    );
    assert_eq!(
        plan_allocations[2], plan_allocations[1],
        "grouped plan allocations: J=64 vs J=8"
    );
}
