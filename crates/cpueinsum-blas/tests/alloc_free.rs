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
        operand(&[4, 4], &[1, 4]),
        operand(&[4, 4], b_strides),
        CSpec::Absent,
        operand(&[4, 4], &[1, 4]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

// Planning must not add per-job order allocations on top of the native fallback (#52).
fn vendor_plan_construction_reuses_group_order_storage() {
    let config = PlanConfig::default();
    let p = problem(&[1, 4]);
    let native = measure(|| {
        std::hint::black_box(cpueinsum::BinaryPlan::<f64>::new(&p, &config).unwrap());
    });
    let vendor = measure(|| {
        std::hint::black_box(BinaryPlan::<f64>::new(&p, &config).unwrap());
    });
    println!("plan allocations: direct binary native={native}, vendor={vendor}");
    let direct = (native, vendor);

    let p = problem(&[2, 5]);
    let native = measure(|| {
        std::hint::black_box(cpueinsum::BinaryPlan::<f64>::new(&p, &config).unwrap());
    });
    let vendor = measure(|| {
        std::hint::black_box(BinaryPlan::<f64>::new(&p, &config).unwrap());
    });
    println!("plan allocations: packed binary native={native}, vendor={vendor}");
    let packed = (native, vendor);

    let p = Problem::from_labels(
        DType::F64,
        operand(&[2, 2, 4], &[1, 2, 4]),
        operand(&[4, 2, 2], &[1, 4, 8]),
        CSpec::Absent,
        operand(&[2, 2, 2, 2], &[1, 2, 4, 8]),
        &Labels::new(&[0, 1, 4], &[4, 2, 3], &[0, 1, 2, 3]),
    )
    .unwrap();
    let native = measure(|| {
        std::hint::black_box(cpueinsum::BinaryPlan::<f64>::new(&p, &config).unwrap());
    });
    let vendor = measure(|| {
        std::hint::black_box(BinaryPlan::<f64>::new(&p, &config).unwrap());
    });
    println!("plan allocations: multi-axis binary native={native}, vendor={vendor}");
    let multi_axis = (native, vendor);
    let mut groups = Vec::new();
    for count in [0, 1, 5, 32] {
        let jobs: Vec<_> = (0..count)
            .map(|i| GroupedGemmJob::new(0, 0, i * 16, 4, 4, 4))
            .collect();
        let native = measure(|| {
            std::hint::black_box(cpueinsum::GroupedPlan::<f64>::new(&jobs, &config).unwrap());
        });
        let vendor = measure(|| {
            std::hint::black_box(GroupedPlan::<f64>::new(&jobs, &config).unwrap());
        });
        println!("plan allocations: grouped jobs={count}, native={native}, vendor={vendor}");
        groups.push((count, native, vendor));
    }
    assert_eq!(
        direct.1, direct.0,
        "direct binary layout must allocate nothing"
    );
    assert_eq!(
        packed.1,
        packed.0 + 4,
        "only the input packing metadata allocates"
    );
    assert_eq!(
        multi_axis.1,
        multi_axis.0 + 1,
        "multi-axis groups share one order buffer"
    );
    for (count, native, vendor) in groups {
        assert!(
            native <= 4,
            "one geometry: native metadata, geometry map, compact jobs and one child buffer"
        );
        assert_eq!(
            vendor,
            native + usize::from(count != 0),
            "only the single vendor job buffer may allocate: {count} jobs"
        );
    }
}

#[test]
fn blas_vendor_steady_state_allocates_no_rust_memory() {
    vendor_plan_construction_reuses_group_order_storage();
    let a: Vec<f64> = (0..16).map(|i| (i % 7) as f64).collect();
    // B with strides [2, 5]: no unit stride on either axis, so BLAS can read it
    // neither in place nor transposed and the plan-time CopyPlan packing path
    // runs. Its span is 21, hence 22 elements.
    let packed_b: Vec<f64> = (0..22).map(|i| (i % 5) as f64).collect();

    // G3 contiguous, G4 a genuinely (non-BLAS-readable) packed B.
    for (name, b_strides, expect_packing) in [
        ("contiguous", &[1isize, 4][..], false),
        ("packed", &[2, 5][..], true),
    ] {
        let plan = BinaryPlan::<f64>::new(&problem(b_strides), &PlanConfig::default()).unwrap();
        assert_eq!(plan.work_len() > 0, expect_packing, "work_len for {name}");
        let mut d = vec![0.0; 16];
        let mut work = vec![0.0; plan.work_len()];
        let mut exec = Exec::serial();
        // Warm the plan's own (exec-independent) state.
        plan.execute_slices_accum(
            &mut exec,
            1.0,
            (&a, 0),
            (&packed_b, 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0),
            &mut work,
        )
        .unwrap();
        let allocations = measure(|| {
            plan.execute_slices_accum(
                &mut exec,
                1.0,
                (&a, 0),
                (&packed_b, 0),
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
    let a16: Vec<f64> = (0..256).map(|i| (i % 7) as f64).collect();
    let b16: Vec<f64> = (0..256).map(|i| (i % 5) as f64).collect();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 16, 16, 16),
        GroupedGemmJob::new(0, 0, 256, 16, 16, 16),
    ];
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let mut out = vec![0.0; 512];
    let mut work = vec![0.0; plan.work_len()];
    let mut exec = Exec::serial();
    assert_eq!(
        plan.execute_into(&mut exec, 1.0, &a16, &b16, &mut out, &mut work)
            .unwrap(),
        GroupedRoute::Vendor { jobs: 2 }
    );
    let allocations = measure(|| {
        plan.execute_into(&mut exec, 1.0, &a16, &b16, &mut out, &mut work)
            .unwrap();
    });
    assert_eq!(allocations, 0, "grouped vendor");
}
