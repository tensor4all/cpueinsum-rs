extern crate blas_src as _;

use std::cell::RefCell;
use std::thread::{self, ThreadId};

use cpueinsum::tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::{Exec, Pool};

use crate::{BinaryPlan, BlasScalar, ExecutionRoute};

thread_local! {
    static CALLS: RefCell<Option<Vec<ThreadId>>> = const { RefCell::new(None) };
}

pub(super) fn record_call() {
    CALLS.with(|calls| {
        if let Some(calls) = calls.borrow_mut().as_mut() {
            calls.push(thread::current().id());
        }
    });
}

fn check_inline<T: BlasScalar>(exec: &mut Exec<'_>) {
    let operand = |dims: &[usize], strides: &[isize]| {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
    };
    let problem = Problem::from_labels(
        T::STORAGE,
        operand(&[2, 3, 2], &[1, 2, 6]),
        operand(&[3, 2, 2], &[1, 3, 6]),
        CSpec::Absent,
        operand(&[2, 2, 2], &[1, 2, 4]),
        &Labels::new(&[0, 1, 3], &[1, 2, 3], &[0, 2, 3]),
    )
    .unwrap();
    let plan = BinaryPlan::<T>::new(&problem, &PlanConfig::default()).unwrap();
    let a = [T::ONE; 12];
    let b = [T::ONE; 12];
    let mut out = [T::default(); 8];
    let mut work = vec![T::default(); plan.work_len()];
    CALLS.with(|calls| *calls.borrow_mut() = Some(Vec::new()));
    let route = plan
        .execute_slices_accum(
            exec,
            T::ONE,
            (&a, 0),
            (&b, 0),
            T::default(),
            SliceAccumulationSource::Absent,
            (&mut out, 0),
            &mut work,
        )
        .unwrap();
    assert_eq!(route, ExecutionRoute::Vendor);
    let calls = CALLS.with(|calls| calls.borrow_mut().take().unwrap());
    assert_eq!(calls, [thread::current().id(); 2]);
    assert!(out.iter().all(|&value| value == T::ONE + T::ONE + T::ONE));
}

#[test]
fn actual_cblas_calls_stay_on_coordinator_without_selected_pool_dispatch() {
    let thread_pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = Pool::borrow(&thread_pool);
    assert_eq!(pool.size(), 4);
    for mut exec in [Exec::serial(), Exec::new(&pool, pool.size()).unwrap()] {
        let root = exec.token().ledger_root();
        check_inline::<f32>(&mut exec);
        check_inline::<f64>(&mut exec);
        check_inline::<num_complex::Complex32>(&mut exec);
        check_inline::<num_complex::Complex64>(&mut exec);
        let jobs = [
            cpueinsum::GroupedGemmJob::new(0, 0, 4, 2, 3, 2),
            cpueinsum::GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
        ];
        let plan = crate::GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
        let mut out = [0.0; 8];
        CALLS.with(|calls| *calls.borrow_mut() = Some(Vec::new()));
        assert_eq!(
            plan.execute_into(&mut exec, 1.0, &[1.0; 6], &[1.0; 6], &mut out, &mut [])
                .unwrap(),
            crate::GroupedRoute::Vendor { jobs: 2 }
        );
        let calls = CALLS.with(|calls| calls.borrow_mut().take().unwrap());
        assert_eq!(calls, [thread::current().id(); 2]);
        assert_eq!(out, [3.0; 8]);
        // No pool entry: every call is serial or runs on the coordinator.
        assert_eq!(root.map_or(0, threadteam::ledger::installs), 0);
    }
}
