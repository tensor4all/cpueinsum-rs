extern crate blas_src as _;

use std::cell::{Cell, RefCell};
use std::thread::{self, ThreadId};

use cpueinsum::tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::tcontract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::{Exec, Pool};

use crate::{BinaryPlan, BlasScalar, ExecutionRoute};

thread_local! {
    static CALLS: RefCell<Option<Vec<ThreadId>>> = const { RefCell::new(None) };
    static GEMVS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_call() {
    CALLS.with(|calls| {
        if let Some(calls) = calls.borrow_mut().as_mut() {
            calls.push(thread::current().id());
        }
    });
}

pub(super) fn record_gemv() {
    GEMVS.with(|gemvs| gemvs.set(gemvs.get() + 1));
}

/// How many GEMV calls this thread has made since the last call.
fn take_gemvs() -> usize {
    GEMVS.with(|gemvs| gemvs.replace(0))
}

/// One column-major GEMM straight to `cblas_zgemm`: the route a thin
/// `Complex64` GEMM would otherwise take.
unsafe fn zgemm(
    g: &crate::ffi::Gemm,
    (a, b, c): (
        *const num_complex::Complex64,
        *const num_complex::Complex64,
        *mut num_complex::Complex64,
    ),
    (alpha, beta): (num_complex::Complex64, num_complex::Complex64),
) {
    let alpha = [alpha.re, alpha.im];
    let beta = [beta.re, beta.im];
    unsafe {
        cblas_sys::cblas_zgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            g.ta.cblas(),
            g.tb.cblas(),
            g.m,
            g.n,
            g.k,
            alpha.as_ptr().cast(),
            a.cast(),
            g.lda,
            b.cast(),
            g.ldb,
            beta.as_ptr().cast(),
            c.cast(),
            g.ldc,
        )
    }
}

/// Thin `Complex64` GEMMs agree with the GEMM they replace, for every input
/// reading the reroute accepts.
#[test]
fn thin_complex64_matches_zgemm() {
    use crate::ffi::{Gemm, Trans};
    use num_complex::Complex64;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(67);
    let value =
        |rng: &mut ChaCha8Rng| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
    for ta in [Trans::No, Trans::T] {
        for tb in [Trans::No, Trans::T] {
            for m in [1i32, 2] {
                for (k, n, pad) in [(1i32, 1i32, 0i32), (3, 5, 2), (7, 4, 0), (4, 7, 3)] {
                    // A is m x k, stored as itself (lda x k) or transposed (lda x k) rows.
                    let (acols, lda) = match ta {
                        Trans::No => (k, m + pad),
                        _ => (m, k + pad),
                    };
                    let (bcols, ldb) = match tb {
                        Trans::No => (n, k + pad),
                        _ => (k, n + pad),
                    };
                    let ldc = m + pad;
                    let a: Vec<Complex64> = (0..lda * acols).map(|_| value(&mut rng)).collect();
                    let b: Vec<Complex64> = (0..ldb * bcols).map(|_| value(&mut rng)).collect();
                    let init: Vec<Complex64> = (0..ldc * n).map(|_| value(&mut rng)).collect();
                    let alpha = value(&mut rng);
                    let g = Gemm {
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    };
                    for beta in [Complex64::new(0.0, 0.0), value(&mut rng)] {
                        let mut expected = init.clone();
                        let mut got = init.clone();
                        take_gemvs();
                        // SAFETY: every buffer holds the dimensions `g` names,
                        // and `got` is a disjoint copy of `expected`.
                        unsafe {
                            zgemm(
                                &g,
                                (a.as_ptr(), b.as_ptr(), expected.as_mut_ptr()),
                                (alpha, beta),
                            );
                            Complex64::gemm(
                                &g,
                                a.as_ptr(),
                                b.as_ptr(),
                                got.as_mut_ptr(),
                                alpha,
                                beta,
                            );
                        }
                        assert_eq!(take_gemvs(), m as usize, "{g:?} {beta:?}");
                        for (i, (&got, &want)) in got.iter().zip(&expected).enumerate() {
                            let err = (got - want).norm();
                            assert!(
                                err <= 1e-13 * (1.0 + want.norm()),
                                "{g:?} {beta:?} element {i}: got {got:?}, expected {want:?}"
                            );
                        }
                    }
                }
            }
        }
    }
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
