//! The N-ary route with the CBLAS backend under a forced batch policy.
//!
//! [`Blas`] reports every step as a serial loop of one GEMM per batch item
//! ([`StepBackend::vendor_loop`]), so the plan resolves the caller's batch
//! policy with the shared `VendorLoopRoute` rule before any write: `Auto` and
//! `Sequential` keep the loop, a forced lane strategy runs the tcontract
//! alternative planned beside the step, and a forced `WholeBatchVendor` is
//! refused typed. These tests observe the route through a delegating backend
//! that counts its own executions.

extern crate blas_src as _;

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::api::Problem;
use cpueinsum::{
    BackendError, BatchPolicy, BatchStrategy, EinsumPlan, EinsumSpec, Error, Exec, Layout, Scratch,
    StepBackend,
};
use cpueinsum_blas::{Blas, BlasStep};

/// [`Blas`] behind a counter, delegating every decision and call, so a test can
/// see whether a step ran on the backend.
#[derive(Debug)]
struct CountingBlas {
    inner: Blas,
    runs: Arc<AtomicUsize>,
}

impl StepBackend<f64> for CountingBlas {
    type Step = BlasStep;

    fn plan(&self, problem: &Problem) -> Option<BlasStep> {
        <Blas as StepBackend<f64>>::plan(&self.inner, problem)
    }

    /// Delegated, so the route these tests observe is the one the real backend
    /// reports.
    fn vendor_loop(step: &BlasStep) -> bool {
        <Blas as StepBackend<f64>>::vendor_loop(step)
    }

    fn work_len(step: &BlasStep) -> usize {
        <Blas as StepBackend<f64>>::work_len(step)
    }

    fn execute(
        &self,
        step: &BlasStep,
        exec: &Exec<'_>,
        a: (&[f64], isize),
        b: (&[f64], isize),
        d: (&mut [f64], isize),
        work: &mut [f64],
    ) -> Result<(), BackendError> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        <Blas as StepBackend<f64>>::execute(&self.inner, step, exec, a, b, d, work)
    }
}

const EXT: usize = 8;

/// `d[i, l] = sum_{j,k} a[i, j] b[j, k] c[k, l]` on `EXT`-sized column-major
/// operands, contracted as `(a b) c`: two steps, both matrix products.
fn chain() -> (
    EinsumSpec,
    [Layout<'static>; 3],
    Layout<'static>,
    Vec<Vec<f64>>,
) {
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [2, 3]]).unwrap();
    let cm = Layout::new(&[EXT, EXT], &[1, EXT as isize]).unwrap();
    // Integer-valued data: every route's arithmetic is exact, so results of two
    // routes are comparable bitwise.
    let values = |offset: usize| -> Vec<f64> {
        (0..EXT * EXT)
            .map(|i| ((i + offset) % 7) as f64 - 3.0)
            .collect()
    };
    (spec, [cm; 3], cm, vec![values(0), values(1), values(2)])
}

fn views(data: &[Vec<f64>]) -> Vec<StridedView<'_, f64>> {
    data.iter()
        .map(|d| StridedView::new(d, &[EXT, EXT], &[1, EXT as isize], 0).unwrap())
        .collect()
}

#[test]
fn forced_lane_strategies_run_the_native_step_and_sequential_keeps_the_blas_step() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);

    // The reference is the all-tcontract plan under the same forced policy: the
    // alternative planned beside a vendor-loop step is that same plan, so a
    // forced lane route must reproduce it bitwise.
    let reference = EinsumPlan::<f64>::new(&spec, &layouts, out_layout).unwrap();
    let exec = Exec::rayon(&pool)
        .with_budget(4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    let mut expected = vec![0.0; EXT * EXT];
    let mut out = StridedViewMut::new(&mut expected, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
    reference
        .execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
        .unwrap();

    for strategy in [
        BatchStrategy::Auto,
        BatchStrategy::Sequential,
        BatchStrategy::Outer,
        BatchStrategy::Inner,
        BatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        },
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let backend = CountingBlas {
            inner: Blas::with_min_macs(0),
            runs: Arc::clone(&runs),
        };
        let plan =
            EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
        assert_eq!(plan.backend_steps(), 2);
        let exec = Exec::rayon(&pool)
            .with_budget(4)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy));
        let mut got = vec![f64::NAN; EXT * EXT];
        let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
        plan.execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
            .unwrap();
        let loop_serves = matches!(strategy, BatchStrategy::Auto | BatchStrategy::Sequential);
        assert_eq!(
            runs.load(Ordering::Relaxed),
            if loop_serves { 2 } else { 0 },
            "{strategy:?} took the BLAS step instead of the native alternative"
        );
        if !loop_serves {
            assert_eq!(got, expected, "{strategy:?}");
        }
    }
}

#[test]
fn forced_whole_batch_vendor_is_refused_before_any_write() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&thread_pool);
    let exec =
        Exec::rayon(&pool).with_batch_policy(BatchPolicy::new(BatchStrategy::WholeBatchVendor));
    let runs = Arc::new(AtomicUsize::new(0));
    let backend = CountingBlas {
        inner: Blas::with_min_macs(0),
        runs: Arc::clone(&runs),
    };
    let plan = EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
    let mut got = vec![77.0; EXT * EXT];
    let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
    let error = plan
        .execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
        .unwrap_err();
    // The BLAS route is a per-item GEMM loop, not one vendor batch call: the
    // plan refuses it with tcontract's typed strategy error, before any step.
    assert!(
        matches!(error, Error::Contract { step: 0, .. }),
        "{error:?}"
    );
    assert_eq!(runs.load(Ordering::Relaxed), 0);
    assert_eq!(got, [77.0; EXT * EXT]);
}

/// `D[i, l, b] = sum_{j,k} A[i, j, b] B[j, k, b] C[k, l, b]` with the batch label
/// `b` at extent 0: an empty batch, so `Blas` declines every step (it has no
/// vendor geometry for an empty output) and the plan holds native steps only.
fn empty_batch_chain() -> (EinsumSpec, [Layout<'static>; 3], Layout<'static>) {
    let spec = EinsumSpec::new(
        &[&[0, 1, 4], &[1, 2, 4], &[2, 3, 4]],
        &[0, 3, 4],
        &[[0, 1], [3, 2]],
    )
    .unwrap();
    let cm = Layout::new(&[2, 2, 0], &[1, 2, 4]).unwrap();
    (spec, [cm; 3], cm)
}

#[test]
fn an_empty_batch_with_real_blas_still_refuses_a_forced_whole_batch_vendor() {
    // `Blas` declines an empty batch, so there is no backend step to refuse the
    // forced route: the plan-wide admission must refuse it for the *native* steps
    // too, before `Scratch::reserve` and before any step. A native plan rejects
    // the policy before its own empty-output shortcut.
    let (spec, layouts, out_layout) = empty_batch_chain();
    let runs = Arc::new(AtomicUsize::new(0));
    let backend = CountingBlas {
        inner: Blas::with_min_macs(0),
        runs: Arc::clone(&runs),
    };
    let plan = EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
    assert_eq!(plan.backend_steps(), 0, "an empty batch has no BLAS step");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    let data = [Vec::new(), Vec::new(), Vec::new()];
    let inputs: Vec<StridedView<'_, f64>> = data
        .iter()
        .map(|d| StridedView::new(d, &[2, 2, 0], &[1, 2, 4], 0).unwrap())
        .collect();

    let exec =
        Exec::rayon(&pool).with_batch_policy(BatchPolicy::new(BatchStrategy::WholeBatchVendor));
    let mut got: Vec<f64> = Vec::new();
    let mut out = StridedViewMut::new(&mut got, &[2, 2, 0], &[1, 2, 4], 0).unwrap();
    let mut scratch = Scratch::new();
    let error = plan
        .execute_into(&exec, &inputs, &mut out, &mut scratch)
        .unwrap_err();
    match error {
        Error::Contract { step, source } => {
            assert_eq!(step, 0);
            assert!(
                matches!(
                    source,
                    cpueinsum::tcontract::Error::Exec(
                        tcontract_exec::ExecError::StrategyUnavailable {
                            strategy: BatchStrategy::WholeBatchVendor
                        }
                    )
                ),
                "{source:?}"
            );
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(runs.load(Ordering::Relaxed), 0);
    assert_eq!(scratch.len(), 0);

    // Every other strategy runs the empty plan and completes.
    for strategy in [BatchStrategy::Auto, BatchStrategy::Sequential] {
        let exec = Exec::rayon(&pool).with_batch_policy(BatchPolicy::new(strategy));
        plan.execute_into(&exec, &inputs, &mut out, &mut scratch)
            .unwrap();
    }
}
