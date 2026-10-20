//! The N-ary route with the CBLAS backend under a forced batch policy.
//!
//! [`Blas`] reports every step as a serial loop of one GEMM per batch item
//! ([`StepBackend::vendor_loop`]), so the plan resolves the caller's batch
//! policy with the shared `VendorLoopRoute` rule before any write: `Auto` and
//! `Sequential` keep the loop and a forced lane strategy runs the tcontract
//! alternative planned beside the step. These tests observe the route through a
//! delegating backend that counts its own executions.

extern crate blas_src as _;

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::api::Problem;
use cpueinsum::{
    BackendError, BatchPolicy, BatchStrategy, EinsumPlan, EinsumSpec, Exec, Layout, Scratch,
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
