//! The `StepBackend` route contract under a forced batch policy.
//!
//! A backend step that reports [`StepBackend::vendor_loop`] is a serial loop of
//! per-item calls, so the plan resolves the executor's batch policy with the
//! shared [`VendorLoopRoute`] rule before any write: a forced lane strategy runs
//! the tcontract alternative planned beside the step, and a forced
//! `WholeBatchVendor` is refused typed. These tests observe the route through a
//! marker backend that only counts its own executions, never through timing.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::api::Problem;
use cpueinsum::tcontract::{Error as ContractError, OutputContract};
use cpueinsum::{
    BackendError, BatchPolicy, BatchStrategy, EinsumPlan, EinsumSpec, Error, Exec, Layout, Scratch,
    StepBackend, VendorLoopRoute,
};
use tcontract_exec::{ExecError, WorkspaceReq};

/// The state a [`Marker`] step keeps.
#[derive(Debug)]
struct MarkerStep {
    /// What this step reports to [`StepBackend::vendor_loop`].
    vendor_loop: bool,
}

/// A backend that takes every step offered and records that `execute` ran.
///
/// It deliberately computes nothing: the tests assert the route, not the
/// arithmetic, and a zero-fill makes a wrongly taken step visible as a wrong
/// result as well.
///
/// `vendor_loops` gives the capability [`StepBackend::vendor_loop`] reports for
/// the step index the plan offers, the last entry repeating: `true` marks a
/// serial per-item loop, which the plan routes by the batch policy; `false`
/// marks a step that resolves the policy itself.
#[derive(Debug, Clone)]
struct Marker {
    runs: Arc<AtomicUsize>,
    planned: Arc<AtomicUsize>,
    vendor_loops: Vec<bool>,
}

impl Marker {
    fn new(runs: &Arc<AtomicUsize>, vendor_loops: &[bool]) -> Self {
        assert!(!vendor_loops.is_empty());
        Self {
            runs: Arc::clone(runs),
            planned: Arc::default(),
            vendor_loops: vendor_loops.to_vec(),
        }
    }

    /// The capability of the next step this instance is offered.
    fn next_vendor_loop(&self) -> bool {
        let k = self.planned.fetch_add(1, Ordering::Relaxed);
        let last = self.vendor_loops.len() - 1;
        self.vendor_loops[k.min(last)]
    }
}

impl StepBackend<f64> for Marker {
    type Step = MarkerStep;

    fn plan(&self, _: &Problem) -> Option<MarkerStep> {
        Some(MarkerStep {
            vendor_loop: self.next_vendor_loop(),
        })
    }

    fn vendor_loop(step: &MarkerStep) -> bool {
        step.vendor_loop
    }

    fn work_len(_: &MarkerStep) -> usize {
        0
    }

    fn execute(
        &self,
        _: &MarkerStep,
        _: &Exec<'_>,
        _: (&[f64], isize),
        _: (&[f64], isize),
        d: (&mut [f64], isize),
        _: &mut [f64],
    ) -> Result<(), BackendError> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        d.0.fill(0.0);
        Ok(())
    }
}

const EXT: usize = 4;

/// `d[i, l] = sum_{j,k} a[i, j] b[j, k] c[k, l]` on `EXT`-sized, column-major
/// operands, contracted as `(a b) c`: two binary steps.
fn chain() -> (
    EinsumSpec,
    [Layout<'static>; 3],
    Layout<'static>,
    Vec<Vec<f64>>,
) {
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [2, 3]]).unwrap();
    let cm = Layout::new(&[EXT, EXT], &[1, EXT as isize]).unwrap();
    // Integer-valued data: every route's arithmetic is exact, so results of
    // two routes are comparable bitwise.
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

fn policy(strategy: BatchStrategy) -> BatchPolicy {
    BatchPolicy::new(strategy)
}

#[test]
fn forced_lane_strategies_run_the_native_step_and_sequential_keeps_the_backend() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);

    // The reference is the all-tcontract plan under the same forced policy: the
    // alternative planned beside a vendor-loop step is that same plan.
    let mut expected = vec![0.0; EXT * EXT];
    let reference = EinsumPlan::<f64>::new(&spec, &layouts, out_layout).unwrap();
    assert_eq!(reference.backend_steps(), 0);
    let exec = Exec::rayon(&pool)
        .with_budget(4)
        .unwrap()
        .with_batch_policy(policy(BatchStrategy::Outer));
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
        let backend = Marker::new(&runs, &[true]);
        let plan =
            EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
        assert_eq!(plan.backend_steps(), 2);
        let exec = Exec::rayon(&pool)
            .with_budget(4)
            .unwrap()
            .with_batch_policy(policy(strategy));
        let mut got = vec![f64::NAN; EXT * EXT];
        let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
        plan.execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
            .unwrap();
        let loop_serves = matches!(strategy, BatchStrategy::Auto | BatchStrategy::Sequential);
        assert_eq!(
            runs.load(Ordering::Relaxed),
            if loop_serves { 2 } else { 0 },
            "{strategy:?} took the backend instead of the native alternative"
        );
        if !loop_serves {
            // The native alternative is the all-tcontract plan, so the forced
            // route's result is bitwise equal to it, and so is the native
            // workspace bound it contributes.
            assert_eq!(got, expected, "{strategy:?}");
            assert_eq!(
                plan.native_workspace_req(&exec, OutputContract::Initialized),
                reference.native_workspace_req(&exec, OutputContract::Initialized),
                "{strategy:?}"
            );
        } else {
            // The backend runs from the element arena, so no native workspace is
            // requested for it under `Auto`/`Sequential`.
            assert_eq!(
                plan.native_workspace_req(&exec, OutputContract::Initialized),
                WorkspaceReq::default(),
                "{strategy:?}"
            );
        }
    }

    // The shared rule the driver applies is the one the routes resolve.
    assert_eq!(
        VendorLoopRoute::resolve(BatchStrategy::Sequential).unwrap(),
        VendorLoopRoute::Loop
    );
    assert_eq!(
        VendorLoopRoute::resolve(BatchStrategy::Outer).unwrap(),
        VendorLoopRoute::Native
    );
}

#[test]
fn forced_whole_batch_vendor_is_refused_before_any_write() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    let exec = Exec::rayon(&pool).with_batch_policy(policy(BatchStrategy::WholeBatchVendor));
    let runs = Arc::new(AtomicUsize::new(0));
    let backend = Marker::new(&runs, &[true]);
    let plan = EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
    let mut got = vec![77.0; EXT * EXT];
    let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
    let error = plan
        .execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
        .unwrap_err();
    match error {
        Error::Contract {
            step,
            source: ContractError::Exec(ExecError::StrategyUnavailable { strategy }),
        } => {
            assert_eq!(step, 0);
            assert_eq!(strategy, BatchStrategy::WholeBatchVendor);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(runs.load(Ordering::Relaxed), 0);
    assert_eq!(got, [77.0; EXT * EXT]);
}

#[test]
fn the_refusal_precedes_a_step_that_would_run_on_the_backend() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    let exec = Exec::rayon(&pool).with_batch_policy(policy(BatchStrategy::WholeBatchVendor));
    // The first step resolves the batch policy itself, so it would run on the
    // backend; only the second is a serial loop. The plan resolves every backend
    // step's route before the first write, so the refusing later step stops the
    // whole plan: the earlier step neither ran nor wrote.
    let runs = Arc::new(AtomicUsize::new(0));
    let backend = Marker::new(&runs, &[false, true]);
    let plan = EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
    assert_eq!(plan.backend_steps(), 2);
    let mut got = vec![77.0; EXT * EXT];
    let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
    let error = plan
        .execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
        .unwrap_err();
    assert!(
        matches!(error, Error::Contract { step: 1, .. }),
        "{error:?}"
    );
    assert_eq!(runs.load(Ordering::Relaxed), 0);
    assert_eq!(got, [77.0; EXT * EXT]);
}

#[test]
fn a_backend_that_resolves_the_policy_itself_serves_every_strategy() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    // The default of `vendor_loop` is `false`: such a step keeps running on the
    // backend under every policy, including a forced whole-batch vendor route,
    // and no alternative is planned for it.
    for strategy in [
        BatchStrategy::WholeBatchVendor,
        BatchStrategy::Outer,
        BatchStrategy::Sequential,
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let backend = Marker::new(&runs, &[false]);
        let plan =
            EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
        let exec = Exec::rayon(&pool)
            .with_budget(4)
            .unwrap()
            .with_batch_policy(policy(strategy));
        let mut got = vec![f64::NAN; EXT * EXT];
        let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
        plan.execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
            .unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 2, "{strategy:?}");
    }
}

#[test]
fn an_empty_batch_resolves_the_route_the_same_way() {
    // `d[i, l, b] = sum_{j,k} a[i, j, b] b[j, k, b] c[k, l, b]` with the batch
    // label `b` at extent 0: an empty batch has no arithmetic to observe, so only
    // the route tells the strategies apart.
    let spec = EinsumSpec::new(
        &[&[0, 1, 4], &[1, 2, 4], &[2, 3, 4]],
        &[0, 3, 4],
        &[[0, 1], [2, 3]],
    )
    .unwrap();
    let cm = Layout::new(&[2, 2, 0], &[1, 2, 4]).unwrap();
    let data = [Vec::new(), Vec::new(), Vec::new()];
    let inputs: Vec<StridedView<'_, f64>> = data
        .iter()
        .map(|d| StridedView::new(d, &[2, 2, 0], &[1, 2, 4], 0).unwrap())
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    for (strategy, loop_serves) in [
        (BatchStrategy::Auto, true),
        (BatchStrategy::Sequential, true),
        (BatchStrategy::Outer, false),
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let backend = Marker::new(&runs, &[true]);
        let plan = EinsumPlan::<f64, _>::with_backend(backend, &spec, &[cm; 3], cm).unwrap();
        let exec = Exec::rayon(&pool)
            .with_budget(2)
            .unwrap()
            .with_batch_policy(policy(strategy));
        let mut got: Vec<f64> = Vec::new();
        let mut out = StridedViewMut::new(&mut got, &[2, 2, 0], &[1, 2, 4], 0).unwrap();
        plan.execute_into(&exec, &inputs, &mut out, &mut Scratch::new())
            .unwrap();
        assert_eq!(
            runs.load(Ordering::Relaxed),
            if loop_serves { 2 } else { 0 },
            "{strategy:?}"
        );
    }
}
