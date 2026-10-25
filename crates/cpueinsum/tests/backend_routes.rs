//! The `StepBackend` route contract under a forced batch policy.
//!
//! A backend step that reports [`StepBackend::vendor_loop`] is a serial loop of
//! per-item calls, so the plan resolves the executor's batch policy with the
//! shared [`VendorLoopRoute`] rule before any write: a forced lane strategy runs
//! the tcontract alternative planned beside the step. These tests observe the
//! route through a marker backend that only counts its own executions, never
//! through timing.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::api::Problem;
use cpueinsum::tcontract::{Error as ContractError, OutputContract, PlanConfig};
use cpueinsum::{
    BackendError, BatchPolicy, BatchStrategy, EinsumPlan, EinsumSpec, EinsumSpecRef, Error, Exec,
    Layout, Scratch, StepBackend, Tcontract, VendorLoopRoute,
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
        _: &mut Exec<'_>,
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
    let pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);

    // The reference is the all-tcontract plan under the same forced policy: the
    // alternative planned beside a vendor-loop step is that same plan.
    let mut expected = vec![0.0; EXT * EXT];
    let reference = EinsumPlan::<f64>::new(&spec, &layouts, out_layout).unwrap();
    assert_eq!(reference.backend_steps(), 0);
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(policy(BatchStrategy::Outer));
    let mut out = StridedViewMut::new(&mut expected, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
    reference
        .execute_into(&mut exec, &inputs, &mut out, &mut Scratch::new())
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
        let mut exec = Exec::new(&pool, 4)
            .unwrap()
            .with_batch_policy(policy(strategy));
        let mut got = vec![f64::NAN; EXT * EXT];
        let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
        plan.execute_into(&mut exec, &inputs, &mut out, &mut Scratch::new())
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
        VendorLoopRoute::resolve(BatchStrategy::Sequential),
        VendorLoopRoute::Loop
    );
    assert_eq!(
        VendorLoopRoute::resolve(BatchStrategy::Outer),
        VendorLoopRoute::Native
    );
}

#[test]
fn a_backend_that_resolves_the_policy_itself_serves_every_strategy() {
    let (spec, layouts, out_layout, data) = chain();
    let inputs = views(&data);
    let pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    // The default of `vendor_loop` is `false`: such a step keeps running on the
    // backend under every policy, and no alternative is planned for it.
    for strategy in [BatchStrategy::Outer, BatchStrategy::Sequential] {
        let runs = Arc::new(AtomicUsize::new(0));
        let backend = Marker::new(&runs, &[false]);
        let plan =
            EinsumPlan::<f64, _>::with_backend(backend, &spec, &layouts, out_layout).unwrap();
        let mut exec = Exec::new(&pool, 4)
            .unwrap()
            .with_batch_policy(policy(strategy));
        let mut got = vec![f64::NAN; EXT * EXT];
        let mut out = StridedViewMut::new(&mut got, &[EXT, EXT], &[1, EXT as isize], 0).unwrap();
        plan.execute_into(&mut exec, &inputs, &mut out, &mut Scratch::new())
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
    let pool = strided_exec::Pool::builder().threads(2).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    for (strategy, loop_serves) in [
        (BatchStrategy::Auto, true),
        (BatchStrategy::Sequential, true),
        (BatchStrategy::Outer, false),
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let backend = Marker::new(&runs, &[true]);
        let plan = EinsumPlan::<f64, _>::with_backend(backend, &spec, &[cm; 3], cm).unwrap();
        let mut exec = Exec::new(&pool, 2)
            .unwrap()
            .with_batch_policy(policy(strategy));
        let mut got: Vec<f64> = Vec::new();
        let mut out = StridedViewMut::new(&mut got, &[2, 2, 0], &[1, 2, 4], 0).unwrap();
        plan.execute_into(&mut exec, &inputs, &mut out, &mut Scratch::new())
            .unwrap();
        assert_eq!(
            runs.load(Ordering::Relaxed),
            if loop_serves { 2 } else { 0 },
            "{strategy:?}"
        );
    }
}

/// `D[i, l] = sum_{j,k} A[i, k] B[k, j] C[i, j, l]`, contracted as `(A B) C` on
/// packed `2`-sized operands: the first step has no batch axis, the second has
/// the batch label `i`.
fn packed_chain() -> (
    EinsumPlan<f64>,
    [Layout<'static>; 3],
    Layout<'static>,
    [Vec<f64>; 3],
) {
    let inputs: &[&[i64]] = &[&[0, 1], &[1, 2], &[0, 2, 3]];
    let spec = EinsumSpecRef {
        inputs,
        output: &[0, 3],
        path: &[[0, 1], [3, 2]],
    };
    let cm2 = Layout::new(&[2, 2], &[1, 2]).unwrap();
    let cm3 = Layout::new(&[2, 2, 2], &[1, 2, 4]).unwrap();
    let plan = EinsumPlan::<f64, Tcontract>::compile(
        spec,
        &[cm2, cm2, cm3],
        cm2,
        Tcontract,
        &PlanConfig::packed(),
    )
    .unwrap();
    let data = [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0; 8],
    ];
    (plan, [cm2, cm2, cm3], cm2, data)
}

#[test]
fn a_native_alternative_that_refuses_the_split_is_admitted_before_the_first_step() {
    // tcontract's packed driver refuses a decision with more than one lane *and*
    // more than one item thread, and it refuses it when the second step's
    // native plan is asked for its route. The plan resolves every step's
    // admission before the first write, so the refusal is reported for step 1
    // with the first step's intermediate never written and the arena never
    // reserved.
    let (plan, layouts, out_layout, data) = packed_chain();
    assert_eq!(plan.steps(), 2);
    let pool = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&pool);
    let inputs: Vec<StridedView<'_, f64>> = data
        .iter()
        .zip([&layouts[0], &layouts[1], &layouts[2]])
        .map(|(d, l)| StridedView::new(d, l.dims(), l.strides(), 0).unwrap())
        .collect();
    let split = BatchStrategy::Split {
        lanes: NonZeroUsize::new(2).unwrap(),
    };
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(policy(split));
    let mut got = vec![77.0; 4];
    let mut out =
        StridedViewMut::new(&mut got, out_layout.dims(), out_layout.strides(), 0).unwrap();
    let mut scratch = Scratch::new();
    let error = plan
        .execute_into(&mut exec, &inputs, &mut out, &mut scratch)
        .unwrap_err();
    match error {
        Error::Contract {
            step,
            source: ContractError::Exec(ExecError::StrategyUnavailable { strategy }),
        } => {
            assert_eq!(step, 1, "the refusing step is the batched one");
            assert_eq!(strategy, split);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(got, [77.0; 4], "the output was written before the refusal");
    // Nothing can have run: the arena is carved out of `scratch`, which a
    // refusal before `Scratch::reserve` never grows.
    assert!(plan.scratch_len() > 0);
    assert_eq!(scratch.len(), 0, "a step ran before the refusal");

    // The same plan under a split the native route can honour (`lanes = 1`, item
    // threads on the whole budget) does run, so the refusal above is the
    // strategy's and not the plan's.
    let mut exec = Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(policy(BatchStrategy::Split {
            lanes: NonZeroUsize::new(1).unwrap(),
        }));
    let mut got = vec![f64::NAN; 4];
    let mut out =
        StridedViewMut::new(&mut got, out_layout.dims(), out_layout.strides(), 0).unwrap();
    plan.execute_into(&mut exec, &inputs, &mut out, &mut scratch)
        .unwrap();
    assert_eq!(got, [4.0, 6.0, 4.0, 6.0]);
    assert_eq!(scratch.len(), plan.scratch_len());
}
