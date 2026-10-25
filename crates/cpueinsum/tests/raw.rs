//! The borrowed-description raw entry and the split workspace vocabulary (#23).

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tcontract::{OutputContract, PlanConfig};
use cpueinsum::{
    EinsumPlan, EinsumSpec, EinsumSpecRef, Error, Exec, Layout, Scratch, Tcontract, Workspace,
};

fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    (a, b)
}

#[test]
fn compile_matches_new_and_workspace_is_split() {
    let cm = Layout::new(&[8, 8], &[1, 8]).unwrap();
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    let plan = EinsumPlan::<f64>::new(&spec, &[cm, cm], cm).unwrap();
    let inputs: &[&[i64]] = &[&[0, 1], &[1, 2]];
    let borrowed = EinsumSpecRef {
        inputs,
        output: &[0, 2],
        path: &[[0, 1]],
    };
    let compiled = EinsumPlan::<f64, Tcontract>::compile(
        borrowed,
        &[cm, cm],
        cm,
        Tcontract,
        &PlanConfig::packed(),
    )
    .unwrap();

    // The intermediate arena is one element count; the native workspace is a
    // separate component-wise request.
    assert_eq!(plan.workspace_len(), plan.scratch_len());
    assert_eq!(compiled.workspace_len(), compiled.scratch_len());
    let req = compiled.native_workspace_req(&Exec::serial(), OutputContract::Initialized);
    assert!(
        req.bytes() > 0,
        "a packed step needs native workspace: {req:?}"
    );

    // Both plans agree on the data.
    let (a, b) = data(64);
    let mut out_new = vec![0.0; 64];
    let mut out_compiled = vec![0.0; 64];
    let mut scratch = Scratch::new();
    {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out_new, &[8, 8], &[1, 8], 0).unwrap();
        plan.execute_into(&mut Exec::serial(), &[av, bv], &mut dv, &mut scratch)
            .unwrap();
    }
    {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out_compiled, &[8, 8], &[1, 8], 0).unwrap();
        compiled
            .execute_into(&mut Exec::serial(), &[av, bv], &mut dv, &mut scratch)
            .unwrap();
    }
    assert_eq!(out_new, out_compiled);
}

#[test]
fn native_workspace_req_folds_component_wise() {
    // A two-step chain: (A B) then (AB) C. The folded request must dominate
    // each step's own request field by field.
    let cm = Layout::new(&[8, 8], &[1, 8]).unwrap();
    let inputs: &[&[i64]] = &[&[0, 1], &[1, 2], &[2, 3]];
    let borrowed = EinsumSpecRef {
        inputs,
        output: &[0, 3],
        path: &[[0, 1], [3, 2]],
    };
    let plan = EinsumPlan::<f64, Tcontract>::compile(
        borrowed,
        &[cm, cm, cm],
        cm,
        Tcontract,
        &PlanConfig::packed(),
    )
    .unwrap();
    let exec = Exec::serial();
    let folded = plan.native_workspace_req(&exec, OutputContract::Initialized);
    assert!(folded.bytes() > 0, "{folded:?}");

    // A one-step plan's request is one term of the fold, so the fold bounds it.
    let single_inputs: &[&[i64]] = &[&[0, 1], &[1, 2]];
    let single = EinsumPlan::<f64, Tcontract>::compile(
        EinsumSpecRef {
            inputs: single_inputs,
            output: &[0, 2],
            path: &[[0, 1]],
        },
        &[cm, cm],
        cm,
        Tcontract,
        &PlanConfig::packed(),
    )
    .unwrap()
    .native_workspace_req(&exec, OutputContract::Initialized);
    assert!(folded.a_bytes >= single.a_bytes);
    assert!(folded.b_bytes >= single.b_bytes);
    assert!(folded.tile_bytes >= single.tile_bytes);
}

#[test]
fn bounded_replay_refuses_a_short_workspace_before_writing() {
    // A packed three-operand chain: one [8, 8] intermediate and native
    // workspace, so both bounds are nonempty.
    let cm = Layout::new(&[8, 8], &[1, 8]).unwrap();
    let inputs: &[&[i64]] = &[&[0, 1], &[1, 2], &[2, 3]];
    let spec = EinsumSpecRef {
        inputs,
        output: &[0, 3],
        path: &[[0, 1], [3, 2]],
    };
    let plan = EinsumPlan::<f64, Tcontract>::compile(
        spec,
        &[cm, cm, cm],
        cm,
        Tcontract,
        &PlanConfig::packed(),
    )
    .unwrap();
    assert!(plan.workspace_len() > 0);
    let (a, b) = data(64);
    let c = vec![1.0; 64];
    let mut out = vec![7.0; 64];
    let mut work = vec![0.0; plan.workspace_len()];

    // A workspace that does not cover the plan's native requirement must be
    // refused, not silently grown, and nothing may be written.
    let req = plan.native_workspace_req(&Exec::serial(), OutputContract::Initialized);
    assert!(req.bytes() > 0, "{req:?}");
    let short = Workspace::default();
    let error = {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let cv = StridedView::new(&c, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &[8, 8], &[1, 8], 0).unwrap();
        plan.execute_bounded(&short, &[av, bv, cv], &mut dv, &mut work)
            .unwrap_err()
    };
    assert!(matches!(error, Error::Contract { .. }), "{error:?}");
    assert!(
        out.iter().all(|&x| x == 7.0),
        "short workspace wrote output"
    );

    // A short arena is refused just as firmly.
    let ok_workspace = Workspace::with_req(&req);
    let mut tiny_work: Vec<f64> = Vec::new();
    let error = {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let cv = StridedView::new(&c, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &[8, 8], &[1, 8], 0).unwrap();
        plan.execute_bounded(&ok_workspace, &[av, bv, cv], &mut dv, &mut tiny_work)
            .unwrap_err()
    };
    assert!(matches!(error, Error::Workspace { .. }), "{error:?}");
    assert!(out.iter().all(|&x| x == 7.0), "short arena wrote output");

    // A covering workspace and arena run, and the result is the unbounded route's.
    {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let cv = StridedView::new(&c, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &[8, 8], &[1, 8], 0).unwrap();
        plan.execute_bounded(&ok_workspace, &[av, bv, cv], &mut dv, &mut work)
            .unwrap();
    }
    let mut reference = vec![0.0; 64];
    let mut scratch = Scratch::new();
    {
        let av = StridedView::new(&a, &[8, 8], &[1, 8], 0).unwrap();
        let bv = StridedView::new(&b, &[8, 8], &[1, 8], 0).unwrap();
        let cv = StridedView::new(&c, &[8, 8], &[1, 8], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut reference, &[8, 8], &[1, 8], 0).unwrap();
        plan.execute_into(&mut Exec::serial(), &[av, bv, cv], &mut dv, &mut scratch)
            .unwrap();
    }
    assert_eq!(out, reference);
}
