//! cpueinsum against a naive einsum over every label assignment.

mod common;

use common::{close, naive, Operand};
use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{
    contract_into, einsum_into, EinsumPlan, EinsumSpec, Error, Exec, Layout, Pool, Scratch,
    ShapeError, SpecError,
};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Run an einsum on column-major operands and compare with the naive sum.
fn check(inputs: &[&[i64]], output: &[i64], path: &[[usize; 2]], ext: &dyn Fn(i64) -> usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let ops: Vec<Operand<f64>> = inputs
        .iter()
        .map(|ls| Operand::random_col_major(&mut rng, ls, ext))
        .collect();
    let out_dims: Vec<usize> = output.iter().map(|&l| ext(l)).collect();
    let expected = naive(&ops, output, &out_dims);

    let spec = EinsumSpec::new(inputs, output, path).unwrap();
    let views: Vec<StridedView<'_, f64>> = ops.iter().map(Operand::view).collect();
    let (strides, len) = common::col_major(&out_dims);
    let mut d = vec![f64::NAN; len];
    let mut dv = StridedViewMut::new(&mut d, &out_dims, &strides, 0).unwrap();
    einsum_into(&Exec::serial(), &spec, &views, &mut dv).unwrap();
    close(&d, &expected, 1e-12);
}

fn three(_: i64) -> usize {
    3
}

#[test]
fn binary_matmul_and_permuted_output() {
    check(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]], &three);
    check(&[&[0, 1], &[1, 2]], &[2, 0], &[[1, 0]], &three);
}

#[test]
fn chains_with_every_order() {
    let inputs: &[&[i64]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 4]];
    let ext = |l: i64| [2, 3, 4, 3, 2][l as usize];
    check(inputs, &[0, 4], &[[0, 1], [4, 2], [5, 3]], &ext);
    check(inputs, &[4, 0], &[[2, 3], [1, 4], [0, 5]], &ext);
    check(inputs, &[0, 4], &[[0, 1], [2, 3], [4, 5]], &ext);
    check(inputs, &[0, 4], &[[1, 2], [0, 4], [5, 3]], &ext);
}

#[test]
fn diagonals_reductions_and_outer_products() {
    // Diagonal of the first input.
    check(&[&[0, 0, 1], &[1, 2]], &[0, 2], &[[0, 1]], &three);
    // Labels summed inside one input.
    check(&[&[0, 1, 3], &[1, 2, 4]], &[0, 2], &[[0, 1]], &three);
    // Outer product and batch (Hadamard).
    check(&[&[0, 1], &[2]], &[2, 0, 1], &[[0, 1]], &three);
    check(&[&[0, 1], &[0, 1]], &[1, 0], &[[0, 1]], &three);
    // A label kept through an intermediate because a later input needs it.
    check(&[&[0, 1], &[1, 2], &[0, 2]], &[], &[[0, 1], [3, 2]], &three);
    // Full contraction to a scalar intermediate, then an outer product.
    check(&[&[0], &[0], &[1]], &[1], &[[0, 1], [3, 2]], &three);
}

#[test]
fn single_input() {
    check(&[&[0, 1, 2]], &[2, 0, 1], &[], &|l| (l + 2) as usize);
    check(&[&[0, 0]], &[0], &[], &three);
    check(&[&[0, 0]], &[], &[], &three);
    check(&[&[0, 1]], &[], &[], &three);
    check(&[&[0, 1, 0]], &[1], &[], &three);
}

#[test]
fn mps_chain() {
    // Four MPS tensors A_k[b_k, s_k, b_{k+1}] contracted left to right.
    // Bond labels 0, 2, 4, 6, 8; physical labels 1, 3, 5, 7.
    let inputs: &[&[i64]] = &[&[0, 1, 2], &[2, 3, 4], &[4, 5, 6], &[6, 7, 8]];
    let ext = |l: i64| if l % 2 == 0 { 4 } else { 2 };
    check(inputs, &[0, 1, 3, 5, 7, 8], &[[0, 1], [4, 2], [5, 3]], &ext);
    // And the norm-like closure of two copies is covered by the random test.
}

#[test]
fn arena_reuses_dead_intermediates() {
    let inputs: Vec<Vec<i64>> = (0..8).map(|k| vec![k, k + 1]).collect();
    let refs: Vec<&[i64]> = inputs.iter().map(Vec::as_slice).collect();
    let path: Vec<[usize; 2]> = (0..7)
        .map(|k| [if k == 0 { 0 } else { 7 + k }, k + 1])
        .collect();
    let spec = EinsumSpec::new(&refs, &[0, 8], &path).unwrap();
    let dims = [5usize, 5];
    let strides = [1isize, 5];
    let cm = Layout::new(&dims, &strides).unwrap();
    let plan = EinsumPlan::<f64>::new(&spec, &[cm; 8], cm).unwrap();
    // Six 5x5 intermediates, at most two alive at once.
    assert_eq!(plan.steps(), 7);
    assert_eq!(plan.scratch_len(), 50);
    check(&refs, &[0, 8], &path, &|_| 5);
}

#[test]
fn strided_and_negative_stride_inputs() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    // A: 3x4 view into a 7x9 buffer, every other row, columns reversed.
    let abuf: Vec<f64> = (0..63).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let a = Operand::new(abuf, &[3, 4], &[2, -7], 7 * 8);
    // B: 4x2 row-major.
    let bbuf: Vec<f64> = (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b = Operand::new(bbuf, &[4, 2], &[2, 1], 0);
    let ops = [a.labelled(&[0, 1]), b.labelled(&[1, 2])];
    let expected = naive(&ops, &[2, 0], &[2, 3]);

    // Output transposed in memory: d[k, i] at k * 3 + i... stored row-major.
    let mut d = vec![f64::NAN; 6];
    let mut dv = StridedViewMut::new(&mut d, &[2, 3], &[3, 1], 0).unwrap();
    let views = [ops[0].view(), ops[1].view()];
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[2, 0], &[[0, 1]]).unwrap();
    einsum_into(&Exec::serial(), &spec, &views, &mut dv).unwrap();
    let got_col_major: Vec<f64> = (0..3)
        .flat_map(|i| (0..2).map(move |k| (k, i)))
        .map(|(k, i)| d[k * 3 + i])
        .collect();
    close(&got_col_major, &expected, 1e-12);
}

#[test]
fn complex_chain() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let inputs: &[&[i64]] = &[&[0, 1], &[1, 2], &[2, 0, 3]];
    let ops: Vec<Operand<Complex64>> = inputs
        .iter()
        .map(|ls| Operand::random_col_major(&mut rng, ls, &three))
        .collect();
    let expected = naive(&ops, &[3], &[3]);
    let spec = EinsumSpec::new(inputs, &[3], &[[0, 2], [3, 1]]).unwrap();
    let views: Vec<_> = ops.iter().map(Operand::view).collect();
    let mut d = vec![Complex64::new(f64::NAN, 0.0); 3];
    let mut dv = StridedViewMut::new(&mut d, &[3], &[1], 0).unwrap();
    einsum_into(&Exec::serial(), &spec, &views, &mut dv).unwrap();
    close(&d, &expected, 1e-12);
}

#[test]
fn plan_reuse_with_dirty_scratch_and_rayon() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let inputs: &[&[i64]] = &[&[0, 1, 2], &[2, 3, 4], &[4, 5, 6]];
    let ext = |l: i64| if l % 2 == 0 { 16 } else { 2 };
    let spec = EinsumSpec::new(inputs, &[0, 1, 3, 5, 6], &[[0, 1], [3, 2]]).unwrap();
    let ops: Vec<Operand<f64>> = inputs
        .iter()
        .map(|ls| Operand::random_col_major(&mut rng, ls, &ext))
        .collect();
    let out_dims = [16usize, 2, 2, 2, 16];
    let expected = naive(&ops, &[0, 1, 3, 5, 6], &out_dims);
    let layouts: Vec<Layout<'_>> = ops.iter().map(Operand::layout).collect();
    let (strides, len) = common::col_major(&out_dims);
    let plan =
        EinsumPlan::<f64>::new(&spec, &layouts, Layout::new(&out_dims, &strides).unwrap()).unwrap();
    let views: Vec<_> = ops.iter().map(Operand::view).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&pool);
    let execs = [Exec::serial(), Exec::rayon(&pool)];
    let mut scratch = Scratch::with_len(plan.scratch_len() + 3);
    for exec in &execs {
        for _ in 0..2 {
            let mut d = vec![f64::NAN; len];
            let mut dv = StridedViewMut::new(&mut d, &out_dims, &strides, 0).unwrap();
            plan.execute_into(exec, &views, &mut dv, &mut scratch)
                .unwrap();
            close(&d, &expected, 1e-12);
        }
    }
}

#[test]
fn binary_contract_into() {
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let a = Operand::<f64>::random_col_major(&mut rng, &[0, 1, 1], &three);
    let b = Operand::<f64>::random_col_major(&mut rng, &[1, 2], &three);
    let expected = naive(&[a.clone(), b.clone()], &[2, 0], &[3, 3]);
    let mut d = vec![f64::NAN; 9];
    let mut dv = StridedViewMut::new(&mut d, &[3, 3], &[1, 3], 0).unwrap();
    contract_into(
        &Exec::serial(),
        &a.view(),
        &[0, 1, 1],
        &b.view(),
        &[1, 2],
        &mut dv,
        &[2, 0],
    )
    .unwrap();
    close(&d, &expected, 1e-12);
}

#[test]
fn random_einsums() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for case in 0..300 {
        let n = rng.gen_range(1..=5);
        let nlabels = rng.gen_range(1..=6);
        let extents: Vec<usize> = (0..nlabels).map(|_| rng.gen_range(1..=3)).collect();
        let inputs: Vec<Vec<i64>> = (0..n)
            .map(|_| {
                let rank = rng.gen_range(0..=3);
                (0..rank)
                    .map(|_| rng.gen_range(0..nlabels as i64))
                    .collect()
            })
            .collect();
        let mut used: Vec<i64> = inputs.iter().flatten().copied().collect();
        used.sort_unstable();
        used.dedup();
        let mut output: Vec<i64> = used.iter().copied().filter(|_| rng.gen_bool(0.5)).collect();
        for i in (1..output.len()).rev() {
            output.swap(i, rng.gen_range(0..=i));
        }
        // A random order in SSA form.
        let mut alive: Vec<usize> = (0..n).collect();
        let mut path = Vec::new();
        for step in 0..n.saturating_sub(1) {
            let x = alive.swap_remove(rng.gen_range(0..alive.len()));
            let y = alive.swap_remove(rng.gen_range(0..alive.len()));
            path.push([x, y]);
            alive.push(n + step);
        }
        let refs: Vec<&[i64]> = inputs.iter().map(Vec::as_slice).collect();
        let ext = |l: i64| extents[l as usize];
        let result = std::panic::catch_unwind(|| check(&refs, &output, &path, &ext));
        assert!(
            result.is_ok(),
            "case {case}: {refs:?} -> {output:?} via {path:?}"
        );
    }
}

#[test]
fn zero_extent() {
    check(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]], &|l| {
        if l == 1 {
            0
        } else {
            2
        }
    });
    check(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]], &|l| {
        if l == 0 {
            0
        } else {
            2
        }
    });
}

#[test]
fn spec_errors() {
    let e = |r: cpueinsum::Result<EinsumSpec>| match r {
        Err(Error::Spec(s)) => s,
        other => panic!("expected a spec error, got {other:?}"),
    };
    assert_eq!(e(EinsumSpec::new(&[], &[], &[])), SpecError::NoInputs);
    assert!(matches!(
        e(EinsumSpec::new(&[&[0], &[0]], &[], &[])),
        SpecError::PathLength {
            expected: 1,
            got: 0
        }
    ));
    assert!(matches!(
        e(EinsumSpec::new(&[&[0], &[0]], &[], &[[0, 0]])),
        SpecError::PathSelf {
            step: 0,
            operand: 0
        }
    ));
    assert!(matches!(
        e(EinsumSpec::new(&[&[0], &[0], &[0]], &[], &[[0, 1], [0, 2]])),
        SpecError::PathReuse {
            operand: 0,
            first: 0,
            second: 1
        }
    ));
    assert!(matches!(
        e(EinsumSpec::new(&[&[0, 1]], &[0, 0], &[])),
        SpecError::OutputRepeated { label: 0 }
    ));
    assert!(matches!(
        e(EinsumSpec::new(&[&[0, 1]], &[2], &[])),
        SpecError::OutputOnly { label: 2 }
    ));
}

#[test]
fn shape_errors() {
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    let l23 = Layout::new(&[2, 3], &[1, 2]).unwrap();
    let l22 = Layout::new(&[2, 2], &[1, 2]).unwrap();
    let shape = |r: cpueinsum::Result<EinsumPlan<f64>>| match r {
        Err(Error::Shape(s)) => s,
        other => panic!("expected a shape error, got {other:?}"),
    };
    assert!(matches!(
        shape(EinsumPlan::new(&spec, &[l23], l22)),
        ShapeError::InputCount {
            expected: 2,
            got: 1
        }
    ));
    assert!(matches!(
        shape(EinsumPlan::new(&spec, &[l23, l22], l22)),
        ShapeError::Extent {
            label: 1,
            first: 3,
            second: 2
        }
    ));
    let l3 = Layout::new(&[3], &[1]).unwrap();
    assert!(matches!(
        shape(EinsumPlan::new(&spec, &[l23, l3], l22)),
        ShapeError::Rank {
            operand: 1,
            rank: 1,
            labels: 2
        }
    ));

    // Views that differ from the planned layouts are rejected before writing.
    let l32 = Layout::new(&[3, 2], &[1, 3]).unwrap();
    let plan = EinsumPlan::<f64>::new(&spec, &[l23, l32], l22).unwrap();
    let a = [0.0; 6];
    let b = [0.0; 9];
    let mut d = [7.0; 4];
    let av = StridedView::new(&a, &[2, 3], &[3, 1], 0).unwrap();
    let bv = StridedView::new(&b, &[3, 2], &[1, 3], 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
    let r = plan.execute_into(&Exec::serial(), &[av, bv], &mut dv, &mut Scratch::new());
    assert!(matches!(
        r,
        Err(Error::Shape(ShapeError::Mismatch { operand: 0 }))
    ));
    assert_eq!(d, [7.0; 4]);
}

#[test]
fn aliasing_output_is_rejected_by_tprims() {
    let spec = EinsumSpec::new(&[&[0, 1]], &[0, 1], &[]).unwrap();
    let l = Layout::new(&[2, 2], &[1, 2]).unwrap();
    let alias = Layout::new(&[2, 2], &[1, 0]).unwrap();
    let r = EinsumPlan::<f64>::new(&spec, &[l], alias);
    assert!(matches!(r, Err(Error::Contract { step: 0, .. })), "{r:?}");
}
