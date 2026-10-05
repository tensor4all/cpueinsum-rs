//! cpueinsum with the BLAS backend against a naive einsum.

extern crate blas_src as _;

#[path = "../../cpueinsum/tests/common/mod.rs"]
mod common;

use common::{close, naive, Elem, Operand};
use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::tprims_contract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Scratch, StepBackend};
use cpueinsum_blas::{Blas, BlasScalar};
use num_complex::{Complex32, Complex64};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A random layout of `dims`: a random axis order, gaps between axes, a
/// reversed axis and a leading offset, each sometimes.
fn random_layout<T: Elem>(rng: &mut impl Rng, labels: &[i64], dims: &[usize]) -> Operand<T> {
    let mut order: Vec<usize> = (0..dims.len()).collect();
    if rng.gen_bool(0.6) {
        for i in (1..order.len()).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }
    }
    let mut strides = vec![0isize; dims.len()];
    let mut s = 1isize;
    for &e in &order {
        strides[e] = s;
        let gap = if rng.gen_bool(0.2) { 2 } else { 1 };
        s *= (dims[e] * gap).max(1) as isize;
    }
    let lead = if rng.gen_bool(0.3) { 3 } else { 0 };
    let mut offset = lead as isize;
    if !dims.is_empty() && rng.gen_bool(0.2) {
        let e = rng.gen_range(0..dims.len());
        offset += strides[e] * (dims[e].max(1) as isize - 1);
        strides[e] = -strides[e];
    }
    let len = lead + s as usize + 2;
    let data = (0..len).map(|_| T::random(rng)).collect();
    let mut o = Operand::new(data, dims, &strides, offset as usize);
    o.labels = labels.to_vec();
    o
}

/// Every element of `o` in column-major order.
fn col_major_values<T: Elem>(o: &Operand<T>) -> Vec<T> {
    let (_, len) = common::col_major(&o.dims);
    let mut idx = vec![0usize; o.dims.len()];
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(o.at(&idx));
        for k in 0..idx.len() {
            idx[k] += 1;
            if idx[k] < o.dims[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

/// Plan with `blas`, run on random layouts and compare with the naive sum;
/// returns the number of steps BLAS took.
fn check<T: Elem + BlasScalar>(
    rng: &mut impl Rng,
    blas: Blas,
    inputs: &[&[i64]],
    output: &[i64],
    path: &[[usize; 2]],
    ext: &dyn Fn(i64) -> usize,
    tol: f64,
) -> usize {
    let ops: Vec<Operand<T>> = inputs
        .iter()
        .map(|ls| {
            let dims: Vec<usize> = ls.iter().map(|&l| ext(l)).collect();
            random_layout(rng, ls, &dims)
        })
        .collect();
    let out_dims: Vec<usize> = output.iter().map(|&l| ext(l)).collect();
    let expected = naive(&ops, output, &out_dims);

    let spec = EinsumSpec::new(inputs, output, path).unwrap();
    let layouts: Vec<_> = ops.iter().map(Operand::layout).collect();
    let mut d: Operand<T> = random_layout(rng, output, &out_dims);
    let nan = T::random(rng) * T::random(rng);
    // A marker in every element the output must not touch.
    let marker = d.data.iter().map(|_| nan).collect::<Vec<_>>();
    d.data = marker.clone();
    let plan = EinsumPlan::<T, Blas>::with_backend(blas, &spec, &layouts, d.layout()).unwrap();
    let views: Vec<StridedView<'_, T>> = ops.iter().map(Operand::view).collect();
    let mut scratch = Scratch::new();
    // Twice, the second time over a dirty scratch.
    for _ in 0..2 {
        let mut dv =
            StridedViewMut::new(&mut d.data, &d.dims, &d.strides, d.offset as isize).unwrap();
        plan.execute_into(&Exec::serial(), &views, &mut dv, &mut scratch)
            .unwrap();
        close(&col_major_values(&d), &expected, tol);
    }
    // Elements outside the output keep the marker.
    let mut touched = vec![false; d.data.len()];
    let mut idx = vec![0usize; d.dims.len()];
    let count: usize = d.dims.iter().product();
    for _ in 0..count {
        let p = d.offset as isize
            + idx
                .iter()
                .zip(&d.strides)
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        touched[p as usize] = true;
        for k in 0..idx.len() {
            idx[k] += 1;
            if idx[k] < d.dims[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    for (k, t) in touched.iter().enumerate() {
        if !t {
            assert!(d.data[k].dist(marker[k]) == 0.0, "element {k} written");
        }
    }
    plan.backend_steps()
}

fn all() -> Blas {
    Blas::with_min_macs(0)
}

/// Matrix products, batched and multi-index, over many layouts per type.
fn products<T: Elem + BlasScalar>(seed: u64, tol: f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let ext = |l: i64| [5, 6, 7, 3, 2, 4][l as usize];
    let cases: &[(&[&[i64]], &[i64])] = &[
        (&[&[0, 1], &[1, 2]], &[0, 2]),
        (&[&[0, 1], &[1, 2]], &[2, 0]),
        (&[&[1, 0], &[2, 1]], &[0, 2]),
        // Matrix-vector and vector-matrix.
        (&[&[0, 1], &[1]], &[0]),
        (&[&[1], &[1, 2]], &[2]),
        // Outer product.
        (&[&[0], &[2]], &[0, 2]),
        // Batched, with batch axes in different places.
        (&[&[3, 0, 1], &[1, 2, 3]], &[0, 2, 3]),
        (&[&[0, 3, 1, 4], &[4, 1, 2, 3]], &[3, 2, 0, 4]),
        // Multi-index groups.
        (&[&[0, 3, 1, 5], &[1, 5, 2, 4]], &[0, 3, 2, 4]),
        (&[&[5, 1, 3, 0], &[2, 1, 4, 5]], &[4, 3, 0, 2]),
        // Dot product.
        (&[&[0, 1], &[0, 1]], &[]),
    ];
    for &(inputs, output) in cases {
        for _ in 0..12 {
            let taken = check::<T>(&mut rng, all(), inputs, output, &[[0, 1]], &ext, tol);
            assert_eq!(taken, 1, "{inputs:?} -> {output:?}");
        }
    }
}

#[test]
fn products_f64() {
    products::<f64>(1, 1e-12);
}

#[test]
fn products_f32() {
    products::<f32>(2, 1e-4);
}

#[test]
fn products_complex64() {
    products::<Complex64>(3, 1e-12);
}

#[test]
fn products_complex32() {
    products::<Complex32>(4, 1e-4);
}

#[test]
fn chains_mix_backends() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    // MPS-like chain; a Hadamard step and an isolated reduction stay on
    // tprims-contract.
    let inputs: &[&[i64]] = &[&[0, 1, 2], &[2, 3, 4], &[4, 5, 6], &[6, 7, 8]];
    let ext = |l: i64| if l % 2 == 0 { 6 } else { 2 };
    let path = [[0, 1], [4, 2], [5, 3]];
    let taken = check::<f64>(
        &mut rng,
        all(),
        inputs,
        &[0, 1, 3, 5, 7, 8],
        &path,
        &ext,
        1e-12,
    );
    assert_eq!(taken, 3);
    let taken = check::<f64>(
        &mut rng,
        all(),
        &[&[0, 1], &[0, 1], &[1, 2, 3]],
        &[2],
        &[[0, 1], [3, 2]],
        &|_| 4,
        1e-12,
    );
    // [0, 1] x [0, 1] -> [1] is a batch of dot products and taken; the
    // reduction over label 3, which only the last input carries, is declined.
    assert_eq!(taken, 1);
    let taken = check::<f64>(
        &mut rng,
        all(),
        &[&[0, 1], &[0, 1], &[0, 1]],
        &[0, 1],
        &[[0, 1], [3, 2]],
        &|_| 4,
        1e-12,
    );
    // Hadamard products have no matrix.
    assert_eq!(taken, 0);
}

#[test]
fn default_threshold_declines_small_steps() {
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    let mm: &[&[i64]] = &[&[0, 1], &[1, 2]];
    let small = check::<f64>(
        &mut rng,
        Blas::default(),
        mm,
        &[0, 2],
        &[[0, 1]],
        &|_| 4,
        1e-12,
    );
    assert_eq!(small, 0);
    let large = check::<f64>(
        &mut rng,
        Blas::default(),
        mm,
        &[0, 2],
        &[[0, 1]],
        &|_| 40,
        1e-12,
    );
    assert_eq!(large, 1);
    // 64 items of 8^3: large in total, each GEMM small.
    let bmm: &[&[i64]] = &[&[0, 1, 3], &[1, 2, 3]];
    let dims = |l: i64| if l == 3 { 64 } else { 8 };
    let batched = check::<f64>(
        &mut rng,
        Blas::default(),
        bmm,
        &[0, 2, 3],
        &[[0, 1]],
        &dims,
        1e-12,
    );
    assert_eq!(batched, 0);
    let forced = Blas::with_min_macs(0);
    let batched = check::<f64>(&mut rng, forced, bmm, &[0, 2, 3], &[[0, 1]], &dims, 1e-12);
    assert_eq!(batched, 1);
}

#[test]
fn random_einsums() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut taken = 0;
    for case in 0..300 {
        let n = rng.gen_range(1..=5);
        let nlabels = rng.gen_range(1..=6);
        let extents: Vec<usize> = (0..nlabels).map(|_| rng.gen_range(1..=4)).collect();
        let inputs: Vec<Vec<i64>> = (0..n)
            .map(|_| {
                let rank = rng.gen_range(0..=4);
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
        let seed = rng.gen::<u64>();
        let complex = case % 3 == 0;
        let result = std::panic::catch_unwind(|| {
            let mut r = ChaCha8Rng::seed_from_u64(seed);
            if complex {
                check::<Complex64>(&mut r, all(), &refs, &output, &path, &ext, 1e-12)
            } else {
                check::<f64>(&mut r, all(), &refs, &output, &path, &ext, 1e-12)
            }
        });
        match result {
            Ok(t) => taken += t,
            Err(_) => panic!("case {case}: {refs:?} -> {output:?} via {path:?}"),
        }
    }
    // The backend took a good share of the steps.
    assert!(taken > 100, "{taken}");
}

fn spec(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

/// D[i,j] = sum_k A[i,k] B[k,j] with these layouts and conjugations, run on
/// the backend directly; returns which operands were packed.
fn direct(
    a: (&[usize], &[isize], bool),
    b: (&[usize], &[isize], bool),
    d: (&[usize], &[isize], bool),
    labels: (&[i64], &[i64], &[i64]),
) -> [bool; 3] {
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let op = |(dims, strides, conj): (&[usize], &[isize], bool)| {
        let s = spec(dims, strides);
        if conj {
            s.conj()
        } else {
            s
        }
    };
    let p = Problem::from_labels(
        DType::C64,
        op(a),
        op(b),
        CSpec::Absent,
        op(d),
        &Labels::new(labels.0, labels.1, labels.2),
    )
    .unwrap();
    let step = <Blas as StepBackend<Complex64>>::plan(&all(), &p).unwrap();
    let packed = step.packed();
    let len = |(dims, strides, _): (&[usize], &[isize], bool)| {
        1 + dims
            .iter()
            .zip(strides)
            .map(|(&e, &s)| (e - 1) * s as usize)
            .sum::<usize>()
    };
    let mk = |x: (&[usize], &[isize], bool), labels: &[i64], rng: &mut ChaCha8Rng| {
        let data: Vec<Complex64> = (0..len(x)).map(|_| Elem::random(rng)).collect();
        let mut o = Operand::new(data, x.0, x.1, 0);
        o.labels = labels.to_vec();
        if x.2 {
            o.data.iter_mut().for_each(|v| *v = v.conj());
        }
        o
    };
    let (oa, ob) = (mk(a, labels.0, &mut rng), mk(b, labels.1, &mut rng));
    let out_dims = d.0;
    let mut expected = naive(&[oa.clone(), ob.clone()], labels.2, out_dims);
    if d.2 {
        expected.iter_mut().for_each(|v| *v = v.conj());
    }
    // Run on the unconjugated data.
    let raw = |o: &Operand<Complex64>, conj: bool| -> Vec<Complex64> {
        o.data
            .iter()
            .map(|v| if conj { v.conj() } else { *v })
            .collect()
    };
    let (ra, rb) = (raw(&oa, a.2), raw(&ob, b.2));
    let mut dd = vec![Complex64::new(f64::NAN, 0.0); len(d)];
    let mut work =
        vec![Complex64::new(f64::NAN, 0.0); <Blas as StepBackend<Complex64>>::work_len(&step)];
    <Blas as StepBackend<Complex64>>::execute(
        &all(),
        &step,
        &Exec::serial(),
        (&ra, 0),
        (&rb, 0),
        (&mut dd, 0),
        &mut work,
    )
    .unwrap();
    let got = Operand::new(dd, d.0, d.1, 0);
    close(&col_major_values(&got), &expected, 1e-12);
    packed
}

#[test]
fn in_place_transposed_and_packed_operands() {
    let mm = (&[0i64, 2][..], &[2i64, 1][..], &[0i64, 1][..]);
    let (m, n, k) = (5usize, 6usize, 7usize);
    let cm = |r: usize, c: usize| ([r, c], [1isize, r as isize]);
    let rm = |r: usize, c: usize| ([r, c], [c as isize, 1isize]);
    let (a, b, d) = (cm(m, k), cm(k, n), cm(m, n));
    // Column-major everything: no copy.
    let p = direct(
        (&a.0, &a.1, false),
        (&b.0, &b.1, false),
        (&d.0, &d.1, false),
        mm,
    );
    assert_eq!(p, [false; 3]);
    // Row-major output: computed transposed, no copy.
    let dr = rm(m, n);
    let p = direct(
        (&a.0, &a.1, false),
        (&b.0, &b.1, false),
        (&dr.0, &dr.1, false),
        mm,
    );
    assert_eq!(p, [false; 3]);
    // Row-major inputs read transposed.
    let (ar, br) = (rm(m, k), rm(k, n));
    let p = direct(
        (&ar.0, &ar.1, false),
        (&br.0, &br.1, false),
        (&d.0, &d.1, false),
        mm,
    );
    assert_eq!(p, [false; 3]);
    // A conjugated row-major A is a ConjTrans; a conjugated column-major one
    // is packed.
    let p = direct(
        (&ar.0, &ar.1, true),
        (&b.0, &b.1, false),
        (&d.0, &d.1, false),
        mm,
    );
    assert_eq!(p, [false; 3]);
    let p = direct(
        (&a.0, &a.1, true),
        (&b.0, &b.1, true),
        (&d.0, &d.1, false),
        mm,
    );
    assert_eq!(p, [true, true, false]);
    // A conjugated output conjugates both inputs.
    let p = direct(
        (&ar.0, &ar.1, false),
        (&br.0, &br.1, false),
        (&d.0, &d.1, true),
        mm,
    );
    assert_eq!(p, [false; 3]);
    // Strided columns on both axes: packed.
    let gap = ([m, k], [2isize, 2 * m as isize]);
    let dgap = ([m, n], [2isize, 2 * m as isize]);
    let p = direct(
        (&gap.0, &gap.1, false),
        (&b.0, &b.1, false),
        (&dgap.0, &dgap.1, false),
        mm,
    );
    assert_eq!(p, [true, false, true]);
}

#[test]
fn execution_checks_bounds() {
    let p = Problem::from_labels(
        DType::F64,
        spec(&[8, 8], &[1, 8]),
        spec(&[8, 8], &[1, 8]),
        CSpec::Absent,
        spec(&[8, 8], &[1, 8]),
        &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
    )
    .unwrap();
    let step = <Blas as StepBackend<f64>>::plan(&all(), &p).unwrap();
    let (a, short) = (vec![1.0; 64], vec![1.0; 63]);
    let mut d = vec![0.0; 64];
    let run = |a: (&[f64], isize), d: &mut [f64]| {
        <Blas as StepBackend<f64>>::execute(
            &all(),
            &step,
            &Exec::serial(),
            a,
            (a.0, 0),
            (d, 0),
            &mut [],
        )
    };
    assert!(run((&short, 0), &mut d).is_err());
    assert!(run((&a, 1), &mut d).is_err());
    assert!(run((&a, 0), &mut d[..10]).is_err());
    assert!(run((&a, 0), &mut d).is_ok());
    assert!(d.iter().all(|&x| x == 8.0));
}
