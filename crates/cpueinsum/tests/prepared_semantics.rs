mod common;

use common::{close, naive, Elem, Operand};
use core::mem::MaybeUninit;
use cpueinsum::tprims_contract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem};
use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::{BinaryPlan, Exec};
use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn all_modes<T: Elem>(conj: impl Fn(T) -> T + Copy, tolerance: f64, k: usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let a: Vec<T> = (0..3 * k - 1).map(|_| T::random(&mut rng)).collect();
    let b: Vec<T> = (0..2 * k).map(|_| T::random(&mut rng)).collect();
    let c: Vec<T> = (0..4).map(|_| T::random(&mut rng)).collect();
    let alpha = T::random(&mut rng);
    let beta = T::random(&mut rng);
    let spec = |dims: &[usize], strides: &[isize], origin, op| {
        OperandSpec::new(LayoutSpec::new(dims, strides, origin).unwrap()).with_op(op)
    };
    for config in [PlanConfig::default(), PlanConfig::packed()] {
        for mask in 0..16 {
            let ops: [Op; 4] = core::array::from_fn(|i| {
                if mask & (1 << i) == 0 {
                    Op::Identity
                } else {
                    Op::Conjugate
                }
            });
            let op = |v, index| if mask & (1 << index) != 0 { conj(v) } else { v };
            let reference = naive(
                &[
                    Operand::new(a.iter().map(|&x| op(x, 0)).collect(), &[2, k], &[-1, 3], 1)
                        .labelled(&[0, 2]),
                    Operand::new(
                        b.iter().map(|&x| op(x, 1)).collect(),
                        &[k, 2],
                        &[1, k as isize],
                        0,
                    )
                    .labelled(&[2, 1]),
                ],
                &[0, 1],
                &[2, 2],
            );
            for mode in 0..3 {
                let c_spec = match mode {
                    0 => CSpec::Absent,
                    1 => CSpec::Output(ops[2]),
                    _ => CSpec::Separate(spec(&[2, 2], &[-1, 2], 1, ops[2])),
                };
                let labels = Labels::new(&[0, 2], &[2, 1], &[0, 1]);
                let labels = if mode == 2 {
                    labels.with_c(&[0, 1])
                } else {
                    labels
                };
                let problem = Problem::from_labels(
                    T::STORAGE,
                    spec(&[2, k], &[-1, 3], 1, ops[0]),
                    spec(&[k, 2], &[1, k as isize], 0, ops[1]),
                    c_spec,
                    spec(&[2, 2], &[1, 2], 0, ops[3]),
                    &labels,
                )
                .unwrap();
                let plan = BinaryPlan::<T>::new(&problem, &config).unwrap();
                for scale in [T::default(), beta] {
                    if mode == 0 && scale != T::default() {
                        continue;
                    }
                    let expected: Vec<T> = reference
                        .iter()
                        .enumerate()
                        .map(|(i, &product)| {
                            let term = alpha * product;
                            let with_c = if scale == T::default() {
                                term
                            } else {
                                let old = if mode == 1 { c[i] } else { c[i ^ 1] };
                                let mut value = term;
                                value += scale * op(old, 2);
                                value
                            };
                            op(with_c, 3)
                        })
                        .collect();
                    let source = match mode {
                        0 => SliceAccumulationSource::Absent,
                        1 => SliceAccumulationSource::Output,
                        _ => SliceAccumulationSource::Separate((&c, 1)),
                    };
                    let mut d = c.clone();
                    plan.execute_slices_accum(
                        &Exec::serial(),
                        alpha,
                        (&a, 1),
                        (&b, 0),
                        scale,
                        source,
                        (&mut d, 0),
                    )
                    .unwrap();
                    close(&d, &expected, tolerance);
                    if mode != 1 || scale == T::default() {
                        let mut fresh = vec![MaybeUninit::uninit(); 4];
                        let (initialized, _) = plan
                            .execute_uninit_slices(
                                &Exec::serial(),
                                alpha,
                                (&a, 1),
                                (&b, 0),
                                scale,
                                source,
                                (&mut fresh, 0),
                            )
                            .unwrap();
                        close(initialized, &expected, tolerance);
                    }
                }
            }
        }
    }
}

#[test]
fn f32_semantics() {
    for k in [3, 513] {
        all_modes::<f32>(|x| x, 2e-5, k);
    }
}
#[test]
fn f64_semantics() {
    for k in [3, 513] {
        all_modes::<f64>(|x| x, 1e-12, k);
    }
}
#[test]
fn c32_semantics() {
    for k in [3, 513] {
        all_modes::<Complex32>(|x| x.conj(), 3e-5, k);
    }
}
#[test]
fn c64_semantics() {
    for k in [3, 513] {
        all_modes::<Complex64>(|x| x.conj(), 1e-12, k);
    }
}

#[test]
fn zero_scale_nan_and_empty_k_do_not_read_unused_values() {
    let spec = |dims: &[usize], strides: &[isize]| {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
    };
    for k in [0, 2] {
        let p = Problem::from_labels(
            cpueinsum::tprims_contract::api::DType::F64,
            spec(&[2, k], &[1, 2]),
            spec(&[k, 2], &[1, k.max(1) as isize]),
            CSpec::Separate(spec(&[2, 2], &[1, 2])),
            spec(&[2, 2], &[1, 2]),
            &Labels::new(&[0, 2], &[2, 1], &[0, 1]).with_c(&[0, 1]),
        )
        .unwrap();
        let plan = BinaryPlan::<f64>::new(&p, &PlanConfig::default()).unwrap();
        let a = vec![f64::NAN; 2 * k];
        let b = vec![f64::NAN; 2 * k];
        let c = [f64::NAN; 4];
        let mut fresh = [MaybeUninit::uninit(); 4];
        let alpha = if k == 0 { f64::NAN } else { 0.0 };
        let (d, _) = plan
            .execute_uninit_slices(
                &Exec::serial(),
                alpha,
                (&a, 0),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Separate((&c, 0)),
                (&mut fresh, 0),
            )
            .unwrap();
        assert_eq!(d, &[0.; 4]);
    }
}

#[test]
fn parallel_elementwise_fresh_output_with_broadcast_and_reversal() {
    let (rows, cols) = (1024usize, 512usize);
    let dims = [rows, cols];
    let spec = |strides: &[isize], offset, op| {
        OperandSpec::new(LayoutSpec::new(&dims, strides, offset).unwrap()).with_op(op)
    };
    let p = Problem::from_labels(
        cpueinsum::tprims_contract::api::DType::C64,
        spec(&[1, rows as isize], 0, Op::Conjugate),
        spec(&[0, 0], 0, Op::Conjugate),
        CSpec::Separate(spec(&[0, 0], 0, Op::Conjugate)),
        spec(&[-1, rows as isize], (rows - 1) as isize, Op::Conjugate),
        &Labels::new(&[0, 1], &[0, 1], &[0, 1]).with_c(&[0, 1]),
    )
    .unwrap();
    let plan = BinaryPlan::<Complex64>::new(&p, &PlanConfig::default()).unwrap();
    let a: Vec<_> = (0..rows * cols)
        .map(|i| Complex64::new((i % 7) as f64, (i % 11) as f64))
        .collect();
    let b = [Complex64::new(2.0, 1.0)];
    let alpha = Complex64::new(0.0, 2.0);
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = cpueinsum::Pool::borrow(&tp);
    let exec = Exec::rayon(&pool).with_budget(2).unwrap();
    for beta in [Complex64::default(), Complex64::new(-1.0, 0.5)] {
        let c = [if beta == Complex64::default() {
            Complex64::new(f64::NAN, f64::NAN)
        } else {
            Complex64::new(3.0, -1.0)
        }];
        let mut fresh = vec![MaybeUninit::uninit(); rows * cols];
        let (d, route) = plan
            .execute_uninit_slices(
                &exec,
                alpha,
                (&a, 0),
                (&b, 0),
                beta,
                SliceAccumulationSource::Separate((&c, 0)),
                (&mut fresh, (rows - 1) as isize),
            )
            .unwrap();
        assert_eq!(route, cpueinsum::ExecutionRoute::Elementwise);
        for j in 0..cols {
            for i in 0..rows {
                let term = alpha * a[i + j * rows].conj() * b[0].conj();
                let expected = if beta == Complex64::default() {
                    term
                } else {
                    term + beta * c[0].conj()
                }
                .conj();
                assert_eq!(d[rows - 1 - i + j * rows], expected);
            }
        }
    }
    assert!(
        pool.stats().entries > 0,
        "a pooled route used its supplied 2T-budget context"
    );
    assert_eq!(pool.stats().broadcasts, 0);
}
