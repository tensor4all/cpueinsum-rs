//! A naive einsum and operand helpers for the reference tests.

#![allow(dead_code)]

use cpueinsum::strided_view::StridedView;
use cpueinsum::Layout;
use num_complex::{Complex32, Complex64};
use rand::Rng;

/// An element the reference supports.
pub trait Elem: cpueinsum::Scalar + std::ops::Mul<Output = Self> + std::ops::AddAssign {
    fn random(rng: &mut impl Rng) -> Self;
    fn zero() -> Self;
    fn dist(self, other: Self) -> f64;
    fn mag(self) -> f64;
}

impl Elem for f64 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn zero() -> Self {
        0.0
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).abs()
    }
    fn mag(self) -> f64 {
        self.abs()
    }
}

impl Elem for Complex64 {
    fn random(rng: &mut impl Rng) -> Self {
        Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).norm()
    }
    fn mag(self) -> f64 {
        self.norm()
    }
}

impl Elem for f32 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn zero() -> Self {
        0.0
    }
    fn dist(self, other: Self) -> f64 {
        f64::from((self - other).abs())
    }
    fn mag(self) -> f64 {
        f64::from(self.abs())
    }
}

impl Elem for Complex32 {
    fn random(rng: &mut impl Rng) -> Self {
        Complex32::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }
    fn dist(self, other: Self) -> f64 {
        f64::from((self - other).norm())
    }
    fn mag(self) -> f64 {
        f64::from(self.norm())
    }
}

/// An owned buffer with a strided layout and, optionally, labels.
#[derive(Clone, Debug)]
pub struct Operand<T> {
    pub data: Vec<T>,
    pub dims: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub labels: Vec<i64>,
}

pub fn col_major(dims: &[usize]) -> (Vec<isize>, usize) {
    let mut strides = Vec::new();
    let mut len = 1usize;
    for &e in dims {
        strides.push(len as isize);
        len *= e;
    }
    (strides, len)
}

impl<T: Elem> Operand<T> {
    pub fn new(data: Vec<T>, dims: &[usize], strides: &[isize], offset: usize) -> Self {
        Self {
            data,
            dims: dims.to_vec(),
            strides: strides.to_vec(),
            offset,
            labels: Vec::new(),
        }
    }

    pub fn random_col_major(
        rng: &mut impl Rng,
        labels: &[i64],
        ext: &dyn Fn(i64) -> usize,
    ) -> Self {
        let dims: Vec<usize> = labels.iter().map(|&l| ext(l)).collect();
        let (strides, len) = col_major(&dims);
        let data = (0..len).map(|_| T::random(rng)).collect();
        Self {
            data,
            dims,
            strides,
            offset: 0,
            labels: labels.to_vec(),
        }
    }

    pub fn labelled(&self, labels: &[i64]) -> Self {
        let mut o = self.clone();
        o.labels = labels.to_vec();
        o
    }

    pub fn view(&self) -> StridedView<'_, T> {
        StridedView::new(&self.data, &self.dims, &self.strides, self.offset as isize).unwrap()
    }

    pub fn layout(&self) -> Layout<'_> {
        Layout::new(&self.dims, &self.strides).unwrap()
    }

    pub fn at(&self, index: &[usize]) -> T {
        let mut p = self.offset as isize;
        for (&i, &s) in index.iter().zip(&self.strides) {
            p += i as isize * s;
        }
        self.data[p as usize]
    }
}

/// The einsum by brute force: column-major output.
pub fn naive<T: Elem>(ops: &[Operand<T>], output: &[i64], out_dims: &[usize]) -> Vec<T> {
    let mut labels: Vec<i64> = ops.iter().flat_map(|o| o.labels.iter().copied()).collect();
    labels.sort_unstable();
    labels.dedup();
    let mut ext = vec![0usize; labels.len()];
    for o in ops {
        for (&l, &e) in o.labels.iter().zip(&o.dims) {
            ext[labels.binary_search(&l).unwrap()] = e;
        }
    }
    let pos = |l: i64| labels.binary_search(&l).unwrap();
    let (_, len) = col_major(out_dims);
    let mut out = vec![<T as Elem>::zero(); len];
    if ext.contains(&0) {
        return out;
    }
    let mut value = vec![0usize; labels.len()];
    let mut idx = Vec::new();
    loop {
        let mut prod = T::ONE;
        for o in ops {
            idx.clear();
            idx.extend(o.labels.iter().map(|&l| value[pos(l)]));
            prod = prod * o.at(&idx);
        }
        let mut p = 0usize;
        let mut stride = 1usize;
        for (&l, &e) in output.iter().zip(out_dims) {
            p += value[pos(l)] * stride;
            stride *= e;
        }
        out[p] += prod;
        // Odometer over all labels.
        let mut k = 0;
        loop {
            if k == value.len() {
                return out;
            }
            value[k] += 1;
            if value[k] < ext[k] {
                break;
            }
            value[k] = 0;
            k += 1;
        }
    }
}

pub fn close<T: Elem>(got: &[T], expected: &[T], tol: f64) {
    assert_eq!(got.len(), expected.len());
    for (k, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let err = g.dist(e);
        assert!(
            err <= tol * (1.0 + e.mag()),
            "element {k}: got {:?}, expected {:?}",
            g,
            e
        );
    }
}
