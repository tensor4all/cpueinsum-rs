//! The element types the harness can drive.
//!
//! cpueinsum accepts exactly four `Scalar` types; the harness adds deterministic
//! data generation and a norm so two engine rows can be compared. The corpus
//! currently declares only `float64`, but the other three are one generic body.

use cpueinsum::Scalar;
use num_complex::{Complex, Complex32, Complex64};

/// An element type the harness can execute and compare.
pub trait Elem: Scalar + Copy + core::fmt::Debug {
    /// Relative-Frobenius-error bound for a cross-row comparison.
    const TOLERANCE: f64;

    /// A deterministic datum for element `i`, in `[-1, 1)`.
    fn sample(i: usize) -> Self;
    /// The norm of `self - other`.
    fn dist(self, other: Self) -> f64;
    /// The norm of `self`.
    fn mag(self) -> f64;
    /// The real part, for the cross-arm diagnostics.
    fn re(self) -> f64;
    /// The imaginary part (zero for a real dtype).
    fn im(self) -> f64;
}

/// A deterministic value in `[-1, 1)`; no RNG dependency, and the same sequence
/// for every run so both engine rows see identical inputs.
fn unit(i: usize) -> f64 {
    const GOLDEN: f64 = 0.618_033_988_749_894_9;
    let x = (i as f64 + 0.5) * GOLDEN;
    (x - x.floor()) * 2.0 - 1.0
}

impl Elem for f64 {
    const TOLERANCE: f64 = 1e-9;
    fn sample(i: usize) -> Self {
        unit(i)
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).abs()
    }
    fn mag(self) -> f64 {
        self.abs()
    }
    fn re(self) -> f64 {
        self
    }
    fn im(self) -> f64 {
        0.0
    }
}

impl Elem for f32 {
    const TOLERANCE: f64 = 2e-3;
    fn sample(i: usize) -> Self {
        unit(i) as f32
    }
    fn dist(self, other: Self) -> f64 {
        f64::from((self - other).abs())
    }
    fn mag(self) -> f64 {
        f64::from(self.abs())
    }
    fn re(self) -> f64 {
        f64::from(self)
    }
    fn im(self) -> f64 {
        0.0
    }
}

impl Elem for Complex64 {
    const TOLERANCE: f64 = 1e-9;
    fn sample(i: usize) -> Self {
        Complex::new(unit(2 * i), unit(2 * i + 1))
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).norm()
    }
    fn mag(self) -> f64 {
        self.norm()
    }
    fn re(self) -> f64 {
        self.re
    }
    fn im(self) -> f64 {
        self.im
    }
}

impl Elem for Complex32 {
    const TOLERANCE: f64 = 2e-3;
    fn sample(i: usize) -> Self {
        Complex::new(unit(2 * i) as f32, unit(2 * i + 1) as f32)
    }
    fn dist(self, other: Self) -> f64 {
        f64::from((self - other).norm())
    }
    fn mag(self) -> f64 {
        f64::from(self.norm())
    }
    fn re(self) -> f64 {
        f64::from(self.re)
    }
    fn im(self) -> f64 {
        f64::from(self.im)
    }
}

/// A dtype label the corpus may declare, and the type it maps to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// `float32`.
    F32,
    /// `float64`.
    F64,
    /// `complex64`.
    C32,
    /// `complex128`.
    C64,
}

impl Dtype {
    /// The type a corpus dtype label maps to, accepting the common aliases.
    ///
    /// A label outside this set is a declared-but-unsupported census cell, not
    /// an error: the harness records it as `skipped`.
    pub fn of(label: &str) -> Option<Dtype> {
        match label {
            "float32" | "f32" => Some(Dtype::F32),
            "float64" | "f64" => Some(Dtype::F64),
            "complex64" | "c32" => Some(Dtype::C32),
            "complex128" | "c64" => Some(Dtype::C64),
            _ => None,
        }
    }
}
