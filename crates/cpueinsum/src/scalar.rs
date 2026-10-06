//! The element types cpueinsum accepts.

use num_complex::Complex;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// An element type: `f32`, `f64`, `Complex<f32>` or `Complex<f64>`.
///
/// Sealed, and exactly the storage types of [`tprims_contract::api::Scalar`].
///
/// # Examples
///
/// ```
/// use cpueinsum::Scalar;
/// assert_eq!(<f64 as Scalar>::ONE, 1.0);
/// assert_eq!(<num_complex::Complex<f32> as Scalar>::ONE.re, 1.0);
/// ```
pub trait Scalar: sealed::Sealed + tprims_contract::api::Scalar + Default + Send + Sync {
    /// The multiplicative identity.
    const ONE: Self;
}

impl Scalar for f32 {
    const ONE: Self = 1.0;
}
impl Scalar for f64 {
    const ONE: Self = 1.0;
}
impl Scalar for Complex<f32> {
    const ONE: Self = Complex::new(1.0, 0.0);
}
impl Scalar for Complex<f64> {
    const ONE: Self = Complex::new(1.0, 0.0);
}
