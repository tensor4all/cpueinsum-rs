//! The CBLAS calls.
//!
//! Ported from tenferro-rs `tenferro-cpu/src/gemm/blas_gemm.rs` at
//! `0b539363`: the CBLAS wrappers, the complex `[re, im]` scalar passing. Layout inference moved to planning
//! ([`crate::layout`]); this module only calls.

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::{Complex32, Complex64};

/// How BLAS reads one input matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum Trans {
    /// As stored, column-major.
    No,
    /// Transposed.
    T,
    /// Conjugate-transposed.
    C,
}

impl Trans {
    fn cblas(self) -> CBLAS_TRANSPOSE {
        match self {
            Trans::No => CBLAS_TRANSPOSE::CblasNoTrans,
            Trans::T => CBLAS_TRANSPOSE::CblasTrans,
            Trans::C => CBLAS_TRANSPOSE::CblasConjTrans,
        }
    }
}

/// One column-major GEMM `C = alpha * op(A) op(B) + beta * C`, every
/// dimension and leading dimension checked against the CBLAS range at planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct Gemm {
    pub(crate) ta: Trans,
    pub(crate) tb: Trans,
    pub(crate) m: i32,
    pub(crate) n: i32,
    pub(crate) k: i32,
    pub(crate) lda: i32,
    pub(crate) ldb: i32,
    pub(crate) ldc: i32,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// An element type with CBLAS GEMM routines: `f32`, `f64`, `Complex32` and
/// `Complex64`.
///
/// Sealed; exactly the element types of [`cpueinsum::Scalar`].
///
/// # Examples
///
/// ```
/// use cpueinsum_blas::BlasScalar;
/// assert!(!<f64 as BlasScalar>::COMPLEX);
/// assert!(<num_complex::Complex32 as BlasScalar>::COMPLEX);
/// ```
pub trait BlasScalar: cpueinsum::Scalar + sealed::Sealed {
    /// Whether conjugation changes a value (for a real type it is ignored).
    const COMPLEX: bool;

    /// Conjugate a scalar (identity for real types).
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum_blas::BlasScalar;
    /// use num_complex::Complex64;
    /// assert_eq!(BlasScalar::conjugate(Complex64::new(2.0, 3.0)), Complex64::new(2.0, -3.0));
    /// ```
    fn conjugate(self) -> Self;

    /// Run `g` on matrices starting at `a`, `b` and `c`.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` address the matrices `g` describes; `c` is exclusive
    /// for them and does not overlap `a` or `b`. The A/B matrices are
    /// initialized. When beta is nonzero C is initialized; with beta zero,
    /// C may be fresh writable storage (CBLAS must not read its old values).
    #[doc(hidden)]
    unsafe fn gemm(g: &Gemm, a: *const Self, b: *const Self, c: *mut Self, alpha: Self, beta: Self);
}

macro_rules! impl_real {
    ($ty:ty, $gemm:path) => {
        impl BlasScalar for $ty {
            const COMPLEX: bool = false;

            fn conjugate(self) -> Self {
                self
            }

            unsafe fn gemm(
                g: &Gemm,
                a: *const Self,
                b: *const Self,
                c: *mut Self,
                alpha: Self,
                beta: Self,
            ) {
                #[cfg(test)]
                tests::record_call();
                // SAFETY: the caller's contract; `g` was checked at planning
                // (dimensions and leading dimensions in range, unit stride on
                // the read axis of each matrix).
                unsafe {
                    $gemm(
                        CBLAS_LAYOUT::CblasColMajor,
                        g.ta.cblas(),
                        g.tb.cblas(),
                        g.m,
                        g.n,
                        g.k,
                        alpha,
                        a,
                        g.lda,
                        b,
                        g.ldb,
                        beta,
                        c,
                        g.ldc,
                    )
                }
            }
        }
    };
}

macro_rules! impl_complex {
    ($ty:ty, $re:ty, $gemm:path) => {
        impl BlasScalar for $ty {
            const COMPLEX: bool = true;

            fn conjugate(self) -> Self {
                self.conj()
            }

            unsafe fn gemm(
                g: &Gemm,
                a: *const Self,
                b: *const Self,
                c: *mut Self,
                alpha: Self,
                beta: Self,
            ) {
                let alpha = [alpha.re, alpha.im];
                let beta = [beta.re, beta.im];
                #[cfg(test)]
                tests::record_call();
                // SAFETY: as the real case; `Complex<_>` is `repr(C)` `[re, im]`,
                // the layout CBLAS reads through `void *`.
                unsafe {
                    $gemm(
                        CBLAS_LAYOUT::CblasColMajor,
                        g.ta.cblas(),
                        g.tb.cblas(),
                        g.m,
                        g.n,
                        g.k,
                        alpha.as_ptr().cast(),
                        a.cast(),
                        g.lda,
                        b.cast(),
                        g.ldb,
                        beta.as_ptr().cast(),
                        c.cast(),
                        g.ldc,
                    )
                }
            }
        }
    };
}

impl_real!(f32, cblas_sys::cblas_sgemm);
impl_real!(f64, cblas_sys::cblas_dgemm);
impl_complex!(Complex32, f32, cblas_sys::cblas_cgemm);
impl_complex!(Complex64, f64, cblas_sys::cblas_zgemm);

#[cfg(test)]
mod tests;
