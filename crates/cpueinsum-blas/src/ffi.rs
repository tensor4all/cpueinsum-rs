//! The CBLAS calls: one GEMM, and the vendor batch of equal GEMMs.
//!
//! Ported from tenferro-rs `tenferro-cpu/src/gemm/blas_gemm.rs` at
//! `0b539363`: the CBLAS wrappers, the complex `[re, im]` scalar passing and
//! the `cblas_?gemm_batch` declarations. Layout inference moved to planning
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

/// One column-major GEMM `C = op(A) op(B)` (alpha one, beta zero), every
/// argument checked against the CBLAS range at planning.
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

    /// Run `g` on matrices starting at `a`, `b` and `c`.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `c` address the matrices `g` describes; `c` is exclusive
    /// for them and does not overlap `a` or `b`.
    #[doc(hidden)]
    unsafe fn gemm(g: &Gemm, a: *const Self, b: *const Self, c: *mut Self);

    /// Run `g` once per pointer triple, in one vendor call.
    ///
    /// # Safety
    ///
    /// As [`gemm`](Self::gemm) for every triple; the `c` matrices are pairwise
    /// disjoint. The three slices have one length that fits `i32`.
    #[cfg(feature = "vendor-batch")]
    #[doc(hidden)]
    unsafe fn gemm_batch(g: &Gemm, a: &[*const Self], b: &[*const Self], c: &[*mut Self]);
}

#[cfg(feature = "vendor-batch")]
mod vendor {
    use super::*;
    use std::ffi::c_void;

    // INVARIANT: the signatures of OpenBLAS's and MKL's `cblas_?gemm_batch`
    // (grouped interface; complex scalars and matrices as `void *`).
    unsafe extern "C" {
        pub(super) fn cblas_sgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const f32,
            a: *const *const f32,
            lda: *const i32,
            b: *const *const f32,
            ldb: *const i32,
            beta: *const f32,
            c: *const *mut f32,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );
        pub(super) fn cblas_dgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const f64,
            a: *const *const f64,
            lda: *const i32,
            b: *const *const f64,
            ldb: *const i32,
            beta: *const f64,
            c: *const *mut f64,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );
        pub(super) fn cblas_cgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const c_void,
            a: *const *const c_void,
            lda: *const i32,
            b: *const *const c_void,
            ldb: *const i32,
            beta: *const c_void,
            c: *const *mut c_void,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );
        pub(super) fn cblas_zgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const c_void,
            a: *const *const c_void,
            lda: *const i32,
            b: *const *const c_void,
            ldb: *const i32,
            beta: *const c_void,
            c: *const *mut c_void,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );
    }
}

macro_rules! impl_real {
    ($ty:ty, $gemm:path, $batch:ident) => {
        impl BlasScalar for $ty {
            const COMPLEX: bool = false;

            unsafe fn gemm(g: &Gemm, a: *const Self, b: *const Self, c: *mut Self) {
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
                        1.0,
                        a,
                        g.lda,
                        b,
                        g.ldb,
                        0.0,
                        c,
                        g.ldc,
                    )
                }
            }

            #[cfg(feature = "vendor-batch")]
            unsafe fn gemm_batch(g: &Gemm, a: &[*const Self], b: &[*const Self], c: &[*mut Self]) {
                let size = a.len() as i32;
                // SAFETY: the caller's contract; one group of `size` equal
                // GEMMs, every descriptor array of length one.
                unsafe {
                    vendor::$batch(
                        CBLAS_LAYOUT::CblasColMajor,
                        &g.ta.cblas(),
                        &g.tb.cblas(),
                        &g.m,
                        &g.n,
                        &g.k,
                        &1.0,
                        a.as_ptr(),
                        &g.lda,
                        b.as_ptr(),
                        &g.ldb,
                        &0.0,
                        c.as_ptr(),
                        &g.ldc,
                        1,
                        &size,
                    )
                }
            }
        }
    };
}

macro_rules! impl_complex {
    ($ty:ty, $re:ty, $gemm:path, $batch:ident) => {
        impl BlasScalar for $ty {
            const COMPLEX: bool = true;

            unsafe fn gemm(g: &Gemm, a: *const Self, b: *const Self, c: *mut Self) {
                let (one, zero): ([$re; 2], [$re; 2]) = ([1.0, 0.0], [0.0, 0.0]);
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
                        one.as_ptr().cast(),
                        a.cast(),
                        g.lda,
                        b.cast(),
                        g.ldb,
                        zero.as_ptr().cast(),
                        c.cast(),
                        g.ldc,
                    )
                }
            }

            #[cfg(feature = "vendor-batch")]
            unsafe fn gemm_batch(g: &Gemm, a: &[*const Self], b: &[*const Self], c: &[*mut Self]) {
                let (one, zero): ([$re; 2], [$re; 2]) = ([1.0, 0.0], [0.0, 0.0]);
                let size = a.len() as i32;
                // SAFETY: as the real case; a pointer array of `*const T` has
                // the layout of one of `*const c_void`.
                unsafe {
                    vendor::$batch(
                        CBLAS_LAYOUT::CblasColMajor,
                        &g.ta.cblas(),
                        &g.tb.cblas(),
                        &g.m,
                        &g.n,
                        &g.k,
                        one.as_ptr().cast(),
                        a.as_ptr().cast(),
                        &g.lda,
                        b.as_ptr().cast(),
                        &g.ldb,
                        zero.as_ptr().cast(),
                        c.as_ptr().cast(),
                        &g.ldc,
                        1,
                        &size,
                    )
                }
            }
        }
    };
}

impl_real!(f32, cblas_sys::cblas_sgemm, cblas_sgemm_batch);
impl_real!(f64, cblas_sys::cblas_dgemm, cblas_dgemm_batch);
impl_complex!(Complex32, f32, cblas_sys::cblas_cgemm, cblas_cgemm_batch);
impl_complex!(Complex64, f64, cblas_sys::cblas_zgemm, cblas_zgemm_batch);
