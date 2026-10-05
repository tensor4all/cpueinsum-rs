//! A CBLAS [`StepBackend`] for cpueinsum.
//!
//! [`Blas`] takes the binary steps of an [`EinsumPlan`](cpueinsum::EinsumPlan)
//! that are matrix products large enough to pay for a BLAS call, and leaves
//! the others (small steps, isolated reductions, Hadamard products) to
//! tprims-contract. A step it takes runs as one column-major GEMM per batch
//! item: in place when the operands' index groups fuse to matrices BLAS can
//! read, else after packing the operands that do not into work space (and
//! copying the output back) with strided-rs.
//!
//! The crate declares the CBLAS symbols (through `cblas-sys`) and links no
//! provider: the final binary links one, for example with
//! `blas-src = { version = "0.14", features = ["openblas"] }` and
//! `extern crate blas_src;`.
//!
//! The CBLAS calls are this crate's only `unsafe` code: every pointer handed
//! to BLAS is checked against the operand's slice at execution, and every
//! dimension and leading dimension at planning.
//!
//! # Examples
//!
//! ```
//! # extern crate blas_src as _;
//! use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Layout, Scratch};
//! use cpueinsum_blas::Blas;
//!
//! // D[i,l] = sum_jk A[i,j] B[j,k] C[k,l], 64 x 64 column-major matrices.
//! let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [2, 3]]).unwrap();
//! let cm = Layout::new(&[64, 64], &[1, 64]).unwrap();
//! let plan = EinsumPlan::<f64, Blas>::with_backend(Blas::default(), &spec, &[cm; 3], cm).unwrap();
//! assert_eq!(plan.backend_steps(), 2);
//! # let x = vec![1.0; 64 * 64];
//! # let mut d = vec![0.0; 64 * 64];
//! # let views: Vec<_> = (0..3).map(|_| cpueinsum::strided_view::StridedView::new(&x, &[64, 64], &[1, 64], 0).unwrap()).collect();
//! # let mut out = cpueinsum::strided_view::StridedViewMut::new(&mut d, &[64, 64], &[1, 64], 0).unwrap();
//! # let mut scratch = Scratch::new();
//! # plan.execute_into(&Exec::serial(), &views, &mut out, &mut scratch).unwrap();
//! # assert_eq!(d[0], 64.0 * 64.0);
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

mod ffi;
mod layout;

use cpueinsum::tprims_contract::api::{OperandId, Problem};
use cpueinsum::{BackendError, Exec, StepBackend};
use strided_view::{StridedView, StridedViewMut};

pub use ffi::BlasScalar;
pub use layout::BlasStep;

use layout::{Pack, Side};

/// Multiply-accumulates of one GEMM below which [`Blas::default`] declines a
/// step, however large its batch.
///
/// The BLAS path is for large matrices; tprims-contract keeps everything
/// else. Each BLAS call costs a fixed overhead, a batch is a loop of calls,
/// and tprims-contract runs a batch in parallel. Set on an M5 Max with
/// Accelerate (`docs/worklogs/2026-10-05-cpueinsum-blas.md`): from 32^3
/// (2^15) BLAS wins or ties at 1 and 4 threads, while 256 items of 16^3
/// (2^12) run 2.2 times faster on tprims-contract at 4 threads.
pub const DEFAULT_MIN_MACS: u64 = 1 << 15;

/// The CBLAS step backend.
///
/// # Examples
///
/// ```
/// # extern crate blas_src as _;
/// use cpueinsum::{EinsumPlan, EinsumSpec, Layout};
/// use cpueinsum_blas::Blas;
///
/// let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
/// let small = Layout::new(&[4, 4], &[1, 4]).unwrap();
/// // A 4x4 product is declined by default and taken with no threshold.
/// let p = EinsumPlan::<f64, Blas>::with_backend(Blas::default(), &spec, &[small, small], small).unwrap();
/// assert_eq!(p.backend_steps(), 0);
/// let p = EinsumPlan::<f64, Blas>::with_backend(Blas::with_min_macs(0), &spec, &[small, small], small).unwrap();
/// assert_eq!(p.backend_steps(), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blas {
    min_macs: u64,
}

impl Default for Blas {
    fn default() -> Self {
        Self::with_min_macs(DEFAULT_MIN_MACS)
    }
}

impl Blas {
    /// A backend that declines steps whose GEMMs have fewer than `min_macs`
    /// multiply-accumulates each.
    pub fn with_min_macs(min_macs: u64) -> Self {
        Self { min_macs }
    }

    /// The GEMM size below which a step is declined.
    pub fn min_macs(&self) -> u64 {
        self.min_macs
    }
}

/// A failure of a step at execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BlasError {
    /// An operand slice does not hold the elements the step addresses.
    #[error("operand {operand} addresses elements outside its slice")]
    OutOfBounds {
        /// The step operand.
        operand: OperandId,
    },
    /// The work space is shorter than the step asked for.
    #[error("work space of {got} elements, {need} needed")]
    Work {
        /// Elements needed.
        need: usize,
        /// Elements given.
        got: usize,
    },
    /// Packing an operand failed.
    #[error("packing operand {operand}: {source}")]
    Pack {
        /// The step operand.
        operand: OperandId,
        /// The strided-rs error.
        source: strided_view::StridedError,
    },
}

impl<T: BlasScalar> StepBackend<T> for Blas {
    type Step = BlasStep;

    fn plan(&self, problem: &Problem) -> Option<BlasStep> {
        // The step's total bounds each GEMM's from above: a cheap early out.
        if problem.macs() < u128::from(self.min_macs) {
            return None;
        }
        let step = layout::plan::<T>(problem)?;
        (step.gemm_macs() >= self.min_macs).then_some(step)
    }

    fn work_len(step: &BlasStep) -> usize {
        step.work_len
    }

    /// Runs on the BLAS library's own threads; `exec` is not used.
    fn execute(
        &self,
        step: &BlasStep,
        _exec: &Exec<'_>,
        a: (&[T], isize),
        b: (&[T], isize),
        d: (&mut [T], isize),
        work: &mut [T],
    ) -> Result<(), BackendError> {
        run(step, a, b, d, work).map_err(Into::into)
    }
}

/// Check that `data` holds the elements `span` addresses from `origin`.
fn check(
    len: usize,
    origin: isize,
    span: (i128, i128),
    operand: OperandId,
) -> Result<(), BlasError> {
    let (lo, hi) = (origin as i128 + span.0, origin as i128 + span.1);
    if lo >= 0 && hi < len as i128 {
        Ok(())
    } else {
        Err(BlasError::OutOfBounds { operand })
    }
}

/// Copy `src` (strides `outer`, from `origin`) into its packed form.
fn pack_in<T: Copy>(
    p: &Pack,
    src: &[T],
    origin: isize,
    work: &mut [T],
    operand: OperandId,
) -> Result<(), BlasError> {
    let err = |source| BlasError::Pack { operand, source };
    let s = StridedView::<T>::new(src, &p.dims, &p.outer, origin).map_err(err)?;
    let mut w = StridedViewMut::new(&mut work[p.offset..p.offset + p.len], &p.dims, &p.inner, 0)
        .map_err(err)?;
    strided_perm::copy_into(&mut w, &s).map_err(err)
}

fn run<T: BlasScalar>(
    step: &BlasStep,
    a: (&[T], isize),
    b: (&[T], isize),
    d: (&mut [T], isize),
    work: &mut [T],
) -> Result<(), BlasError> {
    use OperandId::{A, B, D};
    check(a.0.len(), a.1, step.spans[0], A)?;
    check(b.0.len(), b.1, step.spans[1], B)?;
    check(d.0.len(), d.1, step.spans[2], D)?;
    if work.len() < step.work_len {
        return Err(BlasError::Work {
            need: step.work_len,
            got: work.len(),
        });
    }
    let input = |o: OperandId| if o == A { a } else { b };
    // Pack the inputs, then take the matrices' base pointers (all work-space
    // pointers from one raw pointer, taken after the packs).
    for s in [&step.left, &step.right] {
        if let Some(p) = &s.pack {
            let (data, origin) = input(s.from);
            pack_in(p, data, origin, work, s.from)?;
        }
    }
    let wp = work.as_mut_ptr();
    let base = |s: &Side| -> *const T {
        match &s.pack {
            Some(p) => wp.wrapping_add(p.offset).cast_const(),
            None => {
                let (data, origin) = input(s.from);
                data.as_ptr().wrapping_offset(origin)
            }
        }
    };
    let (pa, pb) = (base(&step.left), base(&step.right));
    let (dslice, dorigin) = d;
    let pd: *mut T = match &step.out.pack {
        Some(p) => wp.wrapping_add(p.offset),
        None => dslice.as_mut_ptr().wrapping_offset(dorigin),
    };

    gemms(step, pa, pb, pd);

    if let Some(p) = &step.out.pack {
        let err = |source| BlasError::Pack { operand: D, source };
        let s = StridedView::<T>::new(&work[p.offset..p.offset + p.len], &p.dims, &p.inner, 0)
            .map_err(err)?;
        let mut o = StridedViewMut::new(dslice, &p.dims, &p.outer, dorigin).map_err(err)?;
        strided_perm::copy_into(&mut o, &s).map_err(err)?;
    }
    Ok(())
}

/// Run the step's GEMM on every batch item.
fn gemms<T: BlasScalar>(step: &BlasStep, pa: *const T, pb: *const T, pd: *mut T) {
    let g = &step.gemm;
    let (ha, hb, hd) = (&step.left.h, &step.right.h, &step.out.h);
    let rank = step.h.len();
    // Odometer over the batch axes.
    let mut idx = vec![0usize; rank];
    let (mut oa, mut ob, mut od) = (0isize, 0isize, 0isize);
    let items: usize = step.h.iter().product();

    for _ in 0..items {
        let (a, b, c) = (
            pa.wrapping_offset(oa),
            pb.wrapping_offset(ob),
            pd.wrapping_offset(od),
        );
        // SAFETY: the operand slices hold every element the step addresses
        // (checked against the problem's spans, or the packs lie in the work
        // space, which holds `work_len` elements); the batch offsets address
        // items inside them. The output is injective (checked at planning),
        // so items are disjoint, and does not overlap the inputs (the plan
        // places a step's result apart from its operands; packs are disjoint
        // ranges of work space).
        unsafe { T::gemm(g, a, b, c) };
        for i in 0..rank {
            idx[i] += 1;
            oa += ha[i];
            ob += hb[i];
            od += hd[i];
            if idx[i] < step.h[i] {
                break;
            }
            let e = step.h[i] as isize;
            oa -= ha[i] * e;
            ob -= hb[i] * e;
            od -= hd[i] * e;
            idx[i] = 0;
        }
    }
}
