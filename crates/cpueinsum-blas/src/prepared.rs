//! Concrete prepared binary execution, with native alternatives selected before writes.

use core::mem::MaybeUninit;

use cpueinsum::tprims_contract::api::{CSpec, OperandId, Problem};
use cpueinsum::tprims_contract::{
    ExecutionRoute as NativeRoute, PlanConfig, SliceAccumulationSource,
};
use cpueinsum::Exec;

use crate::{BlasError, BlasScalar, BlasStep, Side};

/// The numerical route actually executed by a prepared BLAS adapter.
///
/// See [`BinaryPlan`] for an executable example observing this route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionRoute {
    /// Coordinator-inline CBLAS execution.
    Vendor,
    /// A prepared native alternative, with its observed lower route.
    Native(NativeRoute),
}

/// Immutable prepared binary metadata; execution resources belong to the caller.
///
/// Unsupported vendor layouts and C modes are prepared as native alternatives.
/// Alpha-zero calls use that alternative without reading unused operands.
///
/// # Examples
///
/// ```
/// # extern crate blas_src as _;
/// use cpueinsum::{Exec, tprims_contract::{PlanConfig, SliceAccumulationSource}};
/// use cpueinsum::tprims_contract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
/// use cpueinsum_blas::{BinaryPlan, ExecutionRoute};
/// use std::mem::MaybeUninit;
/// let operand = |dims: &[usize], strides: &[isize]| -> Result<_, cpueinsum::tprims_contract::Error> {
///     Ok(OperandSpec::new(LayoutSpec::new(dims, strides, 0)?))
/// };
/// let problem = Problem::from_labels(DType::F64,
///     operand(&[2, 2], &[1, 2])?, operand(&[2, 2], &[1, 2])?, CSpec::Absent,
///     operand(&[2, 2], &[1, 2])?, &Labels::new(&[0, 1], &[1, 2], &[0, 2]))?;
/// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default())?;
/// let mut work = vec![0.0; plan.work_len()];
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 1.0];
/// let mut d = [0.0; 4];
/// let route = plan.execute_slices_accum(&Exec::serial(), 2.0, (&a, 0), (&b, 0),
///     0.0, SliceAccumulationSource::Absent, (&mut d, 0), &mut work)?;
/// assert_eq!(route, ExecutionRoute::Vendor);
/// assert_eq!(d, [2.0, 4.0, 6.0, 8.0]);
/// let mut fresh = [MaybeUninit::uninit(); 4];
/// let (initialized, _) = plan.execute_uninit_slices(&Exec::serial(), 1.0, (&a, 0), (&b, 0),
///     0.0, SliceAccumulationSource::Absent, (&mut fresh, 0), &mut work)?;
/// assert_eq!(initialized, &a);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct BinaryPlan<T: BlasScalar> {
    native: cpueinsum::BinaryPlan<T>,
    vendor: Option<VendorPlan>,
}

/// Shared concrete vendor metadata for binary and grouped adapters.
#[derive(Debug)]
pub(crate) struct VendorPlan {
    step: BlasStep,
    c: CSpec,
    conjugate_output: bool,
    output_count: Option<usize>,
}

impl<T: BlasScalar> BinaryPlan<T> {
    /// Prepare native alternatives and an optional direct-output vendor route.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Prepare`] when native preparation rejects the dtype,
    /// configuration or problem. No execution resources are retained.
    pub fn new(problem: &Problem, config: &PlanConfig) -> Result<Self, BlasError> {
        let native = cpueinsum::BinaryPlan::new(problem, config)
            .map_err(|source| BlasError::Prepare { source })?;
        Ok(Self {
            native,
            vendor: VendorPlan::new::<T>(problem, config),
        })
    }

    /// Required caller packing workspace in elements; native alternatives use Exec's workspace.
    pub fn work_len(&self) -> usize {
        self.vendor.as_ref().map_or(0, VendorPlan::work_len)
    }

    /// Update initialized output using prepared layouts and explicit packing storage.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Native`] on a selected native failure,
    /// [`BlasError::Accumulation`] for mismatched C metadata or nonzero beta
    /// without C, [`BlasError::OutOfBounds`] for operand spans,
    /// [`BlasError::Work`] for short packing storage and [`BlasError::Pack`]
    /// for invalid packing views. Preflight failures leave output unchanged.
    pub fn execute_slices_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d: (&mut [T], isize),
        work: &mut [T],
    ) -> Result<ExecutionRoute, BlasError> {
        let Some(plan) = self.vendor.as_ref().filter(|_| alpha != T::default()) else {
            return self
                .native
                .execute_slices_accum(exec, alpha, a, b, beta, source, d)
                .map(ExecutionRoute::Native)
                .map_err(|source| BlasError::Native { source });
        };
        plan.preflight(a, b, beta, source, d.0.len(), d.1, work)?;
        plan.run_vendor(alpha, a, b, beta, d.0.as_mut_ptr(), d.1, work)?;
        Ok(ExecutionRoute::Vendor)
    }

    /// Overwrite fresh output, publishing initialized storage only after full completion.
    ///
    /// # Errors
    ///
    /// As [`Self::execute_slices_accum`], plus [`BlasError::FreshCoverage`] when
    /// output storage includes holes or padding, and [`BlasError::Accumulation`]
    /// when old output values would be read. Native alternatives retain their
    /// own full-write and execution-admission checks.
    pub fn execute_uninit_slices<'d>(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d: (&'d mut [MaybeUninit<T>], isize),
        work: &mut [T],
    ) -> Result<(&'d mut [T], ExecutionRoute), BlasError> {
        let Some(plan) = self.vendor.as_ref().filter(|_| alpha != T::default()) else {
            return self
                .native
                .execute_uninit_slices(exec, alpha, a, b, beta, source, d)
                .map(|(out, route)| (out, ExecutionRoute::Native(route)))
                .map_err(|source| BlasError::Native { source });
        };
        plan.preflight(a, b, beta, source, d.0.len(), d.1, work)?;
        if beta != T::default() {
            return Err(BlasError::Accumulation);
        }
        let (lo, hi) = plan.step.spans[2];
        if plan.output_count != Some(d.0.len())
            || lo + d.1 as i128 != 0
            || hi + d.1 as i128 + 1 != d.0.len() as i128
        {
            return Err(BlasError::FreshCoverage);
        }
        plan.run_vendor(alpha, a, b, beta, d.0.as_mut_ptr().cast::<T>(), d.1, work)?;
        // SAFETY: planning proves injectivity over the reduced output domain;
        // count and endpoints above prove that domain covers every physical slot.
        // Direct-D beta-zero GEMMs form no initialized D references/read no D,
        // and successful completion wrote every slot with valid T values.
        let initialized =
            unsafe { core::slice::from_raw_parts_mut(d.0.as_mut_ptr().cast::<T>(), d.0.len()) };
        Ok((initialized, ExecutionRoute::Vendor))
    }
}

impl VendorPlan {
    pub(crate) fn new<T: BlasScalar>(problem: &Problem, config: &PlanConfig) -> Option<Self> {
        let step = crate::layout::plan_prepared::<T>(problem)?;
        if config.no_materialize && step.packed().contains(&true) {
            return None;
        }
        let roles = problem.roles();
        let output_count = roles
            .m()
            .iter()
            .chain(roles.n())
            .chain(roles.h())
            .try_fold(1usize, |count, axis| count.checked_mul(axis.extent()));
        Some(Self {
            step,
            c: problem.c_spec().clone(),
            conjugate_output: T::COMPLEX && problem.d().op().is_conj(),
            output_count,
        })
    }

    pub(crate) fn work_len(&self) -> usize {
        self.step.work_len
    }

    pub(crate) fn preflight<T: BlasScalar>(
        &self,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d_len: usize,
        d_origin: isize,
        work: &mut [T],
    ) -> Result<(), BlasError> {
        let step = &self.step;
        let compatible = match (&self.c, source) {
            (_, SliceAccumulationSource::Absent) => beta == T::default(),
            (CSpec::Output(_), SliceAccumulationSource::Output) => true,
            _ => false,
        };
        if !compatible {
            return Err(BlasError::Accumulation);
        }
        crate::check(a.0.len(), a.1, step.spans[0], OperandId::A)?;
        crate::check(b.0.len(), b.1, step.spans[1], OperandId::B)?;
        crate::check(d_len, d_origin, step.spans[2], OperandId::D)?;
        if work.len() < step.work_len {
            return Err(BlasError::Work {
                need: step.work_len,
                got: work.len(),
            });
        }
        // Construct both input and workspace views before any output mutation.
        for side in [&step.left, &step.right] {
            if let Some(pack) = &side.pack {
                let (data, origin) = if side.from == OperandId::A { a } else { b };
                let error = |source| BlasError::Pack {
                    operand: side.from,
                    source,
                };
                strided_view::StridedView::<T>::new(data, &pack.dims, &pack.outer, origin)
                    .map_err(error)?;
                strided_view::StridedViewMut::<T>::new(
                    &mut work[pack.offset..pack.offset + pack.len],
                    &pack.dims,
                    &pack.inner,
                    0,
                )
                .map_err(error)?;
            }
        }
        Ok(())
    }

    pub(crate) fn run_vendor<T: BlasScalar>(
        &self,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        d: *mut T,
        d_origin: isize,
        work: &mut [T],
    ) -> Result<(), BlasError> {
        let step = &self.step;
        for side in [&step.left, &step.right] {
            if let Some(pack) = &side.pack {
                let (data, origin) = if side.from == OperandId::A { a } else { b };
                crate::pack_in(pack, data, origin, work, side.from)?;
            }
        }
        let wp = work.as_mut_ptr();
        let base = |side: &Side| match &side.pack {
            Some(pack) => wp.wrapping_add(pack.offset).cast_const(),
            None => {
                let (data, origin) = if side.from == OperandId::A { a } else { b };
                data.as_ptr().wrapping_offset(origin)
            }
        };
        let (alpha, beta) = if self.conjugate_output {
            (alpha.conjugate(), beta.conjugate())
        } else {
            (alpha, beta)
        };
        crate::gemms(
            step,
            base(&step.left),
            base(&step.right),
            d.wrapping_offset(d_origin),
            alpha,
            beta,
        );
        Ok(())
    }
}
