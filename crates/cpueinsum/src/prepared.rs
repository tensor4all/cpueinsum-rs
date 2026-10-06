//! A binary contraction prepared once and reused across buffers.

use core::mem::MaybeUninit;

use strided_view::{StridedView, StridedViewMut};
use tprims_contract::api::{AccumulationSource, Problem};
use tprims_contract::{
    ExecutionRoute, OutputContract, Plan, PlanConfig, PlanReport, SliceAccumulationSource,
};
use tprims_exec::Exec;

use crate::error::{Error, Result};
use crate::scalar::Scalar;

/// A single binary step's lower-layer failure, as a cpueinsum contract error.
fn contract(source: tprims_contract::Error) -> Error {
    Error::Contract { step: 0, source }
}

/// A binary contraction planned once and executed on many buffers.
///
/// A `BinaryPlan` owns only an immutable tprims-contract [`Plan`] built from a
/// validated [`Problem`] and [`PlanConfig`]. It holds no executor, buffer,
/// arena or lease: each execution borrows the caller's [`Exec`] and operands.
/// The operand operations and C mode are fixed by the [`Problem`]; the caller
/// chooses the runtime buffers and the executor.
///
/// Every execution resolves the numerical [`ExecutionRoute`] on the calling
/// thread *before* writing, then returns it after the write has completed, so
/// a successful call reports the route that actually ran. `T` is one of
/// cpueinsum's [`Scalar`] types and must match the problem's storage type.
///
/// # Examples
///
/// ```
/// use cpueinsum::strided_view::{StridedView, StridedViewMut};
/// use cpueinsum::tprims_contract::api::{
///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
/// };
/// use cpueinsum::tprims_contract::PlanConfig;
/// use cpueinsum::{BinaryPlan, Exec};
/// // D[i, j] = sum_k A[i, k] B[k, j], column-major 2x2.
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// let problem = Problem::from_labels(
///     DType::F64,
///     l(&[2, 2], &[1, 2]),
///     l(&[2, 2], &[1, 2]),
///     CSpec::Absent,
///     l(&[2, 2], &[1, 2]),
///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
/// )
/// .unwrap();
/// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 1.0]; // identity, so D == A
/// let mut d = [0.0; 4];
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
/// let _route = plan
///     .execute_into(&Exec::serial(), 1.0, &av, &bv, &mut dv)
///     .unwrap();
/// assert_eq!(d, [1.0, 2.0, 3.0, 4.0]);
/// ```
#[derive(Debug)]
pub struct BinaryPlan<T: Scalar> {
    plan: Plan<T>,
}

impl<T: Scalar> BinaryPlan<T> {
    /// Prepare `problem` under `config`.
    ///
    /// The plan keeps only layout and configuration metadata: no operand
    /// payload or pointer, and nothing borrowed from an executor. Fresh-output
    /// capability is always prepared, including Faer's packed overwrite route.
    ///
    /// # Errors
    ///
    /// [`Error::Contract`] (step 0) when tprims-contract rejects the problem
    /// or the configuration, including a `T` that does not match the problem's
    /// storage type.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::tprims_contract::api::{
    ///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::PlanConfig;
    /// use cpueinsum::BinaryPlan;
    /// let l = OperandSpec::new(LayoutSpec::new(&[3], &[1], 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F32,
    ///     l.clone(),
    ///     l.clone(),
    ///     CSpec::Absent,
    ///     l,
    ///     &Labels::new(&[0], &[0], &[0]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f32>::new(&problem, &PlanConfig::default()).unwrap();
    /// // A dtype/plan mismatch is caught here, before any execution.
    /// assert!(BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).is_err());
    /// # let _ = plan;
    /// ```
    pub fn new(problem: &Problem, config: &PlanConfig) -> Result<Self> {
        let mut config = config.clone();
        config.fresh_output = true;
        Plan::<T>::new(problem, &config)
            .map(|plan| Self { plan })
            .map_err(contract)
    }

    /// The immutable tprims-contract report of what was prepared.
    ///
    /// This describes the algorithm the plan chose, not an executed route; the
    /// executed route is returned by each execution method.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::tprims_contract::api::{
    ///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::PlanConfig;
    /// use cpueinsum::BinaryPlan;
    /// let l = OperandSpec::new(LayoutSpec::new(&[2], &[1], 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     l.clone(),
    ///     l.clone(),
    ///     CSpec::Absent,
    ///     l,
    ///     &Labels::new(&[0], &[0], &[0]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// let _algorithm = plan.report().algorithm;
    /// ```
    pub fn report(&self) -> &PlanReport {
        self.plan.report()
    }

    /// Resolve the numerical route before writes, then return it on success.
    fn route(&self, exec: &Exec<'_>, alpha: T, output: OutputContract) -> Result<ExecutionRoute> {
        self.plan
            .execution_route(exec, alpha, output)
            .map_err(contract)
    }

    /// Overwrite `d` with `op_D(alpha * dot_general(op_A(a), op_B(b)))`.
    ///
    /// No previous value of `d` is read. The views must match the prepared
    /// layouts and carry initialized storage.
    ///
    /// # Errors
    ///
    /// [`Error::Contract`] (step 0) when a view does not match the plan, the
    /// executor cannot serve the resolved route, or a lower layer fails; all
    /// validation errors leave `d` unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::strided_view::{StridedView, StridedViewMut};
    /// use cpueinsum::tprims_contract::api::{
    ///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::PlanConfig;
    /// use cpueinsum::{BinaryPlan, Exec};
    /// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     l(&[2, 2], &[1, 2]),
    ///     l(&[2, 2], &[1, 2]),
    ///     CSpec::Absent,
    ///     l(&[2, 2], &[1, 2]),
    ///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [1.0, 0.0, 0.0, 1.0];
    /// let mut d = [0.0; 4];
    /// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
    /// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
    /// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
    /// plan.execute_into(&Exec::serial(), 2.0, &av, &bv, &mut dv).unwrap();
    /// assert_eq!(d, [2.0, 4.0, 6.0, 8.0]);
    /// ```
    pub fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<ExecutionRoute> {
        let route = self.route(exec, alpha, OutputContract::Initialized)?;
        self.plan
            .execute_into(exec, alpha, a, b, d)
            .map_err(contract)?;
        Ok(route)
    }

    /// Update `d` with `op_D(alpha * dot_general(op_A(a), op_B(b)) + beta * op_C(C))`.
    ///
    /// `source` must match the prepared C mode:
    /// [`AccumulationSource::Output`] reads the previous `d`,
    /// [`AccumulationSource::Separate`] reads a distinct C view, and
    /// [`AccumulationSource::Absent`] is accepted only with zero `beta`.
    /// `beta == 0` reads no C value; `alpha == 0` or an empty contraction reads
    /// no A or B value.
    ///
    /// # Errors
    ///
    /// As [`BinaryPlan::execute_into`], plus a C-mode error when `source` does
    /// not match the prepared mode or a nonzero/NaN `beta` meets an absent
    /// source.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::strided_view::{StridedView, StridedViewMut};
    /// use cpueinsum::tprims_contract::api::{
    ///     AccumulationSource, CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::PlanConfig;
    /// use cpueinsum::{BinaryPlan, Exec};
    /// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     l(&[2, 2], &[1, 2]),
    ///     l(&[2, 2], &[1, 2]),
    ///     CSpec::Output(Op::Identity),
    ///     l(&[2, 2], &[1, 2]),
    ///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [1.0, 0.0, 0.0, 1.0];
    /// let mut d = [10.0; 4];
    /// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
    /// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
    /// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
    /// plan.execute_into_accum(
    ///     &Exec::serial(), 1.0, &av, &bv, 2.0,
    ///     AccumulationSource::Output, &mut dv,
    /// )
    /// .unwrap();
    /// assert_eq!(d, [21.0, 22.0, 23.0, 24.0]); // A + 2 * old_d
    /// ```
    pub fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<ExecutionRoute> {
        let route = self.route(exec, alpha, OutputContract::Initialized)?;
        self.plan
            .execute_into_accum(exec, alpha, a, b, beta, source, d)
            .map_err(contract)?;
        Ok(route)
    }

    /// [`BinaryPlan::execute_into`] on plain slices, without building views.
    ///
    /// Each operand is a slice and the index of its logical origin in that
    /// slice, as [`StridedView::new`] takes them; the prepared extents and
    /// strides address the rest. This is the entry for a caller that holds
    /// only buffers and keeps layouts in the plan.
    ///
    /// # Errors
    ///
    /// [`Error::Contract`] (step 0) when a slice does not cover the range its
    /// layout addresses from the given origin, the executor cannot serve the
    /// resolved route, or a lower layer fails; nothing is written on a bounds
    /// error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::tprims_contract::api::{
    ///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::PlanConfig;
    /// use cpueinsum::{BinaryPlan, Exec};
    /// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     l(&[2, 2], &[1, 2]),
    ///     l(&[2, 2], &[1, 2]),
    ///     CSpec::Absent,
    ///     l(&[2, 2], &[1, 2]),
    ///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// let (a, b) = ([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0]);
    /// let mut d = [0.0; 4];
    /// let exec = Exec::serial();
    /// plan.execute_slices(&exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0)).unwrap();
    /// assert_eq!(d, [1.0, 2.0, 3.0, 4.0]);
    /// // A slice that cannot cover the planned range is refused before writing.
    /// assert!(plan.execute_slices(&exec, 1.0, (&a[..2], 0), (&b, 0), (&mut d, 0)).is_err());
    /// ```
    pub fn execute_slices(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        d: (&mut [T], isize),
    ) -> Result<ExecutionRoute> {
        let route = self.route(exec, alpha, OutputContract::Initialized)?;
        self.plan
            .execute_slices(exec, alpha, a, b, d)
            .map_err(contract)?;
        Ok(route)
    }

    /// [`BinaryPlan::execute_into_accum`] on plain slices.
    ///
    /// The accumulation source is a
    /// [`SliceAccumulationSource`] carrying its own origin: `Output` reads the
    /// previous `d`, `Separate((c, origin))` reads a distinct initialized
    /// slice, and `Absent` is accepted only with zero `beta`.
    ///
    /// # Errors
    ///
    /// As [`BinaryPlan::execute_slices`], plus a C-mode error when the source
    /// does not match the prepared mode or a nonzero/NaN `beta` meets an absent
    /// source.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::tprims_contract::api::{
    ///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
    /// use cpueinsum::{BinaryPlan, Exec};
    /// // A scalar contraction d = sum_k a[k] * b[k] with a separate C.
    /// let v = || OperandSpec::new(LayoutSpec::new(&[1], &[1], 0).unwrap());
    /// let s = || OperandSpec::new(LayoutSpec::new(&[], &[], 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     v(),
    ///     v(),
    ///     CSpec::Separate(s()),
    ///     s(),
    ///     &Labels::new(&[0], &[0], &[]).with_c(&[]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// let mut d = [5.0];
    /// plan.execute_slices_accum(
    ///     &Exec::serial(), 1.0, (&[3.0], 0), (&[4.0], 0), 2.0,
    ///     SliceAccumulationSource::Separate((&[7.0], 0)),
    ///     (&mut d, 0),
    /// )
    /// .unwrap();
    /// assert_eq!(d, [3.0 * 4.0 + 2.0 * 7.0]);
    /// ```
    pub fn execute_slices_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d: (&mut [T], isize),
    ) -> Result<ExecutionRoute> {
        let route = self.route(exec, alpha, OutputContract::Initialized)?;
        self.plan
            .execute_slices_accum(exec, alpha, a, b, beta, source, d)
            .map_err(contract)?;
        Ok(route)
    }

    /// Overwrite fresh storage and return it as initialized `d` plus the route.
    ///
    /// `d` must have dense exact physical coverage (a dense permutation or
    /// reversal, no holes or padding); the lower layer proves coverage and
    /// writes without forming an initialized reference first. On success the
    /// returned slice is initialized and the route is the one resolved before
    /// the write. Nonzero-beta accumulation from previous D is rejected;
    /// nonzero-beta Separate C remains supported.
    ///
    /// # Errors
    ///
    /// [`Error::Contract`] (step 0) for incomplete coverage, an output/nonzero
    /// `beta` combination, a C-mode mismatch, bounds, or lower execution
    /// failure; no initialized slice escapes on failure, and the storage stays
    /// typed as [`MaybeUninit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use cpueinsum::tprims_contract::api::{
    ///     CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem,
    /// };
    /// use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
    /// use cpueinsum::{BinaryPlan, Exec};
    /// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     l(&[2, 2], &[1, 2]),
    ///     l(&[2, 2], &[1, 2]),
    ///     CSpec::Absent,
    ///     l(&[2, 2], &[1, 2]),
    ///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    /// )
    /// .unwrap();
    /// let plan = BinaryPlan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [1.0, 0.0, 0.0, 1.0];
    /// let mut d = [MaybeUninit::uninit(); 4];
    /// let (initialized, _route) = plan
    ///     .execute_uninit_slices(
    ///         &Exec::serial(), 1.0, (&a, 0), (&b, 0), 0.0,
    ///         SliceAccumulationSource::Absent, (&mut d, 0),
    ///     )
    ///     .unwrap();
    /// assert_eq!(initialized, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn execute_uninit_slices<'d>(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d: (&'d mut [MaybeUninit<T>], isize),
    ) -> Result<(&'d mut [T], ExecutionRoute)> {
        let route = self.route(exec, alpha, OutputContract::Fresh)?;
        let initialized = self
            .plan
            .execute_uninit_slices(exec, alpha, a, b, beta, source, d)
            .map_err(contract)?;
        Ok((initialized, route))
    }
}
