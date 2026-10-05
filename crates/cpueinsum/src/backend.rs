//! The step-backend seam: a second implementation for some binary steps.

use core::convert::Infallible;
use core::fmt::Debug;

use tprims_contract::api::Problem;
use tprims_exec::Exec;

use crate::scalar::Scalar;

/// The error a [`StepBackend`] reports when it fails a step it took.
pub type BackendError = Box<dyn std::error::Error + Send + Sync>;

/// An implementation of some binary steps of an [`EinsumPlan`](crate::EinsumPlan),
/// next to tprims-contract.
///
/// [`EinsumPlan::with_backend`](crate::EinsumPlan::with_backend) offers every
/// step to the backend once, at planning: [`plan`](Self::plan) either takes
/// the step, returning the state it needs to run it, or declines it
/// (`None`), and a declined step is planned and run by tprims-contract as
/// without a backend. The backend owns the routing decision, including the
/// size below which it declines. Execution never declines.
///
/// The plan is generic over the backend, so routing costs one branch per step
/// and no dynamic call. The default backend, [`Tprims`], declines every step.
///
/// A step's operands arrive as data slices with an origin, laid out as the
/// [`Problem`] describes (its `LayoutSpec` offsets are zero), and the step
/// writes `d = a * b` over its labels, overwriting `d` and reading no previous
/// value of it. A step that needs room for packed operands asks for
/// [`work_len`](Self::work_len) elements; the plan places them in its scratch
/// buffer, disjoint from every operand of the step.
///
/// # Examples
///
/// ```
/// use cpueinsum::tprims_contract::api::Problem;
/// use cpueinsum::{BackendError, Exec, StepBackend};
///
/// /// Takes no step.
/// #[derive(Debug)]
/// struct Never;
///
/// impl StepBackend<f64> for Never {
///     type Step = ();
///     fn plan(&self, _: &Problem) -> Option<()> {
///         None
///     }
///     fn work_len(_: &()) -> usize {
///         0
///     }
///     fn execute(
///         &self,
///         _: &(),
///         _: &Exec<'_>,
///         _: (&[f64], isize),
///         _: (&[f64], isize),
///         _: (&mut [f64], isize),
///         _: &mut [f64],
///     ) -> Result<(), BackendError> {
///         unreachable!("every step is declined")
///     }
/// }
/// ```
pub trait StepBackend<T: Scalar>: Debug {
    /// What the backend keeps for a step it took.
    type Step: Debug;

    /// Take the step `problem` (returning its state) or decline it (`None`).
    fn plan(&self, problem: &Problem) -> Option<Self::Step>;

    /// Elements of work space the step needs during execution.
    fn work_len(step: &Self::Step) -> usize;

    /// Run a step this backend took.
    ///
    /// `a`, `b` and `d` are data slices with the element origin of the
    /// operand; `work` has [`work_len`](Self::work_len) elements of
    /// unspecified contents.
    ///
    /// # Errors
    ///
    /// Whatever the backend fails with; the plan reports it as
    /// [`Error::Backend`](crate::Error::Backend).
    fn execute(
        &self,
        step: &Self::Step,
        exec: &Exec<'_>,
        a: (&[T], isize),
        b: (&[T], isize),
        d: (&mut [T], isize),
        work: &mut [T],
    ) -> Result<(), BackendError>;
}

/// The default backend: every step runs on tprims-contract.
///
/// # Examples
///
/// ```
/// use cpueinsum::{EinsumPlan, EinsumSpec, Layout, Tprims};
/// let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
/// let cm = Layout::new(&[2, 2], &[1, 2]).unwrap();
/// let a = EinsumPlan::<f64>::new(&spec, &[cm, cm], cm).unwrap();
/// let b = EinsumPlan::<f64, Tprims>::with_backend(Tprims, &spec, &[cm, cm], cm).unwrap();
/// assert_eq!(a.backend_steps(), 0);
/// assert_eq!(b.backend_steps(), 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tprims;

impl<T: Scalar> StepBackend<T> for Tprims {
    type Step = Infallible;

    #[inline]
    fn plan(&self, _: &Problem) -> Option<Infallible> {
        None
    }

    fn work_len(step: &Infallible) -> usize {
        match *step {}
    }

    fn execute(
        &self,
        step: &Infallible,
        _: &Exec<'_>,
        _: (&[T], isize),
        _: (&[T], isize),
        _: (&mut [T], isize),
        _: &mut [T],
    ) -> Result<(), BackendError> {
        match *step {}
    }
}
