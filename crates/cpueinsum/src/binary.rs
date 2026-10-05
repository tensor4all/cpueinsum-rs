//! One binary contraction, planned and executed in one call.

use strided_view::{StridedView, StridedViewMut};
use tprims_contract::api::{CSpec, Labels, Problem};
use tprims_contract::{Plan, PlanConfig};
use tprims_exec::Exec;

use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::scalar::Scalar;

/// `d = sum a * b` over integer labels: the overwrite form, reading no previous
/// value of `d`.
///
/// `la`, `lb` and `ld` label the axes of `a`, `b` and `d`. A label repeated
/// within `a` or `b` selects a diagonal; a label in one input only and not in
/// `d` is summed. The plan is built on every call; to execute the same shapes
/// many times, build an [`EinsumPlan`](crate::EinsumPlan) once instead.
///
/// # Errors
///
/// [`Error::Contract`] (step 0) when tprims-contract rejects the labels or the
/// layouts: mismatched label counts or extents, an output-only label, or an
/// output whose axes alias.
///
/// # Examples
///
/// ```
/// use cpueinsum::strided_view::{StridedView, StridedViewMut};
/// use cpueinsum::{contract_into, Exec};
/// // d[i, k] = sum_j a[i, j] b[j, k], column-major 2x2.
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [0.0, 1.0, 1.0, 0.0];
/// let mut d = [0.0; 4];
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
/// contract_into(&Exec::serial(), &av, &[0, 1], &bv, &[1, 2], &mut dv, &[0, 2]).unwrap();
/// assert_eq!(d, [3.0, 4.0, 1.0, 2.0]);
/// ```
pub fn contract_into<T: Scalar>(
    exec: &Exec<'_>,
    a: &StridedView<'_, T>,
    la: &[i64],
    b: &StridedView<'_, T>,
    lb: &[i64],
    d: &mut StridedViewMut<'_, T>,
    ld: &[i64],
) -> Result<()> {
    let plan = plan_step::<T>(
        0,
        Layout::of(a),
        la,
        Layout::of(b),
        lb,
        Layout::of_mut(d),
        ld,
    )?;
    plan.execute_into(exec, T::ONE, a, b, d)
        .map_err(|source| Error::Contract { step: 0, source })
}

/// The tprims plan of one binary step.
pub(crate) fn plan_step<T: Scalar>(
    step: usize,
    a: Layout<'_>,
    la: &[i64],
    b: Layout<'_>,
    lb: &[i64],
    d: Layout<'_>,
    ld: &[i64],
) -> Result<Plan<T>> {
    let wrap = |source| Error::Contract { step, source };
    let problem = Problem::from_labels(
        T::STORAGE,
        a.spec(),
        b.spec(),
        CSpec::Absent,
        d.spec(),
        &Labels::new(la, lb, ld),
    )
    .map_err(wrap)?;
    Plan::<T>::new(&problem, &PlanConfig::default()).map_err(wrap)
}
