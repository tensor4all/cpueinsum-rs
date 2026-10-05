//! Borrowed operand layouts.

use strided_view::{StridedView, StridedViewMut};

use crate::error::{Result, ShapeError};

/// Extents and signed element strides of one operand, borrowed.
///
/// A plan is built from layouts alone; the data arrives at execution as views
/// with the same extents and strides.
///
/// # Examples
///
/// ```
/// use cpueinsum::Layout;
/// let l = Layout::new(&[2, 3], &[1, 2]).unwrap();
/// assert_eq!(l.dims(), &[2, 3]);
/// assert!(Layout::new(&[2, 3], &[1]).is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout<'a> {
    dims: &'a [usize],
    strides: &'a [isize],
}

impl<'a> Layout<'a> {
    /// A layout from extents and strides.
    ///
    /// # Errors
    ///
    /// [`ShapeError::LayoutRank`] when the lengths differ.
    pub fn new(dims: &'a [usize], strides: &'a [isize]) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(ShapeError::LayoutRank {
                dims: dims.len(),
                strides: strides.len(),
            }
            .into());
        }
        Ok(Self { dims, strides })
    }

    /// The layout of a rank-0 operand.
    pub const fn rank0() -> Self {
        Self {
            dims: &[],
            strides: &[],
        }
    }

    /// The layout of a view.
    pub fn of<T>(view: &'a StridedView<'_, T>) -> Self {
        Self {
            dims: view.dims(),
            strides: view.strides(),
        }
    }

    /// The layout of a mutable view.
    pub fn of_mut<T>(view: &'a StridedViewMut<'_, T>) -> Self {
        Self {
            dims: view.dims(),
            strides: view.strides(),
        }
    }

    /// Extent of each axis.
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    /// Signed element stride of each axis.
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    pub(crate) fn spec(&self) -> tprims_contract::api::OperandSpec {
        // INVARIANT: `new` and the view constructors guarantee equal lengths,
        // the only failure of `LayoutSpec::new`.
        let l = tprims_contract::api::LayoutSpec::new(self.dims, self.strides, 0)
            .expect("layout lengths agree by construction");
        tprims_contract::api::OperandSpec::new(l)
    }
}
