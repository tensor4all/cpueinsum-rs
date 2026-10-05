//! CPU einsum over integer labels with a caller-supplied contraction order.
//!
//! cpueinsum sits above [tprims-contract]: each binary step of an einsum is one
//! tprims-contract plan (packed TBLIS-style driver, faer on a copy-free GEMM
//! fusion, or an elementwise pass), and cpueinsum adds what an N-ary einsum
//! needs on top: the step order, the labels each intermediate keeps, and one
//! reusable scratch buffer for all intermediates.
//!
//! Deliberately out of scope: contraction-order search (the caller passes the
//! order), string notation (labels are `i64`), BLAS, and GPUs.
//!
//! [tprims-contract]: https://github.com/tensor4all/tprims-rs
//!
//! # Examples
//!
//! ```
//! use cpueinsum::strided_view::{StridedView, StridedViewMut};
//! use cpueinsum::{einsum_into, EinsumSpec, Exec};
//! // d[i] = sum_{j,k} a[i, j] b[j, k] c[k], contracting (b c) first.
//! let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2]], &[0], &[[1, 2], [0, 3]]).unwrap();
//! let a = [1.0, 2.0, 3.0, 4.0]; // column-major [[1, 3], [2, 4]]
//! let b = [1.0, 0.0, 0.0, 1.0];
//! let c = [1.0, 1.0];
//! let mut d = [0.0; 2];
//! let inputs = [
//!     StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
//!     StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap(),
//!     StridedView::new(&c, &[2], &[1], 0).unwrap(),
//! ];
//! let mut dv = StridedViewMut::new(&mut d, &[2], &[1], 0).unwrap();
//! einsum_into(&Exec::serial(), &spec, &inputs, &mut dv).unwrap();
//! assert_eq!(d, [4.0, 6.0]);
//! ```
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

mod binary;
mod error;
mod layout;
mod plan;
mod scalar;
mod spec;

pub use binary::contract_into;
pub use error::{Error, Result, ShapeError, SpecError};
pub use layout::Layout;
pub use plan::{einsum_into, EinsumPlan, Scratch};
pub use scalar::Scalar;
pub use spec::EinsumSpec;
pub use tprims_exec::{Exec, Pool};

pub use strided_view;
