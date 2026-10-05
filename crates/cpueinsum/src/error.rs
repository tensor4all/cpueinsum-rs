//! The error type.

/// A cpueinsum error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The einsum description itself is invalid (labels or contraction order).
    #[error(transparent)]
    Spec(#[from] SpecError),
    /// The operands do not fit the description or the plan.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// A binary contraction was rejected or failed in tprims-contract.
    #[error("contraction step {step}: {source}")]
    Contract {
        /// The step of the contraction order (0 for a binary contraction).
        step: usize,
        /// The tprims-contract error.
        source: tprims_contract::Error,
    },
    /// A [`StepBackend`](crate::StepBackend) failed a step it took.
    #[error("contraction step {step} (backend): {source}")]
    Backend {
        /// The step of the contraction order.
        step: usize,
        /// The backend's error.
        source: crate::BackendError,
    },
}

/// An invalid einsum description.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SpecError {
    /// No input operand.
    #[error("an einsum needs at least one input")]
    NoInputs,
    /// The contraction order does not have `inputs - 1` steps.
    #[error(
        "the contraction order has {got} steps, {expected} expected (one fewer than the inputs)"
    )]
    PathLength {
        /// `inputs - 1`.
        expected: usize,
        /// Steps given.
        got: usize,
    },
    /// A step names an operand that does not exist yet.
    #[error(
        "step {step} names operand {operand}, but only operands 0..{available} exist at that step"
    )]
    PathOperand {
        /// The step.
        step: usize,
        /// The operand id.
        operand: usize,
        /// Number of operands that exist before the step (inputs plus earlier results).
        available: usize,
    },
    /// A step contracts an operand with itself.
    #[error("step {step} contracts operand {operand} with itself")]
    PathSelf {
        /// The step.
        step: usize,
        /// The operand id.
        operand: usize,
    },
    /// An operand is consumed by two steps.
    #[error("operand {operand} is consumed twice (steps {first} and {second})")]
    PathReuse {
        /// The operand id.
        operand: usize,
        /// The first consuming step.
        first: usize,
        /// The second consuming step.
        second: usize,
    },
    /// An output label is repeated.
    #[error("output label {label} is repeated")]
    OutputRepeated {
        /// The label.
        label: i64,
    },
    /// An output label appears in no input.
    #[error("output label {label} appears in no input")]
    OutputOnly {
        /// The label.
        label: i64,
    },
}

/// Operands that do not fit the description or the plan.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ShapeError {
    /// The number of input layouts or views differs from the description.
    #[error("{got} inputs given, the description has {expected}")]
    InputCount {
        /// Inputs in the description.
        expected: usize,
        /// Inputs given.
        got: usize,
    },
    /// An operand's rank differs from its number of labels.
    #[error("operand {operand} has rank {rank} but {labels} labels")]
    Rank {
        /// Input index, or `inputs` for the output.
        operand: usize,
        /// Rank of the layout.
        rank: usize,
        /// Number of labels.
        labels: usize,
    },
    /// A label has two different extents.
    #[error("label {label} has extents {first} and {second}")]
    Extent {
        /// The label.
        label: i64,
        /// One extent.
        first: usize,
        /// Another extent.
        second: usize,
    },
    /// The dims and strides of a layout differ in length.
    #[error("a layout has {dims} extents and {strides} strides")]
    LayoutRank {
        /// Number of extents.
        dims: usize,
        /// Number of strides.
        strides: usize,
    },
    /// A view's dims or strides differ from the layout the plan was built for.
    #[error("operand {operand} differs from the layout the plan was built for")]
    Mismatch {
        /// Input index, or `inputs` for the output.
        operand: usize,
    },
    /// An intermediate's element count overflows `usize`.
    #[error("an intermediate's element count overflows usize")]
    Overflow,
}

/// `Result<T, cpueinsum::Error>`.
pub type Result<T> = core::result::Result<T, Error>;
