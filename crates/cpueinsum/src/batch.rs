//! How a serial loop of per-item vendor calls resolves a batch policy.

use tcontract_exec::batch::BatchStrategy;

/// Which route serves a *serial loop of per-item vendor calls* under a caller's
/// batch policy.
///
/// Several routes in this stack run a batch as a loop of one library call per
/// item on the calling thread: the prepared binary adapter
/// ([`BinaryPlan`](crate::BinaryPlan)), the grouped adapter
/// ([`GroupedPlan`](crate::GroupedPlan)), and the CBLAS
/// [`StepBackend`](crate::StepBackend) of `cpueinsum-blas`. Such a loop has no
/// outer lanes, no item threads and no single whole-batch call, so all three
/// resolve the caller's [`BatchPolicy`](crate::BatchPolicy) with this one rule
/// instead of each deciding for itself:
///
/// * [`Auto`](BatchStrategy::Auto) and [`Sequential`](BatchStrategy::Sequential)
///   are served by the loop: `Auto` keeps the loop's historical decision, and
///   `Sequential` is exactly one serial coordinator.
/// * A forced lane strategy grants outer lanes or item threads the loop cannot
///   honour, so [`Native`](Self::Native) hands it to the route's prepared native
///   alternative, which resolves the same policy.
///
/// # Examples
///
/// ```
/// use cpueinsum::{BatchStrategy, VendorLoopRoute};
///
/// let auto = VendorLoopRoute::resolve(BatchStrategy::Auto);
/// assert_eq!(auto, VendorLoopRoute::Loop);
/// let outer = VendorLoopRoute::resolve(BatchStrategy::Outer);
/// assert_eq!(outer, VendorLoopRoute::Native);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VendorLoopRoute {
    /// The serial loop serves the policy.
    Loop,
    /// The forced policy belongs to the route's native alternative.
    Native,
}

impl VendorLoopRoute {
    /// Resolve `strategy` for a serial loop of per-item vendor calls.
    pub fn resolve(strategy: BatchStrategy) -> Self {
        match strategy {
            BatchStrategy::Auto | BatchStrategy::Sequential => Self::Loop,
            // INVARIANT: `BatchStrategy` is `non_exhaustive`; every other
            // variant, present or future, forces outer lanes or item threads
            // (`Outer`, `Inner`, `Split`), which a serial loop cannot honour.
            // The native alternative resolves the same policy and refuses a
            // future route it cannot serve either.
            _ => Self::Native,
        }
    }
}
