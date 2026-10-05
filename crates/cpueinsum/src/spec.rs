//! The einsum description: labels and the contraction order.

use crate::error::{Result, SpecError};

/// Integer labels of every input and of the output, and the order in which the
/// inputs are contracted.
///
/// The order is a list of pairs in **SSA form**: inputs are operands
/// `0..n`, and step `k` contracts two existing operands into a new operand
/// `n + k`. Every operand except the last result is consumed exactly once, so
/// there are `n - 1` steps (none for a single input). cpueinsum does not
/// search for an order; the caller supplies it.
///
/// A label repeated within one input selects a diagonal; a label that appears
/// in one input only and not in the output is summed. Output labels must be
/// distinct and must each appear in some input.
///
/// An `opt_einsum`-style positional path (pairs of positions in a shrinking
/// operand list, the result appended at the end) converts to this form by
/// replaying the list once.
///
/// # Examples
///
/// ```
/// use cpueinsum::EinsumSpec;
/// // D[i, l] = sum_{j, k} A[i, j] B[j, k] C[k, l], contracting (A B) first.
/// let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [3, 2]]).unwrap();
/// assert_eq!(spec.inputs().len(), 3);
/// assert_eq!(spec.path(), &[[0, 1], [3, 2]]);
/// // Operand 3 does not exist yet at step 0.
/// assert!(EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 3], [1, 2]]).is_err());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumSpec {
    inputs: Vec<Vec<i64>>,
    output: Vec<i64>,
    path: Vec<[usize; 2]>,
}

impl EinsumSpec {
    /// A validated description.
    ///
    /// # Errors
    ///
    /// [`SpecError`] when there is no input, when the order does not have
    /// `inputs - 1` steps, names a missing operand, pairs an operand with
    /// itself or consumes one twice, or when an output label is repeated or
    /// appears in no input.
    pub fn new(inputs: &[&[i64]], output: &[i64], path: &[[usize; 2]]) -> Result<Self> {
        let n = inputs.len();
        if n == 0 {
            return Err(SpecError::NoInputs.into());
        }
        if path.len() != n - 1 {
            return Err(SpecError::PathLength {
                expected: n - 1,
                got: path.len(),
            }
            .into());
        }
        // consumed_by[id] = the step that consumes operand `id`.
        let mut consumed_by = vec![usize::MAX; 2 * n - 1];
        for (step, &[x, y]) in path.iter().enumerate() {
            let available = n + step;
            for id in [x, y] {
                if id >= available {
                    return Err(SpecError::PathOperand {
                        step,
                        operand: id,
                        available,
                    }
                    .into());
                }
            }
            if x == y {
                return Err(SpecError::PathSelf { step, operand: x }.into());
            }
            for id in [x, y] {
                if consumed_by[id] != usize::MAX {
                    return Err(SpecError::PathReuse {
                        operand: id,
                        first: consumed_by[id],
                        second: step,
                    }
                    .into());
                }
                consumed_by[id] = step;
            }
        }
        // INVARIANT: the output rank is small (a tensor's rank), so the
        // quadratic scans below are bounded by rank^2 and rank * total labels.
        for (i, &l) in output.iter().enumerate() {
            if output[..i].contains(&l) {
                return Err(SpecError::OutputRepeated { label: l }.into());
            }
            if !inputs.iter().any(|ls| ls.contains(&l)) {
                return Err(SpecError::OutputOnly { label: l }.into());
            }
        }
        Ok(Self {
            inputs: inputs.iter().map(|l| l.to_vec()).collect(),
            output: output.to_vec(),
            path: path.to_vec(),
        })
    }

    /// Labels of each input.
    pub fn inputs(&self) -> &[Vec<i64>] {
        &self.inputs
    }

    /// Labels of the output.
    pub fn output(&self) -> &[i64] {
        &self.output
    }

    /// The contraction order in SSA form.
    pub fn path(&self) -> &[[usize; 2]] {
        &self.path
    }
}
