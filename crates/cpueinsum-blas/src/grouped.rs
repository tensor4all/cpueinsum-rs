//! Whole-group preparation and preflight; vendor jobs run on the coordinator.

use cpueinsum::tprims_contract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem};
use cpueinsum::tprims_contract::{PlanConfig, SliceAccumulationSource};
use cpueinsum::{Exec, GroupedGemmJob};

use crate::prepared::VendorPlan;
use crate::{BlasError, BlasScalar};

/// Observed execution of a concrete prepared BLAS group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedRoute {
    /// No jobs were submitted.
    Empty,
    /// This many GEMMs ran serially on the coordinator.
    Vendor {
        /// Number of coordinator GEMM calls.
        jobs: usize,
    },
    /// The whole group's prepared native alternative executed.
    Native(cpueinsum::GroupedRoute),
}

#[derive(Debug)]
struct VendorJob {
    id: usize,
    lhs_offset: isize,
    rhs_offset: isize,
    out_offset: isize,
    plan: VendorPlan,
}

/// Prepared grouped GEMMs over shared, initialized caller buffers.
///
/// Every descriptor is validated during preparation. If any child cannot use
/// direct vendor output, the whole group selects a prepared native alternative.
/// Fresh grouped backing storage is deliberately not supported.
///
/// # Examples
///
/// ```
/// # extern crate blas_src as _;
/// use cpueinsum::{Exec, GroupedGemmJob, tprims_contract::PlanConfig};
/// use cpueinsum_blas::{GroupedPlan, GroupedRoute};
/// let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
/// let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default())?;
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 1.0];
/// let mut out = [0.0; 4];
/// let mut work = vec![0.0; plan.work_len()];
/// assert_eq!(plan.execute_into(&Exec::serial(), 1.0, &a, &b, &mut out, &mut work)?,
///     GroupedRoute::Vendor { jobs: 1 });
/// assert_eq!(out, a);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct GroupedPlan<T: BlasScalar> {
    native: cpueinsum::GroupedPlan<T>,
    vendor: Option<Vec<VendorJob>>,
    work_len: usize,
}

impl<T: BlasScalar> GroupedPlan<T> {
    /// Prepare unconjugated grouped GEMMs.
    ///
    /// # Errors
    ///
    /// As [`Self::new_with_ops`].
    pub fn new(jobs: &[GroupedGemmJob], config: &PlanConfig) -> Result<Self, BlasError> {
        Self::new_with_ops(jobs, [Op::Identity; 3], config)
    }

    /// Prepare a group with common operand/output operations.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Prepare`] for checked geometry/byte overflow,
    /// overlapping outputs, invalid configuration or a rejected child problem.
    pub fn new_with_ops(
        jobs: &[GroupedGemmJob],
        ops: [Op; 3],
        config: &PlanConfig,
    ) -> Result<Self, BlasError> {
        let native = cpueinsum::GroupedPlan::<T>::new_with_ops(jobs, ops, config)
            .map_err(|source| BlasError::Prepare { source })?;
        let mut vendor = Vec::with_capacity(jobs.len());
        for (id, job) in jobs.iter().enumerate() {
            // INVARIANT: native preparation above checked every product, byte
            // end, dimension conversion and output overlap before this loop.
            let rows = (job.rows as isize).max(1);
            let contracted = (job.contracted as isize).max(1);
            let operand = |dims: &[usize], strides: &[isize], op: Op| {
                LayoutSpec::new(dims, strides, 0).map(|layout| OperandSpec::new(layout).with_op(op))
            };
            let problem = (|| {
                Problem::from_labels(
                    T::STORAGE,
                    operand(&[job.rows, job.contracted], &[1, rows], ops[0])?,
                    operand(&[job.contracted, job.cols], &[1, contracted], ops[1])?,
                    CSpec::Output(ops[2]),
                    operand(&[job.rows, job.cols], &[1, rows], ops[2])?,
                    &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
                )
            })()
            .map_err(|source| BlasError::Prepare {
                source: cpueinsum::Error::Contract { step: id, source },
            })?;
            let Some(plan) = VendorPlan::new::<T>(&problem, config) else {
                return Ok(Self {
                    native,
                    vendor: None,
                    work_len: 0,
                });
            };
            vendor.push(VendorJob {
                id,
                lhs_offset: job.lhs_offset as isize,
                rhs_offset: job.rhs_offset as isize,
                out_offset: job.out_offset as isize,
                plan,
            });
        }
        let work_len = vendor
            .iter()
            .map(|job| job.plan.work_len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            native,
            vendor: Some(vendor),
            work_len,
        })
    }

    /// Required caller packing storage in elements; it is reused across jobs.
    pub fn work_len(&self) -> usize {
        self.work_len
    }

    /// Overwrite each job, reading no old output values.
    ///
    /// # Errors
    ///
    /// As [`Self::execute_into_accum`].
    pub fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        out: &mut [T],
        work: &mut [T],
    ) -> Result<GroupedRoute, BlasError> {
        self.execute(exec, alpha, lhs, rhs, T::default(), false, out, work)
    }

    /// Accumulate into each initialized job output, preserving untouched gaps.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Native`] for a selected native failure or
    /// [`BlasError::Grouped`] retaining the original job ID and vendor bounds,
    /// workspace or packing-view error. All jobs are preflighted before writes.
    pub fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        out: &mut [T],
        work: &mut [T],
    ) -> Result<GroupedRoute, BlasError> {
        self.execute(exec, alpha, lhs, rhs, beta, true, out, work)
    }

    fn execute(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        accumulate: bool,
        out: &mut [T],
        work: &mut [T],
    ) -> Result<GroupedRoute, BlasError> {
        let Some(jobs) = self.vendor.as_ref().filter(|_| alpha != T::default()) else {
            let result = if accumulate {
                self.native
                    .execute_into_accum(exec, alpha, lhs, rhs, beta, out)
            } else {
                self.native.execute_into(exec, alpha, lhs, rhs, out)
            };
            return result
                .map(GroupedRoute::Native)
                .map_err(|source| BlasError::Native { source });
        };
        if jobs.is_empty() {
            return Ok(GroupedRoute::Empty);
        }
        let source = if accumulate {
            SliceAccumulationSource::Output
        } else {
            SliceAccumulationSource::Absent
        };
        for job in jobs {
            job.plan
                .preflight(
                    (lhs, job.lhs_offset),
                    (rhs, job.rhs_offset),
                    beta,
                    source,
                    out.len(),
                    job.out_offset,
                    work,
                )
                .map_err(|source| BlasError::Grouped {
                    job: job.id,
                    source: Box::new(source),
                })?;
        }
        // No native children or runtime capability decisions follow vendor writes.
        // All packing views were validated above; workspace is disjoint from the
        // shared inputs/output by the exclusive Rust borrows held for this call.
        for job in jobs {
            job.plan
                .run_vendor(
                    alpha,
                    (lhs, job.lhs_offset),
                    (rhs, job.rhs_offset),
                    beta,
                    out.as_mut_ptr(),
                    job.out_offset,
                    work,
                )
                .map_err(|source| BlasError::Grouped {
                    job: job.id,
                    source: Box::new(source),
                })?;
        }
        Ok(GroupedRoute::Vendor { jobs: jobs.len() })
    }
}
