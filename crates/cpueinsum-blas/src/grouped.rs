//! Whole-group preparation and preflight; vendor jobs run on the coordinator.

use core::mem::MaybeUninit;

use cpueinsum::tcontract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem};
use cpueinsum::tcontract::{
    execute_uninit_with, PlanConfig, SliceAccumulationSource, UninitOutputError,
};
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
    out_len: usize,
    plan: VendorPlan,
}

/// Prepared grouped GEMMs over shared caller buffers.
///
/// Every descriptor is validated during preparation. If any child cannot use
/// direct vendor output, the whole group selects a prepared native alternative.
/// [`Self::execute_uninit`] accepts fresh storage with zero beta and exact coverage.
///
/// # Examples
///
/// ```
/// # extern crate blas_src as _;
/// use cpueinsum::{Exec, GroupedGemmJob, tcontract::PlanConfig};
/// use cpueinsum_blas::{GroupedPlan, GroupedRoute};
/// let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
/// let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default())?;
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 1.0];
/// let mut out = [0.0; 4];
/// let mut work = vec![0.0; plan.work_len()];
/// assert_eq!(plan.execute_into(&mut Exec::serial(), 1.0, &a, &b, &mut out, &mut work)?,
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
    /// Returns [`BlasError::Prepare`] for overflowing geometry, overlapping
    /// outputs, invalid configuration or a rejected child problem.
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
                out_len: job.rows * job.cols,
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
    /// Returns [`BlasError::Native`] for native admission/execution failures or
    /// [`BlasError::Grouped`] for a vendor job's bounds, workspace, packing or
    /// execution failure. Every job is preflighted before writes.
    pub fn execute_into(
        &self,
        exec: &mut Exec<'_>,
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
    /// The executor's batch policy resolves with
    /// [`VendorLoopRoute`](cpueinsum::VendorLoopRoute): `Auto` and `Sequential`
    /// keep the vendor loop, a forced `Outer`/`Inner`/`Split` uses the native
    /// alternative.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Native`] for a selected native failure, or
    /// [`BlasError::Grouped`] retaining the original job ID and vendor bounds,
    /// workspace or packing-view error. All jobs are preflighted before writes.
    pub fn execute_into_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        out: &mut [T],
        work: &mut [T],
    ) -> Result<GroupedRoute, BlasError> {
        self.execute(exec, alpha, lhs, rhs, beta, true, out, work)
    }

    /// Overwrite fresh grouped storage and publish it only after full completion.
    ///
    /// The vendor loop keeps its prepared route; the native alternative runs
    /// when a job cannot use direct vendor output, when `alpha` is zero, or when
    /// the executor's batch policy forces lanes. `beta` must be zero and the
    /// group's jobs must cover `out` exactly once without gaps, both checked
    /// before any write.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Accumulation`] for a nonzero `beta`,
    /// [`BlasError::FreshCoverage`] for holes or padding, [`BlasError::Grouped`]
    /// retaining the original job ID for a vendor bounds, workspace, packing or
    /// execution failure, and [`BlasError::Native`] for a selected native
    /// failure. Every pre-write check runs before the first write, and no
    /// initialized slice escapes on failure.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate blas_src as _;
    /// use core::mem::MaybeUninit;
    /// use cpueinsum::{Exec, GroupedGemmJob, tcontract::PlanConfig};
    /// use cpueinsum_blas::{GroupedPlan, GroupedRoute};
    /// let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
    /// let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default())?;
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [1.0, 0.0, 0.0, 1.0];
    /// let mut work = vec![0.0; plan.work_len()];
    /// let mut out = [MaybeUninit::uninit(); 4];
    /// let (out, route) = plan.execute_uninit(&mut Exec::serial(), 1.0, &a, &b, 0.0,
    ///     &mut out, &mut work)?;
    /// assert_eq!(out, a);
    /// assert_eq!(route, GroupedRoute::Vendor { jobs: 1 });
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn execute_uninit<'d>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        out: &'d mut [MaybeUninit<T>],
        work: &mut [T],
    ) -> Result<(&'d mut [T], GroupedRoute), BlasError> {
        // The vendor jobs share the native descriptors, so the native checker
        // proves the same beta and exact-coverage conditions before any write.
        self.native
            .validate_uninit(beta, out.len())
            .map_err(validation)?;
        let vendor_route = crate::vendor_loop_route(exec);
        let Some(jobs) = self
            .vendor
            .as_ref()
            .filter(|_| alpha != T::default() && vendor_route)
        else {
            return self
                .native
                .execute_uninit(exec, alpha, lhs, rhs, beta, out)
                .map(|(out, route)| (out, GroupedRoute::Native(route)))
                .map_err(|source| BlasError::Native { source });
        };
        execute_uninit_with(out, |storage| {
            if jobs.is_empty() {
                return Ok((Vec::new(), GroupedRoute::Empty));
            }
            // Every vendor job is preflighted with no accumulation source before
            // the first write, in descriptor order so the error names the lowest
            // failing job id.
            for job in jobs {
                job.plan
                    .preflight(
                        (lhs, job.lhs_offset),
                        (rhs, job.rhs_offset),
                        beta,
                        SliceAccumulationSource::Absent,
                        storage.len(),
                        job.out_offset,
                        work,
                    )
                    .map_err(|source| BlasError::Grouped {
                        job: job.id,
                        source: Box::new(source),
                    })?;
            }
            // Fresh output is split by increasing output offset; the initialized
            // loop's descriptor order is left alone.
            let mut sorted: Vec<&VendorJob> = jobs.iter().collect();
            sorted.sort_unstable_by_key(|job| job.out_offset);
            let mut receipts = Vec::with_capacity(sorted.len());
            let mut remaining: &mut [MaybeUninit<T>] = storage;
            let mut cursor = 0usize;
            for job in sorted {
                // Exact coverage was proven above, so this job starts at the
                // cursor and its slice is exactly its own output range.
                let start = job.out_offset as usize;
                let current = core::mem::take(&mut remaining);
                let (_, rest) = current.split_at_mut(start - cursor);
                let (slice, tail) = rest.split_at_mut(job.out_len);
                remaining = tail;
                cursor = start + job.out_len;
                let initialized = job
                    .plan
                    .execute_uninit_slices(
                        alpha,
                        (lhs, job.lhs_offset),
                        (rhs, job.rhs_offset),
                        beta,
                        SliceAccumulationSource::Absent,
                        // The slice is exactly the job's output domain, so its
                        // own origin is zero; the whole-buffer origin above is
                        // the preflight's.
                        (slice, 0),
                        work,
                    )
                    .map_err(|source| BlasError::Grouped {
                        job: job.id,
                        source: Box::new(source),
                    })?;
                receipts.push(initialized);
            }
            Ok((receipts, GroupedRoute::Vendor { jobs: jobs.len() }))
        })
        .map_err(|source| match source {
            UninitOutputError::Execution(source) => source,
            UninitOutputError::Coverage => BlasError::FreshCoverage,
        })
    }

    fn execute(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        accumulate: bool,
        out: &mut [T],
        work: &mut [T],
    ) -> Result<GroupedRoute, BlasError> {
        // Resolve forced-strategy admission before every shortcut and before
        // any write, with the same rule the prepared binary adapter and the
        // N-ary BLAS backend use. A forced lane strategy that grants outer lanes
        // or item threads cannot be honored by the serial loop, so it goes to
        // the native alternative, which resolves the same policy. `Auto` keeps
        // today's preferred vendor loop and `Sequential` is exactly one serial
        // coordinator, so the loop serves both.
        let vendor_route = crate::vendor_loop_route(exec);
        let Some(jobs) = self
            .vendor
            .as_ref()
            .filter(|_| alpha != T::default() && vendor_route)
        else {
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

/// Map the shared native fresh-output check onto this crate's error names.
fn validation(source: cpueinsum::Error) -> BlasError {
    match source {
        cpueinsum::Error::Grouped(cpueinsum::GroupedError::FreshCoverage) => {
            BlasError::FreshCoverage
        }
        cpueinsum::Error::Grouped(cpueinsum::GroupedError::NonzeroBeta) => BlasError::Accumulation,
        source => BlasError::Native { source },
    }
}
