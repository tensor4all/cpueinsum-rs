//! Prepared heterogeneous column-major GEMM groups.
//!
//! A [`GroupedPlan`] owns one immutable tprims-contract plan per job.  The
//! caller lends flat input and output buffers at execution time; no pointers
//! or executors are retained by the plan.

use std::sync::Mutex;

use tprims_contract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tprims_contract::{OutputContract, Plan, PlanConfig, SliceAccumulationSource, NS_PER_FLOP};
use tprims_exec::{Exec, WidthPolicy};

use crate::{Error, Result, Scalar};

/// One compact column-major matrix multiplication in a grouped execution.
///
/// The three offsets and three dimensions are measured in elements.  A job
/// reads `rows * contracted` elements from `lhs`, `contracted * cols` from
/// `rhs`, and writes `rows * cols` elements to `out`.
///
/// # Examples
///
/// ```
/// use cpueinsum::GroupedGemmJob;
/// let job = GroupedGemmJob::new(0, 4, 8, 2, 3, 4);
/// assert_eq!(job.rows, 2);
/// assert_eq!(job.cols, 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedGemmJob {
    /// Element offset of the left matrix.
    pub lhs_offset: usize,
    /// Element offset of the right matrix.
    pub rhs_offset: usize,
    /// Element offset of the output matrix.
    pub out_offset: usize,
    /// Number of rows in this job.
    pub rows: usize,
    /// Contracted dimension in this job.
    pub contracted: usize,
    /// Number of columns in this job.
    pub cols: usize,
}

impl GroupedGemmJob {
    /// Construct a grouped GEMM descriptor.
    pub const fn new(
        lhs_offset: usize,
        rhs_offset: usize,
        out_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Self {
        Self {
            lhs_offset,
            rhs_offset,
            out_offset,
            rows,
            contracted,
            cols,
        }
    }
}

/// Validation failures owned by a grouped plan.
///
/// The `job` fields are the caller's original descriptor indices, even though
/// nonempty outputs are sorted for safe mutable splitting.
///
/// # Examples
///
/// ```
/// use cpueinsum::{Error, GroupedError, GroupedGemmJob, GroupedPlan};
/// let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1); 2];
/// let error = GroupedPlan::<f64>::new(&jobs, &Default::default()).unwrap_err();
/// assert!(matches!(error, Error::Grouped(GroupedError::Overlap { .. })));
/// ```
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GroupedError {
    /// Checked planning arithmetic overflowed.
    #[error("grouped job {job} overflowed while computing {what}")]
    Overflow {
        /// Original job index.
        job: usize,
        /// The checked quantity which overflowed.
        what: &'static str,
    },
    /// A runtime buffer is too short for a declared job range.
    #[error("grouped job {job} {operand} range ends at {end}, buffer has {available} elements")]
    Bounds {
        /// Original job index.
        job: usize,
        /// Buffer whose range was invalid.
        operand: GroupedOperand,
        /// Exclusive end of the requested range.
        end: usize,
        /// Available buffer length.
        available: usize,
    },
    /// Two nonempty output ranges overlap.
    #[error("grouped output jobs {first} and {second} overlap")]
    Overlap {
        /// First job in prepared output order.
        first: usize,
        /// Second job in prepared output order.
        second: usize,
    },
}

/// A grouped buffer named by a [`GroupedError::Bounds`] failure.
///
/// # Examples
///
/// ```
/// use cpueinsum::GroupedOperand;
/// assert_eq!(GroupedOperand::Output.to_string(), "output");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedOperand {
    /// The shared left-input buffer.
    Lhs,
    /// The shared right-input buffer.
    Rhs,
    /// The shared output buffer.
    Output,
}

impl std::fmt::Display for GroupedOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Lhs => "lhs",
            Self::Rhs => "rhs",
            Self::Output => "output",
        })
    }
}

/// The scheduler route selected for a grouped execution.
///
/// `Outer` is the only parallel route: each child contraction is forced to a
/// serial tprims context and receives the outer context's reusable workspace.
///
/// # Examples
///
/// See [`GroupedPlan::execute_into`] for a runnable route observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedRoute {
    /// There were no jobs.
    Empty,
    /// All jobs ran on the calling thread.
    Serial,
    /// Jobs were partitioned among this many outer lanes.
    Outer {
        /// Number of outer lanes.
        lanes: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct JobMeta {
    id: usize,
    lhs_len: usize,
    rhs_len: usize,
    out_len: usize,
    lhs_offset: usize,
    rhs_offset: usize,
    out_offset: usize,
}

#[derive(Debug)]
struct PreparedJob<T: Scalar> {
    meta: JobMeta,
    plan: Plan<T>,
}

/// Immutable plans and aggregate metadata for heterogeneous GEMM jobs.
///
/// # Examples
///
/// Shared inputs with reusable serial workspace, overwrite and accumulation:
///
/// ```
/// use cpueinsum::{ArenaProvider, Exec, GroupedGemmJob, GroupedPlan};
/// let jobs = [
///     GroupedGemmJob::new(0, 0, 0, 2, 2, 1),
///     GroupedGemmJob::new(0, 0, 2, 1, 1, 1),
/// ];
/// let plan = GroupedPlan::<f64>::new(&jobs, &Default::default())?;
/// let arena = ArenaProvider::new();
/// let exec = Exec::serial_with_workspace(&arena);
/// let (a, b) = ([1., 2., 3., 4.], [1., 1.]);
/// let mut d = [f64::NAN; 3];
/// plan.execute_into(&exec, 1.0, &a, &b, &mut d)?;
/// plan.execute_into_accum(&exec, 2.0, &a, &b, 0.5, &mut d)?;
/// assert_eq!(d, [10.0, 15.0, 2.5]);
/// # Ok::<(), cpueinsum::Error>(())
/// ```
#[derive(Debug)]
pub struct GroupedPlan<T: Scalar> {
    jobs: Vec<PreparedJob<T>>,
    required_lhs: usize,
    required_rhs: usize,
    required_out: usize,
    work: u128,
}

impl<T: Scalar> GroupedPlan<T> {
    /// Prepare unconjugated compact column-major jobs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] for checked arithmetic or overlapping output
    /// ranges, and [`Error::Contract`] when tprims rejects a child problem.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{GroupedGemmJob, GroupedPlan};
    /// use cpueinsum::tprims_contract::PlanConfig;
    /// let plan = GroupedPlan::<f64>::new(
    ///     &[GroupedGemmJob::new(0, 0, 0, 1, 1, 1)],
    ///     &PlanConfig::default(),
    /// )?;
    /// assert_eq!(plan.required_lhs_len(), 1);
    /// assert_eq!(plan.required_rhs_len(), 1);
    /// assert_eq!(plan.required_out_len(), 1);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    pub fn new(jobs: &[GroupedGemmJob], config: &PlanConfig) -> Result<Self> {
        Self::new_with_ops(jobs, [Op::Identity; 3], config)
    }

    /// Prepare jobs with common A, B and D operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan};
    /// use cpueinsum::tprims_contract::api::Op;
    /// use num_complex::Complex64 as C;
    /// let plan = GroupedPlan::<C>::new_with_ops(
    ///     &[GroupedGemmJob::new(0, 0, 0, 1, 1, 1)],
    ///     [Op::Conjugate, Op::Identity, Op::Conjugate], &Default::default(),
    /// )?;
    /// let mut d = [C::new(3.0, 1.0)];
    /// plan.execute_into_accum(&Exec::serial(), C::i(), &[C::new(1.0, 2.0)],
    ///     &[C::new(2.0, 0.0)], C::i(), &mut d)?;
    /// assert_eq!(d, [C::new(5.0, -5.0)]);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// The third operation is also used for the in-place accumulation source,
    /// matching tprims-contract's `CSpec::Output` semantics. The config's
    /// `fresh_output` capability is unused: grouped output is initialized.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] for checked arithmetic or overlapping output
    /// ranges, and [`Error::Contract`] when tprims rejects a child problem.
    pub fn new_with_ops(
        jobs: &[GroupedGemmJob],
        ops: [Op; 3],
        config: &PlanConfig,
    ) -> Result<Self> {
        let mut prepared = Vec::with_capacity(jobs.len());
        let mut required_lhs = 0;
        let mut required_rhs = 0;
        let mut required_out = 0;
        let mut work = 0u128;

        for (id, &job) in jobs.iter().enumerate() {
            let lhs_len = checked_product(id, job.rows, job.contracted, "lhs size")?;
            let rhs_len = checked_product(id, job.contracted, job.cols, "rhs size")?;
            let out_len = checked_product(id, job.rows, job.cols, "output size")?;
            let lhs_end = checked_add(id, job.lhs_offset, lhs_len, "lhs offset")?;
            let rhs_end = checked_add(id, job.rhs_offset, rhs_len, "rhs offset")?;
            let out_end = checked_add(id, job.out_offset, out_len, "output offset")?;
            for (end, what) in [
                (lhs_end, "lhs byte span"),
                (rhs_end, "rhs byte span"),
                (out_end, "output byte span"),
            ] {
                let bytes = checked_product(id, end, core::mem::size_of::<T>(), what)?;
                isize_dim(id, bytes, what)?;
            }
            required_lhs = required_lhs.max(lhs_end);
            required_rhs = required_rhs.max(rhs_end);
            required_out = required_out.max(out_end);
            let job_work = (job.rows as u128)
                .checked_mul(job.contracted as u128)
                .and_then(|v| v.checked_mul(job.cols as u128))
                .ok_or_else(|| {
                    grouped(GroupedError::Overflow {
                        job: id,
                        what: "work estimate",
                    })
                })?;
            work = work.checked_add(job_work).ok_or_else(|| {
                grouped(GroupedError::Overflow {
                    job: id,
                    what: "total work",
                })
            })?;

            isize_dim(id, job.rows, "rows")?;
            isize_dim(id, job.contracted, "contracted")?;
            isize_dim(id, job.cols, "cols")?;
            prepared.push(JobMeta {
                id,
                lhs_len,
                rhs_len,
                out_len,
                lhs_offset: job.lhs_offset,
                rhs_offset: job.rhs_offset,
                out_offset: job.out_offset,
            });
        }

        // Keep output order deterministic and retain original IDs. Empty jobs
        // stay after nonempty jobs and receive independent empty slices below.
        prepared.sort_unstable_by_key(|meta| (meta.out_len == 0, meta.out_offset, meta.id));
        for pair in prepared.windows(2) {
            if pair[1].out_len != 0 && pair[0].out_offset + pair[0].out_len > pair[1].out_offset {
                return Err(grouped(GroupedError::Overlap {
                    first: pair[0].id,
                    second: pair[1].id,
                }));
            }
        }
        // Validate every descriptor and overlap before preparing numerical plans.
        let mut child_config = config.clone();
        child_config.fresh_output = false;
        let prepared = prepared
            .into_iter()
            .map(|meta| {
                let job = jobs[meta.id];
                // Dimension-to-stride conversions were checked above for every job.
                let rows = (job.rows as isize).max(1);
                let contracted = (job.contracted as isize).max(1);
                let problem = Problem::from_labels(
                    T::STORAGE,
                    spec(&[job.rows, job.contracted], &[1, rows], ops[0]),
                    spec(&[job.contracted, job.cols], &[1, contracted], ops[1]),
                    CSpec::Output(ops[2]),
                    spec(&[job.rows, job.cols], &[1, rows], ops[2]),
                    &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
                )
                .map_err(|source| Error::Contract {
                    step: meta.id,
                    source,
                })?;
                let plan =
                    Plan::new(&problem, &child_config).map_err(|source| Error::Contract {
                        step: meta.id,
                        source,
                    })?;
                Ok(PreparedJob { meta, plan })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            jobs: prepared,
            required_lhs,
            required_rhs,
            required_out,
            work,
        })
    }

    /// Minimum length of the shared left-input buffer.
    /// See [`Self::new`] for a runnable example.
    pub fn required_lhs_len(&self) -> usize {
        self.required_lhs
    }

    /// Minimum length of the shared right-input buffer.
    /// See [`Self::new`] for a runnable example.
    pub fn required_rhs_len(&self) -> usize {
        self.required_rhs
    }

    /// Minimum length of the shared output buffer.
    /// See [`Self::new`] for a runnable example.
    pub fn required_out_len(&self) -> usize {
        self.required_out
    }

    /// Overwrite all jobs with `alpha * A B`, reading no previous output.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan, GroupedRoute};
    /// let plan = GroupedPlan::<f64>::new(
    ///     &[GroupedGemmJob::new(0, 0, 0, 2, 2, 1)], &Default::default(),
    /// )?;
    /// let mut d = [f64::NAN; 2];
    /// let route = plan.execute_into(&Exec::serial(), 2.0, &[1., 2., 3., 4.],
    ///     &[1., 1.], &mut d)?;
    /// assert_eq!(d, [8.0, 12.0]);
    /// assert_eq!(route, GroupedRoute::Serial);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] when any declared range is outside a shared
    /// buffer, and [`Error::Contract`] for a child execution failure. All
    /// bounds and route checks happen before the first write.
    pub fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        out: &mut [T],
    ) -> Result<GroupedRoute> {
        self.execute(exec, alpha, lhs, rhs, T::default(), false, out)
    }

    /// Accumulate `alpha * A B + beta * D` in every job's output range.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan};
    /// let plan = GroupedPlan::<f64>::new(
    ///     &[GroupedGemmJob::new(0, 0, 0, 1, 1, 1)], &Default::default(),
    /// )?;
    /// let mut d = [8.0];
    /// plan.execute_into_accum(&Exec::serial(), 2.0, &[3.0], &[4.0], 0.5, &mut d)?;
    /// assert_eq!(d, [28.0]);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] when any declared range is outside a shared
    /// buffer, and [`Error::Contract`] for a child execution failure. All
    /// bounds and route checks happen before the first write.
    pub fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        out: &mut [T],
    ) -> Result<GroupedRoute> {
        self.execute(exec, alpha, lhs, rhs, beta, true, out)
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
    ) -> Result<GroupedRoute> {
        for job in &self.jobs {
            check_bounds(
                job.meta.id,
                GroupedOperand::Lhs,
                job.meta.lhs_offset,
                job.meta.lhs_len,
                lhs.len(),
            )?;
            check_bounds(
                job.meta.id,
                GroupedOperand::Rhs,
                job.meta.rhs_offset,
                job.meta.rhs_len,
                rhs.len(),
            )?;
            check_bounds(
                job.meta.id,
                GroupedOperand::Output,
                job.meta.out_offset,
                job.meta.out_len,
                out.len(),
            )?;
        }
        let child_exec = exec
            .workspace()
            .map_or(Exec::Serial, Exec::serial_with_workspace);
        let source = if accumulate && beta != T::default() {
            SliceAccumulationSource::Output
        } else {
            SliceAccumulationSource::Absent
        };
        for job in &self.jobs {
            job.plan
                .execution_route(&child_exec, alpha, OutputContract::Initialized)
                .map_err(|source| Error::Contract {
                    step: job.meta.id,
                    source,
                })?;
        }

        if self.jobs.is_empty() {
            return Ok(GroupedRoute::Empty);
        }
        let serial_ns =
            2.0 * self.work as f64 * NS_PER_FLOP * if T::IS_COMPLEX { 4.0 } else { 1.0 };
        let lanes = exec
            .width_for(serial_ns, &WidthPolicy::default())
            .min(exec.budget())
            .min(self.jobs.len())
            .max(1);
        let route = if lanes == 1 {
            GroupedRoute::Serial
        } else {
            GroupedRoute::Outer { lanes }
        };

        let mut items = Vec::with_capacity(self.jobs.len());
        let mut cursor = 0usize;
        let mut remaining = out;
        for job in &self.jobs {
            let d = if job.meta.out_len == 0 {
                &mut [][..]
            } else {
                let start = job.meta.out_offset;
                let end = start + job.meta.out_len;
                let current = core::mem::take(&mut remaining);
                let (_, rest) = current.split_at_mut(start - cursor);
                let (slice, tail) = rest.split_at_mut(end - start);
                remaining = tail;
                cursor = end;
                slice
            };
            items.push(JobItem {
                id: job.meta.id,
                plan: &job.plan,
                lhs: &lhs[job.meta.lhs_offset..job.meta.lhs_offset + job.meta.lhs_len],
                rhs: &rhs[job.meta.rhs_offset..job.meta.rhs_offset + job.meta.rhs_len],
                out: d,
            });
        }

        if lanes == 1 {
            for item in &mut items {
                run_item(item, &child_exec, alpha, beta, source)?;
            }
            return Ok(route);
        }

        let item_count = items.len();
        let mut remaining_items = items.as_mut_slice();
        let mut lane_items = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let count = item_count / lanes + usize::from(lane < item_count % lanes);
            let current = core::mem::take(&mut remaining_items);
            let (part, tail) = current.split_at_mut(count);
            remaining_items = tail;
            lane_items.push(Mutex::new(part));
        }
        let failure = Mutex::new(None);
        exec.for_each_partition(lanes, &|lane| {
            let mut jobs = lane_items[lane]
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for item in jobs.iter_mut() {
                if let Err(error) = run_item(item, &child_exec, alpha, beta, source) {
                    let mut first = failure.lock().unwrap_or_else(|poison| poison.into_inner());
                    if first.is_none() {
                        *first = Some(error);
                    }
                }
            }
        });
        failure
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner())
            .map_or(Ok(route), Err)
    }
}

struct JobItem<'a, T: Scalar> {
    id: usize,
    plan: &'a Plan<T>,
    lhs: &'a [T],
    rhs: &'a [T],
    out: &'a mut [T],
}

fn run_item<T: Scalar>(
    item: &mut JobItem<'_, T>,
    exec: &Exec<'_>,
    alpha: T,
    beta: T,
    source: SliceAccumulationSource<'_, T>,
) -> Result<()> {
    item.plan
        .execute_slices_accum(
            exec,
            alpha,
            (item.lhs, 0),
            (item.rhs, 0),
            beta,
            source,
            (item.out, 0),
        )
        .map_err(|source| Error::Contract {
            step: item.id,
            source,
        })
}

fn spec(dims: &[usize], strides: &[isize], op: Op) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).expect("validated dimensions")).with_op(op)
}

fn grouped(source: GroupedError) -> Error {
    Error::Grouped(source)
}

fn checked_product(job: usize, a: usize, b: usize, what: &'static str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| grouped(GroupedError::Overflow { job, what }))
}

fn checked_add(job: usize, a: usize, b: usize, what: &'static str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| grouped(GroupedError::Overflow { job, what }))
}

fn isize_dim(job: usize, value: usize, what: &'static str) -> Result<isize> {
    isize::try_from(value).map_err(|_| grouped(GroupedError::Overflow { job, what }))
}

fn check_bounds(
    job: usize,
    operand: GroupedOperand,
    offset: usize,
    len: usize,
    available: usize,
) -> Result<()> {
    let end = offset.checked_add(len).ok_or_else(|| {
        grouped(GroupedError::Overflow {
            job,
            what: "buffer range",
        })
    })?;
    if end > available {
        return Err(grouped(GroupedError::Bounds {
            job,
            operand,
            end,
            available,
        }));
    }
    Ok(())
}
