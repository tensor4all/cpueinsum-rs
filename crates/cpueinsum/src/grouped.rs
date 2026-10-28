//! Prepared heterogeneous column-major GEMM groups.
//!
//! A [`GroupedPlan`] owns one immutable tcontract plan per job.  The
//! caller lends flat input and output buffers at execution time; no pointers
//! or executors are retained by the plan.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use tcontract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::ns_per_flop;
use tcontract::{OutputContract, Plan, PlanConfig, SliceAccumulationSource};
use tcontract_exec::{BatchPolicy, BatchQuery, BatchStrategy, Exec, WorkspaceReq};

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

/// Dispatches the lanes of a grouped execution perform, declared to the token's width
/// rule: one fork over the jobs.
const GROUPED_SECTIONS: u32 = 1;

/// The scheduler route selected for a grouped execution.
///
/// The route is resolved from the caller's [`Exec`] batch policy before any
/// write: `Auto` takes the token's `width_for` decision, and a forced strategy
/// is honored or refused up front. [`GroupedRoute::Outer`], `Inner` and `Split`
/// report how many outer lanes and how many threads per item actually ran.
///
/// # Examples
///
/// See [`GroupedPlan::execute_into`] for a runnable route observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupedRoute {
    /// There were no jobs.
    Empty,
    /// All jobs ran on the calling thread, one thread per item.
    Serial,
    /// Jobs were partitioned among this many outer lanes, each item serial.
    Outer {
        /// Number of outer lanes.
        lanes: usize,
    },
    /// One outer lane, with `threads` threads granted to each item.
    Inner {
        /// Threads granted to each item.
        threads: usize,
    },
    /// Jobs were partitioned among `lanes`, each item on `item_threads`.
    Split {
        /// Number of outer lanes.
        lanes: usize,
        /// Threads granted to each item.
        item_threads: usize,
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
/// let mut exec = Exec::serial_with_workspace(&arena);
/// let (a, b) = ([1., 2., 3., 4.], [1., 1.]);
/// let mut d = [f64::NAN; 3];
/// plan.execute_into(&mut exec, 1.0, &a, &b, &mut d)?;
/// plan.execute_into_accum(&mut exec, 2.0, &a, &b, 0.5, &mut d)?;
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
    max_job_work: u128,
}

impl<T: Scalar> GroupedPlan<T> {
    /// Prepare unconjugated compact column-major jobs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] for checked arithmetic or overlapping output
    /// ranges, and [`Error::Contract`] when tcontract rejects a child problem.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{GroupedGemmJob, GroupedPlan};
    /// use cpueinsum::tcontract::PlanConfig;
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
    /// use cpueinsum::tcontract::api::Op;
    /// use num_complex::Complex64 as C;
    /// let plan = GroupedPlan::<C>::new_with_ops(
    ///     &[GroupedGemmJob::new(0, 0, 0, 1, 1, 1)],
    ///     [Op::Conjugate, Op::Identity, Op::Conjugate], &Default::default(),
    /// )?;
    /// let mut d = [C::new(3.0, 1.0)];
    /// plan.execute_into_accum(&mut Exec::serial(), C::i(), &[C::new(1.0, 2.0)],
    ///     &[C::new(2.0, 0.0)], C::i(), &mut d)?;
    /// assert_eq!(d, [C::new(5.0, -5.0)]);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// The third operation is also used for the in-place accumulation source,
    /// matching tcontract's `CSpec::Output` semantics. The config's
    /// `fresh_output` capability is unused: grouped output is initialized.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] for checked arithmetic or overlapping output
    /// ranges, and [`Error::Contract`] when tcontract rejects a child problem.
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
        let mut max_job_work = 0u128;

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
            max_job_work = max_job_work.max(job_work);

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
        //
        // Pre-size the job vector and push in order: a `Result` collect has no
        // lower size hint, so it would regrow this vector about log2(J) times,
        // each regrow copying a full `Plan<T>` per job.
        let mut child_config = config.clone();
        child_config.fresh_output = false;
        let mut planned = Vec::with_capacity(prepared.len());
        for meta in prepared {
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
            let plan = Plan::new(&problem, &child_config).map_err(|source| Error::Contract {
                step: meta.id,
                source,
            })?;
            planned.push(PreparedJob { meta, plan });
        }
        Ok(Self {
            jobs: planned,
            required_lhs,
            required_rhs,
            required_out,
            work,
            max_job_work,
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

    /// Elements of the caller-owned intermediate arena this grouped plan needs.
    ///
    /// A grouped plan keeps no cpueinsum intermediate: each job is one tcontract
    /// plan that runs from the executor's workspace. This is therefore zero, and
    /// the native requirement is separate and component-wise; see
    /// [`native_workspace_req`](Self::native_workspace_req).
    pub fn workspace_len(&self) -> usize {
        0
    }

    /// The component-wise workspace the jobs' native plans need on `exec` for an
    /// `output` storage contract, folded field by field with a maximum.
    ///
    /// This is a conservative bound on the outer `exec`; each job also runs on
    /// the child context [`GroupedPlan::execute_into`] derives (one thread per
    /// item under `Auto`), so callers that size a
    /// [`Workspace`](tcontract_exec::Workspace) directly should use the item
    /// width the chosen policy grants. See
    /// [`EinsumPlan::native_workspace_req`](crate::EinsumPlan::native_workspace_req)
    /// for the workspace-preparation contract.
    pub fn native_workspace_req(&self, exec: &Exec<'_>, output: OutputContract) -> WorkspaceReq {
        let mut req = WorkspaceReq::default();
        for job in &self.jobs {
            let r = job.plan.workspace_req(exec, output);
            req.a_bytes = req.a_bytes.max(r.a_bytes);
            req.tile_bytes = req.tile_bytes.max(r.tile_bytes);
            req.worker_scatter = req.worker_scatter.max(r.worker_scatter);
            req.b_bytes = req.b_bytes.max(r.b_bytes);
            req.team_scatter = req.team_scatter.max(r.team_scatter);
            req.barriers = req.barriers.max(r.barriers);
        }
        req
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
    /// let route = plan.execute_into(&mut Exec::serial(), 2.0, &[1., 2., 3., 4.],
    ///     &[1., 1.], &mut d)?;
    /// assert_eq!(d, [8.0, 12.0]);
    /// assert_eq!(route, GroupedRoute::Serial);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] when any declared range is outside a shared
    /// buffer, and [`Error::Contract`] for a child execution failure.
    ///
    /// All bounds and route checks happen before the first write, so a
    /// pre-write failure leaves the output untouched; a child execution failure
    /// may leave other jobs' outputs written. When several jobs fail, the
    /// error returned names the job with the lowest original descriptor index,
    /// identical for serial and multi-lane execution.
    pub fn execute_into(
        &self,
        exec: &mut Exec<'_>,
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
    /// plan.execute_into_accum(&mut Exec::serial(), 2.0, &[3.0], &[4.0], 0.5, &mut d)?;
    /// assert_eq!(d, [28.0]);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] when any declared range is outside a shared
    /// buffer, and [`Error::Contract`] for a child execution failure.
    ///
    /// All bounds and route checks happen before the first write, so a
    /// pre-write failure leaves the output untouched; a child execution failure
    /// may leave other jobs' outputs written. When several jobs fail, the
    /// error returned names the job with the lowest original descriptor index,
    /// identical for serial and multi-lane execution.
    pub fn execute_into_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        out: &mut [T],
    ) -> Result<GroupedRoute> {
        self.execute(exec, alpha, lhs, rhs, beta, true, out)
    }

    /// One error rule for every failure site: the reported failure names the
    /// lowest original job id, so it never depends on the stored (output-offset)
    /// order or on lane scheduling.
    fn execute(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        accumulate: bool,
        out: &mut [T],
    ) -> Result<GroupedRoute> {
        // Resolve the route first, before bounds checks, the empty shortcut or
        // any write, so the outer width is fixed before the children run.
        let (lanes, item_threads) = self.lane_decision(exec)?;

        // Scan every job before any write and report the failure naming the
        // lowest original job id: the stored order is by output offset, so it
        // must not pick the error.
        let mut bounds_failure: Option<(usize, Error)> = None;
        for job in &self.jobs {
            let id = job.meta.id;
            for (operand, offset, len, available) in [
                (
                    GroupedOperand::Lhs,
                    job.meta.lhs_offset,
                    job.meta.lhs_len,
                    lhs.len(),
                ),
                (
                    GroupedOperand::Rhs,
                    job.meta.rhs_offset,
                    job.meta.rhs_len,
                    rhs.len(),
                ),
                (
                    GroupedOperand::Output,
                    job.meta.out_offset,
                    job.meta.out_len,
                    out.len(),
                ),
            ] {
                if let Err(error) = check_bounds(id, operand, offset, len, available) {
                    keep_lowest(&mut bounds_failure, id, error);
                }
            }
        }
        if let Some((_, error)) = bounds_failure {
            return Err(error);
        }
        let source = if accumulate && beta != T::default() {
            SliceAccumulationSource::Output
        } else {
            SliceAccumulationSource::Absent
        };
        {
            let child_exec = child_context(exec, item_threads);
            let mut route_failure: Option<(usize, Error)> = None;
            for job in &self.jobs {
                let id = job.meta.id;
                // Admission only: a child that would take a packed SPMD team on the
                // coordinator is not refused here. tcontract's packed driver checks
                // `can_broadcast` on the token it is given, so a child running on a
                // pool worker takes the barrier-free route, before any lane writes.
                if let Err(source) =
                    job.plan
                        .execution_route(&child_exec, alpha, OutputContract::Initialized)
                {
                    keep_lowest(&mut route_failure, id, Error::Contract { step: id, source });
                }
            }
            if let Some((_, error)) = route_failure {
                return Err(error);
            }
        }

        if self.jobs.is_empty() {
            return Ok(GroupedRoute::Empty);
        }
        let route = grouped_route(lanes, item_threads);

        if lanes == 1 {
            // Serial: drive the jobs by index and carve each output slice in
            // turn. Nothing is collected, so this path allocates none of its
            // own (tcontract's own workspace aside).
            let mut child_exec = child_context(exec, item_threads);
            let mut cursor = 0usize;
            let mut remaining: &mut [T] = out;
            let mut failure: Option<(usize, Error)> = None;
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
                let mut item = JobItem {
                    id: job.meta.id,
                    plan: &job.plan,
                    lhs: &lhs[job.meta.lhs_offset..job.meta.lhs_offset + job.meta.lhs_len],
                    rhs: &rhs[job.meta.rhs_offset..job.meta.rhs_offset + job.meta.rhs_len],
                    out: d,
                };
                if let Err(error) = run_item(&mut item, &mut child_exec, alpha, beta, source) {
                    // Keep running the remaining jobs: a failure here must not
                    // hide a lower-id one that runs later in stored order.
                    keep_lowest(&mut failure, item.id, error);
                }
            }
            return failure.map_or(Ok(route), |(_, error)| Err(error));
        }

        // Parallel: one contiguous job range per lane, each owning the output
        // chunk that covers its jobs' ranges. Only `lanes` (already clamped to
        // the budget) chunks are collected -- never one slot per job, and no
        // Mutex per lane either, since each lane holds its own range.
        //
        // Boundaries are monotone even though empty jobs sort last and may carry
        // a smaller offset: an empty job contributes no output, so it must never
        // pull a boundary back and leave an earlier non-empty job without room.
        let item_count = self.jobs.len();
        let lane_range = |lane: usize| {
            let base = item_count / lanes;
            let rem = item_count % lanes;
            let lo = lane * base + lane.min(rem);
            (lo, lo + base + usize::from(lane < rem))
        };
        let mut lane_chunks = Vec::with_capacity(lanes);
        let out_len = out.len();
        let mut remaining: &mut [T] = out;
        let mut cursor = 0usize;
        for lane in 0..lanes {
            let (lo, hi) = lane_range(lane);
            // End of this lane's chunk: the furthest non-empty output end among
            // its jobs, never before the previous boundary. Empty jobs add
            // nothing, so a lane at or after the transition to empty jobs gets an
            // empty chunk.
            let mut end = cursor;
            for job in &self.jobs[lo..hi] {
                if job.meta.out_len != 0 {
                    end = end.max(job.meta.out_offset + job.meta.out_len);
                }
            }
            if lane + 1 == lanes {
                end = out_len;
            }
            let current = core::mem::take(&mut remaining);
            let (chunk, tail) = current.split_at_mut(end - cursor);
            remaining = tail;
            lane_chunks.push(Mutex::new((cursor, chunk)));
            cursor = end;
        }
        let failure: Mutex<Option<(usize, Error)>> = Mutex::new(None);
        // Each lane is one forked child of the caller's token: `fork` gives
        // lane `i` its share of the budget, and every share is at least
        // `item_threads` because `lanes * item_threads <= budget`.
        let workspace = exec.workspace();
        let policy = child_policy(item_threads);
        let item_budget = NonZeroUsize::new(item_threads).unwrap_or(NonZeroUsize::MIN);
        exec.token().fork(lanes, |lane, token| {
            let mut lane_exec = Exec::from_token(token.limit(item_budget), workspace, policy);
            let (lo, hi) = lane_range(lane);
            let mut guard = lane_chunks[lane]
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let (base_offset, chunk) = &mut *guard;
            let base_offset = *base_offset;
            let mut lane_cursor = 0usize;
            for job in &self.jobs[lo..hi] {
                let d = if job.meta.out_len == 0 {
                    &mut [][..]
                } else {
                    let start = job.meta.out_offset - base_offset;
                    let current = core::mem::take(chunk);
                    let (_, rest) = current.split_at_mut(start - lane_cursor);
                    let (slice, tail) = rest.split_at_mut(job.meta.out_len);
                    *chunk = tail;
                    lane_cursor = start + job.meta.out_len;
                    slice
                };
                let mut item = JobItem {
                    id: job.meta.id,
                    plan: &job.plan,
                    lhs: &lhs[job.meta.lhs_offset..job.meta.lhs_offset + job.meta.lhs_len],
                    rhs: &rhs[job.meta.rhs_offset..job.meta.rhs_offset + job.meta.rhs_len],
                    out: d,
                };
                if let Err(error) = run_item(&mut item, &mut lane_exec, alpha, beta, source) {
                    let mut slot = failure.lock().unwrap_or_else(|poison| poison.into_inner());
                    // Recording by id, not by arrival, keeps the reported error
                    // independent of lane scheduling.
                    keep_lowest(&mut slot, item.id, error);
                }
            }
        });
        failure
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner())
            .map_or(Ok(route), |(_, error)| Err(error))
    }

    /// Resolve `(lanes, item_threads)` from the caller's batch policy.
    ///
    /// `Auto` is the width the total estimated work pays for under the token's
    /// dispatch cost (`Exec::width_for`, one section: the lanes are one fork over
    /// the jobs), clamped to the budget and the job count -- because neither the
    /// shared generic `Auto` formula (which fans two tiny scalar jobs onto two
    /// lanes) nor the packed balanced-lanes rule (which adds a near-even
    /// restriction here absent) is this route's rule.
    ///
    /// Every forced strategy goes through [`Exec::batch_decision`] on the
    /// original host budget, which clamps but never falls back.
    fn lane_decision(&self, exec: &Exec<'_>) -> Result<(usize, usize)> {
        if let BatchStrategy::Auto = exec.batch_policy().strategy {
            let serial_ns =
                2.0 * self.work as f64 * ns_per_flop() * if T::IS_COMPLEX { 4.0 } else { 1.0 };
            let lanes = exec
                .width_for(serial_ns, GROUPED_SECTIONS)
                .min(exec.budget())
                .min(self.jobs.len())
                .max(1);
            return Ok((lanes, 1));
        }
        let query =
            BatchQuery::new(self.jobs.len(), self.max_job_work as f64).with_item_parallel(true);
        let decision = exec.batch_decision(&query);
        Ok((decision.lanes, decision.item_threads))
    }
}

/// The child context every item of a grouped execution runs on.
///
/// With one thread per item this is the historical serial context that borrows
/// the caller's workspace, so `Auto` (item-serial) is byte-for-byte what it was.
/// A forced strategy that grants item threads gets a child bounded to that width
/// on the caller's pool, with its own policy forced to [`BatchStrategy::Inner`]
/// so the granted width is actually executed: a reset to `Auto` would let a
/// small item fall back below the Auto threshold and run serially while the
/// route still reported several threads.
fn child_context<'b>(exec: &'b mut Exec<'_>, item_threads: usize) -> Exec<'b> {
    let workspace = exec.workspace();
    if item_threads <= 1 {
        return workspace.map_or(Exec::serial(), Exec::serial_with_workspace);
    }
    let budget = NonZeroUsize::new(item_threads).unwrap_or(NonZeroUsize::MIN);
    Exec::from_token(
        exec.token().limit(budget),
        workspace,
        child_policy(item_threads),
    )
}

/// The batch policy of an item context: serial items keep the default policy,
/// and a granted item width is forced to [`BatchStrategy::Inner`] so it runs.
fn child_policy(item_threads: usize) -> BatchPolicy {
    if item_threads <= 1 {
        BatchPolicy::default()
    } else {
        BatchPolicy::new(BatchStrategy::Inner)
    }
}

/// The route a resolved `(lanes, item_threads)` pair describes.
fn grouped_route(lanes: usize, item_threads: usize) -> GroupedRoute {
    if lanes > 1 && item_threads > 1 {
        GroupedRoute::Split {
            lanes,
            item_threads,
        }
    } else if lanes > 1 {
        GroupedRoute::Outer { lanes }
    } else if item_threads > 1 {
        GroupedRoute::Inner {
            threads: item_threads,
        }
    } else {
        GroupedRoute::Serial
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
    exec: &mut Exec<'_>,
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

/// Keep the failure naming the lowest original job id.
///
/// Jobs are stored by output offset and lanes observe failures in scheduling
/// order, so the first failure seen is not the one a serial run reports.
/// Recording by id makes the reported error independent of both.
fn keep_lowest(slot: &mut Option<(usize, Error)>, id: usize, error: Error) {
    match slot {
        Some((recorded, _)) if *recorded <= id => {}
        _ => *slot = Some((id, error)),
    }
}

#[cfg(test)]
mod tests {
    use super::keep_lowest;
    use crate::{Error, GroupedError, GroupedOperand};

    fn bounds(job: usize) -> Error {
        Error::Grouped(GroupedError::Bounds {
            job,
            operand: GroupedOperand::Lhs,
            end: job + 1,
            available: job,
        })
    }

    #[test]
    fn lowest_job_id_wins_in_any_arrival_order() {
        // Lane children fail in scheduling order, so the reduction must not
        // depend on it.
        for order in [[5usize, 2, 9], [2, 5, 9], [9, 5, 2], [9, 2, 5]] {
            let mut slot = None;
            for job in order {
                keep_lowest(&mut slot, job, bounds(job));
            }
            match slot {
                Some((job, error)) => {
                    assert_eq!(job, 2, "arrival order {order:?}");
                    assert!(
                        matches!(error, Error::Grouped(GroupedError::Bounds { job: 2, .. })),
                        "arrival order {order:?}"
                    );
                }
                None => panic!("a failure was recorded for order {order:?}"),
            }
        }
    }
}
