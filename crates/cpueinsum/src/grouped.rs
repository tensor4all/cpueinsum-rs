//! Prepared heterogeneous column-major GEMM groups.
//!
//! A [`GroupedPlan`] holds one immutable tcontract plan per distinct geometry,
//! shared through an [`Arc`] so a caller-owned cache can hand the same child to
//! many grouped plans ([`GroupedPlan::from_prepared_with`], cpueinsum-rs#79).
//! The caller lends flat input and output buffers at execution time; no
//! pointers or executors are retained by the plan.

use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::ns_per_flop;
use tcontract::{
    execute_uninit_with, OutputContract, Plan, PlanConfig, SliceAccumulationSource,
    UninitOutputError,
};
use tcontract_exec::{BatchPolicy, BatchQuery, BatchStrategy, Exec, WorkspaceReq};

use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, GemmRun, Trans};
use t4a_desc::ScalarKind;

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
    /// A fresh-output call was given a nonzero `beta`.
    #[error("fresh grouped output requires beta = 0")]
    NonzeroBeta,
    /// The jobs do not cover the fresh output exactly once, in order, from 0.
    #[error("grouped jobs do not cover the fresh output exactly once without gaps")]
    FreshCoverage,
    /// A child plan supplied to [`GroupedPlan::from_prepared_with`] was not
    /// planned for the problem its [`ChildKey`] describes.
    #[error("the child plan supplied for grouped job {job} does not match its key")]
    ChildMismatch {
        /// Lowest original job index with the child's geometry.
        job: usize,
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

/// Preparation options of a [`GroupedPlan`].
///
/// The default is today's per-job route: one child execution per job.
///
/// # Examples
///
/// ```
/// use cpueinsum::GroupedOptions;
/// assert!(!GroupedOptions::default().strided_runs());
/// assert!(GroupedOptions::default().with_strided_runs(true).strided_runs());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupedOptions {
    strided_runs: bool,
}

impl GroupedOptions {
    /// Execute each batched strided run as one child (cpueinsum-rs#78).
    ///
    /// A *strided run* is a maximal contiguous sequence of jobs, in the
    /// caller's order, with the same `(rows, contracted, cols)` whose `lhs`,
    /// `rhs` and `out` offsets each advance by one constant stride, with the
    /// output stride at least `rows * cols`. The partition is
    /// [`GemmBatch::runs`] of the tensor4all/tcontract-rs#209 descriptor:
    /// runs are taken greedily from the left, so a job that could end one run
    /// or start the next belongs to the earlier one. A run is *batched* when it
    /// has at least two jobs, a non-empty output, and no other job's output
    /// inside its output span; every other job keeps the per-job route,
    /// bit-identically. A batched run executes as one tcontract problem with a
    /// batch (Hadamard) label over the run, through the same child execution as
    /// a single job.
    ///
    /// The route is opt-in until the corpus campaign shows where it wins
    /// (cpueinsum-rs#80). See [`GroupedPlan::strided_runs`] for the partition.
    pub const fn with_strided_runs(mut self, on: bool) -> Self {
        self.strided_runs = on;
        self
    }

    /// Whether batched strided runs are enabled.
    /// See [`GroupedOptions`] for a runnable example.
    pub const fn strided_runs(&self) -> bool {
        self.strided_runs
    }
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

#[derive(Clone, Copy, Debug)]
struct PreparedJob {
    meta: JobMeta,
    plan: u32,
}

/// The part of one shared buffer an item reads or writes: `start..start + len`,
/// with the child's origin at `start + origin`.
#[derive(Clone, Copy, Debug, Default)]
struct Window {
    start: usize,
    len: usize,
    origin: usize,
}

impl Window {
    /// A single job's compact range, its origin at the start.
    fn job(offset: usize, len: usize) -> Self {
        Self {
            start: offset,
            len,
            origin: 0,
        }
    }

    /// The union of `count` ranges of `len` elements at `offset + i * stride`.
    /// An empty range reads nothing, so its window is empty at `offset`.
    fn run(offset: usize, len: usize, stride: isize, count: usize) -> Self {
        if len == 0 {
            return Self::job(offset, 0);
        }
        // INVARIANT: every job of the run passed `prepare_jobs`, so both
        // extreme offsets and their ends are representable.
        let last = offset as isize + stride * (count as isize - 1);
        let (lo, hi) = if stride < 0 {
            (last as usize, offset)
        } else {
            (offset, last as usize)
        };
        Self {
            start: lo,
            len: hi - lo + len,
            origin: offset - lo,
        }
    }

    fn end(&self) -> usize {
        self.start + self.len
    }
}

/// One scheduled unit of a grouped execution: a single job, or a batched
/// strided run named by its first (lowest) original job id.
#[derive(Clone, Copy, Debug)]
struct Item {
    id: usize,
    plan: u32,
    lhs: Window,
    rhs: Window,
    out: Window,
}

impl Item {
    fn job(job: &PreparedJob) -> Self {
        let meta = &job.meta;
        Self {
            id: meta.id,
            plan: job.plan,
            lhs: Window::job(meta.lhs_offset, meta.lhs_len),
            rhs: Window::job(meta.rhs_offset, meta.rhs_len),
            out: Window::job(meta.out_offset, meta.out_len),
        }
    }
}

/// One distinct batched-run child: its geometry, its run length and its three
/// offset strides, with the lowest original id of a run that uses it.
#[derive(Clone, Copy, Debug)]
struct RunChild {
    geometry: u32,
    len: usize,
    stride: [isize; 3],
    id: usize,
}

/// Checked grouped job metadata, before any numerical child plan exists.
///
/// Produced by [`GroupedPlan::prepare_jobs`] and consumed by
/// [`GroupedPlan::from_prepared`] or [`GroupedPlan::from_prepared_with`]. It is
/// opaque: it exists so the checks run once before the children are chosen.
/// `cpueinsum-blas` uses it so a vendor plan pays the checked arithmetic, the
/// output overlap admission and the geometry count, and builds the tcontract
/// children only when it actually runs natively (cpueinsum-rs#52); a caller
/// with its own child cache passes it to `from_prepared_with` (cpueinsum-rs#79).
///
/// # Examples
///
/// See [`GroupedPlan::from_prepared_with`].
#[derive(Clone, Debug)]
pub struct PreparedGroupedJobs {
    jobs: Vec<PreparedJob>,
    /// One entry per distinct local geometry, in child-index order.
    children: Vec<ChildGeometry>,
    required_lhs: usize,
    required_rhs: usize,
    required_out: usize,
    work: u128,
    max_job_work: u128,
    /// The batched strided runs, in original order; empty unless enabled.
    runs: Vec<GemmRun>,
    /// One entry per distinct batched-run child, after the geometry children.
    run_children: Vec<RunChild>,
    /// The execution items in output order when any run is batched; empty
    /// otherwise, and then every job is its own item.
    items: Vec<Item>,
    /// The largest work of one item.
    max_item_work: u128,
}

/// One distinct local geometry and the lowest original id that uses it.
#[derive(Clone, Copy, Debug)]
struct ChildGeometry {
    rows: usize,
    contracted: usize,
    cols: usize,
    id: usize,
}

/// Everything one child plan of a [`GroupedPlan`] depends on.
///
/// A key is handed to the `child` closure of
/// [`GroupedPlan::from_prepared_with`], once per distinct geometry. It carries
/// the dtype, the `(rows, contracted, cols)` geometry, the column-major leading
/// dimensions `(lda, ldb, ldc)`, the three operations, and the effective child
/// [`PlanConfig`] (the caller's config with `fresh_output` cleared, because the
/// same child serves initialized and fresh output). Equal keys always describe
/// the same problem under the same config, so a caller can key a cache by it
/// across grouped plans. The converse is not promised: two keys whose ops
/// tcontract lowers alike (for example a conjugation on a real dtype) differ,
/// which costs a cache miss, never a wrong child.
///
/// The leading dimensions are those of today's compact jobs, `max(rows, 1)`,
/// `max(contracted, 1)` and `max(rows, 1)`, so they follow from the shape for
/// now. They are in the key in anticipation of jobs that carry their own
/// leading dimensions (the tensor4all/tcontract-rs#209 descriptor), which is
/// not implemented here.
///
/// [`Hash`] covers every field except the config, which has no `Hash`; [`Eq`]
/// compares the config too. A cache that mixes many configs therefore puts
/// their keys in the same buckets, which stays correct.
///
/// The host facts a plan is tuned against (CPU features, cache shape, tuning
/// class) are not in the key because they are fixed for the process; a cache is
/// therefore valid within one process, not across hosts.
///
/// # Examples
///
/// ```
/// use cpueinsum::{ChildKey, GroupedGemmJob, GroupedPlan};
/// let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 3, 4)];
/// let mut seen: Vec<ChildKey> = Vec::new();
/// let plan = GroupedPlan::<f64>::from_prepared_with(
///     GroupedPlan::<f64>::prepare_jobs(&jobs)?,
///     Default::default(),
///     &Default::default(),
///     |key| {
///         seen.push(key.clone());
///         key.build::<f64>().map(std::sync::Arc::new)
///     },
/// )?;
/// assert_eq!(seen[0].shape(), (2, 3, 4));
/// assert_eq!(seen[0].leading_dims(), (2, 3, 2));
/// assert_eq!(plan.required_out_len(), 8);
/// # Ok::<(), cpueinsum::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChildKey {
    dtype: DType,
    rows: usize,
    contracted: usize,
    cols: usize,
    // Stored as strides: whoever sets them proves they fit `isize`, so
    // `problem` never has to convert (or saturate) one.
    lda: isize,
    ldb: isize,
    ldc: isize,
    ops: [Op; 3],
    config: PlanConfig,
}

impl Hash for ChildKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // `PlanConfig` has no `Hash`; leaving it out keeps `a == b => hash(a) ==
        // hash(b)`, which is all `Hash` requires.
        self.dtype.hash(state);
        (self.rows, self.contracted, self.cols).hash(state);
        (self.lda, self.ldb, self.ldc).hash(state);
        self.ops.hash(state);
    }
}

impl ChildKey {
    fn new(dtype: DType, ops: [Op; 3], config: &PlanConfig) -> Self {
        let mut config = config.clone();
        config.fresh_output = false;
        Self {
            dtype,
            rows: 0,
            contracted: 0,
            cols: 0,
            lda: 1,
            ldb: 1,
            ldc: 1,
            ops,
            config,
        }
    }

    /// Point the key at one compact geometry. [`GroupedPlan::prepare_jobs`]
    /// checked every dimension against `isize` (`isize_dim`), so the casts are
    /// lossless; a job format with its own leading dimensions must check them
    /// the same way before setting them here.
    fn set_geometry(&mut self, child: &ChildGeometry) {
        self.rows = child.rows;
        self.contracted = child.contracted;
        self.cols = child.cols;
        self.lda = (child.rows as isize).max(1);
        self.ldb = (child.contracted as isize).max(1);
        self.ldc = (child.rows as isize).max(1);
    }

    /// The element type of the child.
    /// See [`ChildKey`] for a runnable example.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The child's `(rows, contracted, cols)`, unclamped: an empty reduction
    /// `(m, 0, n)` and a one-element one `(m, 1, n)` are different children.
    /// See [`ChildKey`] for a runnable example.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.rows, self.contracted, self.cols)
    }

    /// The column-major leading dimensions `(lda, ldb, ldc)` of the child's A, B
    /// and D. See [`ChildKey`] for a runnable example.
    pub fn leading_dims(&self) -> (usize, usize, usize) {
        // Positive by construction.
        (self.lda as usize, self.ldb as usize, self.ldc as usize)
    }

    /// The A, B and D operations; the third also applies to the accumulation
    /// source. See [`ChildKey`] for a runnable example.
    pub fn ops(&self) -> [Op; 3] {
        self.ops
    }

    /// The effective child config: the grouped plan's config with
    /// `fresh_output` cleared. See [`ChildKey`] for a runnable example.
    pub fn config(&self) -> &PlanConfig {
        &self.config
    }

    /// The tcontract problem the key describes: `D[i,j] = sum_l A[i,l] B[l,j]`
    /// over column-major operands with this key's leading dimensions.
    ///
    /// A plan supplied for this key must report exactly this problem from
    /// [`Plan::problem`].
    ///
    /// # Errors
    ///
    /// Returns tcontract's error when it rejects the problem.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{ChildKey, GroupedGemmJob, GroupedPlan};
    /// let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];
    /// let mut problems = Vec::new();
    /// GroupedPlan::<f32>::from_prepared_with(
    ///     GroupedPlan::<f32>::prepare_jobs(&jobs)?, Default::default(), &Default::default(),
    ///     |key| {
    ///         problems.push(key.problem()?);
    ///         key.build::<f32>().map(std::sync::Arc::new)
    ///     },
    /// )?;
    /// assert_eq!(problems.len(), 1);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    pub fn problem(&self) -> tcontract::Result<Problem> {
        let (m, k, n) = (self.rows, self.contracted, self.cols);
        Problem::from_labels(
            self.dtype,
            spec(&[m, k], &[1, self.lda], self.ops[0])?,
            spec(&[k, n], &[1, self.ldb], self.ops[1])?,
            CSpec::Output(self.ops[2]),
            spec(&[m, n], &[1, self.ldc], self.ops[2])?,
            &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
        )
    }

    /// The batched child of a strided run of `len` jobs with this key's
    /// geometry whose offsets advance by `stride` (`lhs`, `rhs`, `out`):
    /// `D[i,j,h] = sum_l A[i,l,h] B[l,j,h]`, the batch label `h` over the run.
    /// A has dims `[m, k, len]` and strides `[1, lda, stride[0]]`, and likewise
    /// B and D; the origin of each operand is the run's first job.
    fn run_problem(&self, len: usize, stride: [isize; 3]) -> tcontract::Result<Problem> {
        let (m, k, n) = (self.rows, self.contracted, self.cols);
        Problem::from_labels(
            self.dtype,
            spec(&[m, k, len], &[1, self.lda, stride[0]], self.ops[0])?,
            spec(&[k, n, len], &[1, self.ldb, stride[1]], self.ops[1])?,
            CSpec::Output(self.ops[2]),
            spec(&[m, n, len], &[1, self.ldc, stride[2]], self.ops[2])?,
            &Labels::new(&[0, 1, 3], &[1, 2, 3], &[0, 2, 3]),
        )
    }

    /// Build the child plan for this key exactly as [`GroupedPlan::from_prepared`]
    /// does, so a cache's miss path is one call.
    ///
    /// # Errors
    ///
    /// Returns tcontract's error when it rejects the problem or the config,
    /// including a `T` whose storage type is not [`ChildKey::dtype`].
    ///
    /// # Examples
    ///
    /// See [`GroupedPlan::from_prepared_with`] for a caller-side cache.
    pub fn build<T: Scalar>(&self) -> tcontract::Result<Plan<T>> {
        let plan = Plan::from_problem(self.problem()?, &self.config)?;
        #[cfg(test)]
        tests::PLAN_CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
        Ok(plan)
    }
}

impl PreparedGroupedJobs {
    /// The batched strided runs, in original job order: each has at least two
    /// jobs and executes as one child. Empty unless
    /// [`GroupedOptions::with_strided_runs`] was set; jobs outside every listed
    /// run take the per-job route.
    ///
    /// The partition is a function of the job list alone (and of the element
    /// size, through the checked byte spans), computed once here; see
    /// [`GroupedOptions::with_strided_runs`] for the rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{GroupedGemmJob, GroupedOptions, GroupedPlan};
    /// // Four 2x2x2 jobs advancing by 4 in every operand, then one irregular job.
    /// let mut jobs: Vec<_> = (0..4).map(|i| GroupedGemmJob::new(4 * i, 4 * i, 4 * i, 2, 2, 2)).collect();
    /// jobs.push(GroupedGemmJob::new(0, 0, 16, 1, 1, 1));
    /// let options = GroupedOptions::default().with_strided_runs(true);
    /// let prepared = GroupedPlan::<f64>::prepare_jobs_with(&jobs, options)?;
    /// let runs = prepared.strided_runs();
    /// assert_eq!((runs.len(), runs[0].first, runs[0].len, runs[0].stride), (1, 0, 4, [4, 4, 4]));
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    pub fn strided_runs(&self) -> &[GemmRun] {
        &self.runs
    }

    /// Number of scheduled items: one per batched run and one per other job.
    fn item_count(&self) -> usize {
        if self.items.is_empty() {
            self.jobs.len()
        } else {
            self.items.len()
        }
    }

    /// Item `index` in output order. Without batched runs every job is its own
    /// item. The execution loops read their item source directly (`run_items`);
    /// this is the tests' view of the same list.
    #[cfg(test)]
    fn item(&self, index: usize) -> Item {
        if self.items.is_empty() {
            Item::job(&self.jobs[index])
        } else {
            self.items[index]
        }
    }

    /// Reject a `beta`/coverage combination a fresh-output call cannot support.
    ///
    /// The same rule [`GroupedPlan::validate_uninit`] applies, on metadata only,
    /// so a vendor plan can enforce it without a native plan.
    ///
    /// # Errors
    ///
    /// [`GroupedError::NonzeroBeta`] when `beta` is not the additive identity,
    /// and [`GroupedError::FreshCoverage`] when the jobs leave a gap or do not
    /// stop exactly at `out_len`.
    pub fn validate_uninit<T: Scalar>(&self, beta: T, out_len: usize) -> Result<()> {
        if beta != T::default() {
            return Err(grouped(GroupedError::NonzeroBeta));
        }
        let mut cursor = 0usize;
        for job in &self.jobs {
            if job.meta.out_len == 0 {
                continue;
            }
            // Stored order is by output offset, so exact coverage means each
            // nonempty job starts where the previous one stopped.
            if job.meta.out_offset != cursor {
                return Err(grouped(GroupedError::FreshCoverage));
            }
            cursor = job
                .meta
                .out_offset
                .checked_add(job.meta.out_len)
                .ok_or_else(|| {
                    grouped(GroupedError::Overflow {
                        job: job.meta.id,
                        what: "fresh output coverage",
                    })
                })?;
        }
        if cursor != out_len {
            return Err(grouped(GroupedError::FreshCoverage));
        }
        Ok(())
    }
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
    meta: PreparedGroupedJobs,
    plans: Vec<Arc<Plan<T>>>,
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
    /// `fresh_output` capability is unused: the same child plans support both
    /// initialized and fresh execution contracts.
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
        Self::from_prepared(Self::prepare_jobs(jobs)?, ops, config)
    }

    /// Prepare jobs with common A, B and D operations under `options`.
    ///
    /// With default options this is [`Self::new_with_ops`]. With
    /// [`GroupedOptions::with_strided_runs`], each batched strided run executes
    /// as one child; see [`Self::strided_runs`].
    ///
    /// # Errors
    ///
    /// As [`Self::new_with_ops`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{Exec, GroupedGemmJob, GroupedOptions, GroupedPlan};
    /// // Three 1x1x1 jobs advancing by one element: one batched run.
    /// let jobs: Vec<_> = (0..3).map(|i| GroupedGemmJob::new(i, i, i, 1, 1, 1)).collect();
    /// let options = GroupedOptions::default().with_strided_runs(true);
    /// let plan = GroupedPlan::<f64>::new_with_options(&jobs, Default::default(),
    ///     &Default::default(), options)?;
    /// assert_eq!(plan.strided_runs().len(), 1);
    /// let mut d = [0.0; 3];
    /// plan.execute_into(&mut Exec::serial(), 1.0, &[1., 2., 3.], &[4., 5., 6.], &mut d)?;
    /// assert_eq!(d, [4.0, 10.0, 18.0]);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    pub fn new_with_options(
        jobs: &[GroupedGemmJob],
        ops: [Op; 3],
        config: &PlanConfig,
        options: GroupedOptions,
    ) -> Result<Self> {
        Self::from_prepared(Self::prepare_jobs_with(jobs, options)?, ops, config)
    }

    /// The batched strided runs this plan executes as one child each, in
    /// original job order; empty unless enabled. See
    /// [`PreparedGroupedJobs::strided_runs`].
    /// See [`Self::new_with_options`] for a runnable example.
    pub fn strided_runs(&self) -> &[GemmRun] {
        self.meta.strided_runs()
    }

    /// Check every descriptor, the output overlap rule and the distinct
    /// geometries, without preparing any numerical child plan.
    ///
    /// This is the whole preparation a vendor route needs, and it is what lets
    /// `cpueinsum-blas` build its own jobs first and construct the tcontract
    /// children only if it ever executes natively (cpueinsum-rs#52).
    ///
    /// The checks are `T`'s: a byte span is `elements * size_of::<T>()`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Grouped`] for checked arithmetic or overlapping output
    /// ranges, with the original descriptor index in the error.
    pub fn prepare_jobs(jobs: &[GroupedGemmJob]) -> Result<PreparedGroupedJobs> {
        Self::prepare_jobs_with(jobs, GroupedOptions::default())
    }

    /// [`Self::prepare_jobs`] under `options`.
    ///
    /// With [`GroupedOptions::with_strided_runs`], this also computes the
    /// strided-run partition, the distinct batched-run children and the
    /// execution items, once; see [`PreparedGroupedJobs::strided_runs`]. With
    /// the default options it is exactly [`Self::prepare_jobs`].
    ///
    /// # Errors
    ///
    /// As [`Self::prepare_jobs`].
    ///
    /// # Examples
    ///
    /// See [`PreparedGroupedJobs::strided_runs`].
    pub fn prepare_jobs_with(
        jobs: &[GroupedGemmJob],
        options: GroupedOptions,
    ) -> Result<PreparedGroupedJobs> {
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
            prepared.push(PreparedJob {
                meta: JobMeta {
                    id,
                    lhs_len,
                    rhs_len,
                    out_len,
                    lhs_offset: job.lhs_offset,
                    rhs_offset: job.rhs_offset,
                    out_offset: job.out_offset,
                },
                plan: 0,
            });
        }

        // Keep output order deterministic and retain original IDs. Empty jobs
        // stay after nonempty jobs and receive independent empty slices below.
        let by_output =
            |job: &PreparedJob| (job.meta.out_len == 0, job.meta.out_offset, job.meta.id);
        prepared.sort_unstable_by_key(by_output);
        for pair in prepared.windows(2) {
            if pair[1].meta.out_len != 0
                && pair[0].meta.out_offset + pair[0].meta.out_len > pair[1].meta.out_offset
            {
                return Err(grouped(GroupedError::Overlap {
                    first: pair[0].meta.id,
                    second: pair[1].meta.id,
                }));
            }
        }
        // Every descriptor and the overlap rule are checked before any child is
        // prepared.
        //
        // One numerical child per distinct local geometry (#51). Visiting the
        // jobs in geometry order makes each geometry one contiguous run, so
        // finding the shared child costs no lookup table and the child vector is
        // sized once, to the number of distinct geometries rather than job count.
        // Geometry is the raw `(rows, contracted, cols)`: zero and one differ,
        // and `max(1)` applies only to strides. The id tie-break keeps the child
        // order deterministic and makes a rejected child name the lowest
        // original id in its run.
        let geometry = |job: &PreparedJob| {
            let source = jobs[job.meta.id];
            (source.rows, source.contracted, source.cols)
        };
        prepared.sort_unstable_by_key(|job| (geometry(job), job.meta.id));
        let mut children: Vec<ChildGeometry> = Vec::new();
        let mut previous = None;
        for job in prepared.iter_mut() {
            let id = job.meta.id;
            let key = geometry(job);
            if previous != Some(key) {
                previous = Some(key);
                let (rows, contracted, cols) = key;
                children.push(ChildGeometry {
                    rows,
                    contracted,
                    cols,
                    id,
                });
            }
            job.plan = child_index(id, children.len() - 1)?;
        }
        // Restore the output-sorted storage order the execution paths index by.
        prepared.sort_unstable_by_key(by_output);
        let mut prepared = PreparedGroupedJobs {
            jobs: prepared,
            children,
            required_lhs,
            required_rhs,
            required_out,
            work,
            max_job_work,
            runs: Vec::new(),
            run_children: Vec::new(),
            items: Vec::new(),
            max_item_work: max_job_work,
        };
        if options.strided_runs {
            let runs = batched_runs(jobs, scalar_kind(T::STORAGE), &prepared.jobs);
            if !runs.is_empty() {
                plan_runs(&mut prepared, jobs, runs)?;
            }
        }
        Ok(prepared)
    }

    /// Prepare one numerical child per distinct geometry of `prepared`.
    ///
    /// This builds each child with [`ChildKey::build`] through the same
    /// construction as [`Self::from_prepared_with`], so the child order, the
    /// sharing and the error ids are the same. It skips the supplied-plan
    /// check, because every child here is planned from its own key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Contract`] when tcontract rejects a child problem; the
    /// error names the lowest original id with that geometry.
    pub fn from_prepared(
        prepared: PreparedGroupedJobs,
        ops: [Op; 3],
        config: &PlanConfig,
    ) -> Result<Self> {
        // The built-in source plans each child from its own key, so checking the
        // result against the key again would only repeat the problem lowering.
        Self::with_children(prepared, ops, config, false, |key| {
            key.build::<T>().map(Arc::new)
        })
    }

    /// Prepare `prepared` with child plans supplied by the caller, for example
    /// from a cache that outlives this grouped plan (cpueinsum-rs#79).
    ///
    /// `child` is called exactly once per distinct geometry, in child-index
    /// order, before any execution. Child-index order is deterministic: the
    /// distinct `(rows, contracted, cols)` geometries in ascending lexicographic
    /// order, which does not depend on the order of the jobs. Every job with a
    /// geometry shares that geometry's child; ties among them are broken by the
    /// lowest original job id, which is the id an error names. The same job set
    /// therefore produces the same sequence of keys and the same job-to-child
    /// assignment however its jobs are ordered.
    ///
    /// Supplied plans are not trusted: each one must report from
    /// [`Plan::problem`] exactly the problem [`ChildKey::problem`] describes and
    /// must not be a rank-k plan, or construction fails before anything runs.
    /// The config the child was built under is not visible on a [`Plan`] and is
    /// not checked; a cache keyed by [`ChildKey`] gets it right by construction.
    ///
    /// The grouped plan keeps the returned [`Arc`]s and never rebuilds a child,
    /// so one cached plan can serve any number of grouped plans and calls.
    ///
    /// A child's memory is released when the last [`GroupedPlan`] holding it
    /// drops, not when a caller's cache evicts or clears the entry. A caller
    /// that budgets its cache in bytes therefore has to charge the children
    /// that live grouped plans still reference to its own retained memory, or
    /// drop those plans when the call ends.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Contract`] when `child` fails, and
    /// [`GroupedError::ChildMismatch`] when it returns a plan for another
    /// problem. Both name the lowest original job id with the child's geometry;
    /// construction stops at the first failing child in child-index order.
    ///
    /// # Examples
    ///
    /// A caller-owned cache shared by two grouped plans with different job
    /// lists; the second plan builds nothing:
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    /// use cpueinsum::tcontract::{Plan, PlanConfig};
    /// use cpueinsum::{ChildKey, Exec, GroupedGemmJob, GroupedPlan};
    ///
    /// let mut cache: HashMap<ChildKey, Arc<Plan<f64>>> = HashMap::new();
    /// let mut misses = 0;
    /// let mut prepare = |jobs: &[GroupedGemmJob]| {
    ///     GroupedPlan::<f64>::from_prepared_with(
    ///         GroupedPlan::<f64>::prepare_jobs(jobs)?,
    ///         Default::default(),
    ///         &PlanConfig::default(),
    ///         |key| {
    ///             if let Some(plan) = cache.get(key) {
    ///                 return Ok(Arc::clone(plan));
    ///             }
    ///             misses += 1;
    ///             let plan = Arc::new(key.build::<f64>()?);
    ///             cache.insert(key.clone(), Arc::clone(&plan));
    ///             Ok(plan)
    ///         },
    ///     )
    /// };
    /// let first = prepare(&[GroupedGemmJob::new(0, 0, 0, 1, 1, 1)])?;
    /// let second = prepare(&[
    ///     GroupedGemmJob::new(0, 0, 1, 1, 1, 1),
    ///     GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
    /// ])?;
    /// drop(prepare);
    /// assert_eq!(misses, 1);
    ///
    /// let mut d = [0.0; 2];
    /// second.execute_into(&mut Exec::serial(), 1.0, &[3.0], &[4.0], &mut d)?;
    /// assert_eq!(d, [12.0, 12.0]);
    /// # let _ = first;
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    pub fn from_prepared_with<F>(
        prepared: PreparedGroupedJobs,
        ops: [Op; 3],
        config: &PlanConfig,
        child: F,
    ) -> Result<Self>
    where
        F: FnMut(&ChildKey) -> tcontract::Result<Arc<Plan<T>>>,
    {
        Self::with_children(prepared, ops, config, true, child)
    }

    /// The construction shared by both constructors: one `child` call per
    /// distinct geometry, in child-index order. Only caller-supplied plans
    /// (`verify`) are checked against their key; built-in ones come from it.
    fn with_children<F>(
        prepared: PreparedGroupedJobs,
        ops: [Op; 3],
        config: &PlanConfig,
        verify: bool,
        mut child: F,
    ) -> Result<Self>
    where
        F: FnMut(&ChildKey) -> tcontract::Result<Arc<Plan<T>>>,
    {
        // One key, re-pointed per geometry: the config is cloned once per
        // grouped plan, not once per child.
        let mut key = ChildKey::new(T::STORAGE, ops, config);
        let mut plans = Vec::with_capacity(prepared.children.len());
        for geometry in &prepared.children {
            let id = geometry.id;
            key.set_geometry(geometry);
            let plan = child(&key).map_err(|source| Error::Contract { step: id, source })?;
            if verify {
                let expected = key
                    .problem()
                    .map_err(|source| Error::Contract { step: id, source })?;
                if plan.problem() != &expected || plan.rank_k().is_some() {
                    return Err(grouped(GroupedError::ChildMismatch { job: id }));
                }
            }
            plans.push(plan);
        }
        // Batched-run children (#78) are planned here from their own problems,
        // never supplied: a caller's cache sees exactly the per-geometry keys
        // whether or not runs are enabled.
        plans.reserve_exact(prepared.run_children.len());
        for run in &prepared.run_children {
            let id = run.id;
            key.set_geometry(&prepared.children[run.geometry as usize]);
            let plan = key
                .run_problem(run.len, run.stride)
                .and_then(|problem| Plan::from_problem(problem, key.config()))
                .map_err(|source| Error::Contract { step: id, source })?;
            plans.push(Arc::new(plan));
        }
        Ok(Self {
            meta: prepared,
            plans,
        })
    }

    /// Minimum length of the shared left-input buffer.
    /// See [`Self::new`] for a runnable example.
    pub fn required_lhs_len(&self) -> usize {
        self.meta.required_lhs
    }

    /// Minimum length of the shared right-input buffer.
    /// See [`Self::new`] for a runnable example.
    pub fn required_rhs_len(&self) -> usize {
        self.meta.required_rhs
    }

    /// Minimum length of the shared output buffer.
    /// See [`Self::new`] for a runnable example.
    pub fn required_out_len(&self) -> usize {
        self.meta.required_out
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
        for plan in &self.plans {
            let r = plan.workspace_req(exec, output);
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

    /// Reject a `beta`/coverage combination a fresh-output call cannot support.
    ///
    /// Coverage is exact physical coverage of `0..out_len` by the plan's jobs:
    /// nonempty jobs, in the stored output order, start at the running cursor
    /// and advance it, and the final cursor is exactly `out_len`. Empty jobs
    /// contribute nothing. The constructor already proves the ranges disjoint
    /// and their products representable, so this is one linear scan with no
    /// allocation. Both conditions are checked before any write.
    ///
    /// # Errors
    ///
    /// [`GroupedError::NonzeroBeta`] when `beta` is not the additive identity,
    /// and [`GroupedError::FreshCoverage`] when the jobs leave a gap or do not
    /// stop exactly at `out_len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{GroupedGemmJob, GroupedPlan};
    /// let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
    /// let plan = GroupedPlan::<f64>::new(&jobs, &Default::default())?;
    /// assert!(plan.validate_uninit(0.0, 4).is_ok());
    /// assert!(plan.validate_uninit(0.0, 5).is_err());
    /// assert!(plan.validate_uninit(1.0, 4).is_err());
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    pub fn validate_uninit(&self, beta: T, out_len: usize) -> Result<()> {
        self.meta.validate_uninit(beta, out_len)
    }

    /// Overwrite all jobs into fresh storage and return it only on success.
    ///
    /// `beta` must be zero and the jobs must cover `out` exactly once without
    /// gaps; both are checked before the first write, so a refused call leaves
    /// every element of `out` untouched and no initialized slice exists. The
    /// returned slice is the whole output, initialized by this call, plus the
    /// route resolved before it.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan, GroupedRoute};
    /// let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 1)];
    /// let plan = GroupedPlan::<f64>::new(&jobs, &Default::default())?;
    /// let mut d = [MaybeUninit::uninit(); 2];
    /// let (d, route) = plan.execute_uninit(&mut Exec::serial(), 2.0,
    ///     &[1., 2., 3., 4.], &[1., 1.], 0.0, &mut d)?;
    /// assert_eq!(d, [8.0, 12.0]);
    /// assert_eq!(route, GroupedRoute::Serial);
    /// # Ok::<(), cpueinsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Grouped`] with [`GroupedError::NonzeroBeta`] for a nonzero
    /// `beta` and [`GroupedError::FreshCoverage`] for gaps or trailing output,
    /// then [`Error::Grouped`] for out-of-range input/output spans and
    /// [`Error::Contract`] for a child execution failure. Bounds, coverage and
    /// route checks all happen before the first write; a child failure may leave
    /// some output slots written but exposes no initialized slice and the lowest
    /// original job id names the error.
    pub fn execute_uninit<'d>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        beta: T,
        out: &'d mut [MaybeUninit<T>],
    ) -> Result<(&'d mut [T], GroupedRoute)> {
        self.validate_uninit(beta, out.len())?;
        let widths = self.prepare(exec, alpha, lhs, rhs, out.len(), OutputContract::Fresh)?;
        let source = SliceAccumulationSource::Absent;
        // One publication: the lower helper proves the jobs' receipts cover the
        // storage, then hands out the initialized whole slice on success only.
        execute_uninit_with(out, |storage| {
            let (route, receipts) =
                self.run_items(exec, widths, lhs, rhs, storage, |item, exec| {
                    run_item_uninit(item, exec, alpha, beta, source)
                })?;
            Ok((receipts, route))
        })
        .map_err(|source| match source {
            UninitOutputError::Execution(source) => source,
            UninitOutputError::Coverage => grouped(GroupedError::FreshCoverage),
        })
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
        let widths = self.prepare(
            exec,
            alpha,
            lhs,
            rhs,
            out.len(),
            OutputContract::Initialized,
        )?;
        let source = if accumulate && beta != T::default() {
            SliceAccumulationSource::Output
        } else {
            SliceAccumulationSource::Absent
        };
        let (route, _receipts) = self.run_items(exec, widths, lhs, rhs, out, |item, exec| {
            run_item(item, exec, alpha, beta, source)
        })?;
        Ok(route)
    }

    /// Dispatch once between the job list (no batched run: every job is its own
    /// item, exactly the per-job route) and the item list (#78), so the item
    /// loops are specialised and carry no per-item branch.
    fn run_items<'d, D, R, F>(
        &self,
        exec: &mut Exec<'_>,
        widths: Widths,
        lhs: &[T],
        rhs: &[T],
        out: &'d mut [D],
        run: F,
    ) -> Result<(GroupedRoute, Vec<R>)>
    where
        D: Send,
        R: Send,
        F: Fn(JobItem<'_, 'd, T, D>, &mut Exec<'_>) -> Result<R> + Sync,
    {
        if self.meta.items.is_empty() {
            self.run_lanes(&self.meta.jobs[..], exec, widths, lhs, rhs, out, run)
        } else {
            self.run_lanes(&self.meta.items[..], exec, widths, lhs, rhs, out, run)
        }
    }

    /// Resolve the route and run every pre-write check: buffer bounds, then child
    /// admission for the given output contract. Returns the resolved widths.
    fn prepare(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        lhs: &[T],
        rhs: &[T],
        out_len: usize,
        contract: OutputContract,
    ) -> Result<Widths> {
        // Resolve the route first, before bounds checks, the empty shortcut or
        // any write, so the outer width is fixed before the children run.
        let widths = self.lane_decision(exec)?;

        // Construction checked every endpoint against these aggregate lengths.
        if lhs.len() < self.required_lhs_len()
            || rhs.len() < self.required_rhs_len()
            || out_len < self.required_out_len()
        {
            // Stored order is by output offset; retain lowest original error ID.
            let mut bounds_failure: Option<(usize, Error)> = None;
            for job in &self.meta.jobs {
                #[cfg(test)]
                tests::BOUNDS_VISITS.with(|count| count.set(count.get() + 1));
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
                        out_len,
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
        }
        {
            let child_exec = child_context(exec, widths);
            let mut route_failure: Option<(usize, Error)> = None;
            let mut admit = |id: usize, plan: u32| {
                // Admission only: a child that would take a packed SPMD team on the
                // coordinator is not refused here. tcontract's packed driver checks
                // `can_broadcast` on the token it is given, so a child running on a
                // pool worker takes the barrier-free route, before any lane writes.
                if let Err(source) =
                    self.plans[plan as usize].execution_route(&child_exec, alpha, contract)
                {
                    keep_lowest(&mut route_failure, id, Error::Contract { step: id, source });
                }
            };
            if self.meta.items.is_empty() {
                for job in &self.meta.jobs {
                    admit(job.meta.id, job.plan);
                }
            } else {
                for item in &self.meta.items {
                    admit(item.id, item.plan);
                }
            }
            if let Some((_, error)) = route_failure {
                return Err(error);
            }
        }
        Ok(widths)
    }

    /// Drive every job on the resolved widths, one receipt per successful job.
    ///
    /// The destination element type and the receipt type are the only things the
    /// initialized and the fresh path change: `D = T, R = ()` for an overwrite
    /// and `D = MaybeUninit<T>, R = &mut [T]` for fresh storage. Initialized
    /// receipts use no heap (`Vec<()>`); fresh receipts collect the initialized
    /// sub-slices. Parallel execution collects them lane-locally.
    fn run_lanes<'d, S, D, R, F>(
        &self,
        items: &[S],
        exec: &mut Exec<'_>,
        widths: Widths,
        lhs: &[T],
        rhs: &[T],
        out: &'d mut [D],
        run: F,
    ) -> Result<(GroupedRoute, Vec<R>)>
    where
        S: ItemSource,
        D: Send,
        R: Send,
        F: Fn(JobItem<'_, 'd, T, D>, &mut Exec<'_>) -> Result<R> + Sync,
    {
        if self.meta.jobs.is_empty() {
            return Ok((GroupedRoute::Empty, Vec::new()));
        }
        let Widths {
            lanes,
            item_threads,
            ..
        } = widths;
        let route = grouped_route(lanes, item_threads);
        let item_count = items.len();
        let plans = &self.plans[..];

        if lanes == 1 {
            // Serial: carve disjoint output slices, retaining completion
            // receipts (zero-sized for initialized output).
            let mut child_exec = child_context(exec, widths);
            let mut cursor = 0usize;
            let mut remaining: &'d mut [D] = out;
            let mut failure: Option<(usize, Error)> = None;
            let mut receipts: Vec<R> = Vec::new();
            for source in items {
                let id = source.id();
                let (out_start, out_len) = source.out_range();
                let current = core::mem::take(&mut remaining);
                let d = if out_len == 0 {
                    // An empty job writes nothing; borrow the empty front of the
                    // remaining output so the slice keeps the output lifetime.
                    let (slice, tail) = current.split_at_mut(0);
                    remaining = tail;
                    slice
                } else {
                    let start = out_start;
                    let end = start + out_len;
                    let (_, rest) = current.split_at_mut(start - cursor);
                    let (slice, tail) = rest.split_at_mut(end - start);
                    remaining = tail;
                    cursor = end;
                    slice
                };
                match run(source.job_item(plans, lhs, rhs, d), &mut child_exec) {
                    Ok(receipt) => receipts.push(receipt),
                    // Keep running the remaining jobs: a failure here must not
                    // hide a lower-id one that runs later in stored order.
                    Err(error) => keep_lowest(&mut failure, id, error),
                }
            }
            return match failure {
                Some((_, error)) => Err(error),
                None => Ok((route, receipts)),
            };
        }

        // Parallel: one contiguous job range per lane, each owning the output
        // chunk that covers its jobs' ranges. Only `lanes` (already clamped to
        // the budget) chunks are collected -- never one slot per job. Each
        // chunk sits behind its own uncontended `Mutex` only to lend it to its
        // lane through the shared `fork` callback.
        //
        // Boundaries are monotone even though empty jobs sort last and may carry
        // a smaller offset: an empty job contributes no output, so it must never
        // pull a boundary back and leave an earlier non-empty job without room.
        let lane_range = |lane: usize| {
            let base = item_count / lanes;
            let rem = item_count % lanes;
            let lo = lane * base + lane.min(rem);
            (lo, lo + base + usize::from(lane < rem))
        };
        let mut lane_chunks = Vec::with_capacity(lanes);
        let out_len = out.len();
        let mut remaining: &'d mut [D] = out;
        let mut cursor = 0usize;
        for lane in 0..lanes {
            let (lo, hi) = lane_range(lane);
            // End of this lane's chunk: the furthest non-empty output end among
            // its jobs, never before the previous boundary. Empty jobs add
            // nothing, so a lane at or after the transition to empty jobs gets an
            // empty chunk.
            let mut end = cursor;
            for source in &items[lo..hi] {
                let (out_start, out_len) = source.out_range();
                if out_len != 0 {
                    end = end.max(out_start + out_len);
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
        // Successful receipts are disjoint output slices, so they are collected
        // lane-locally and appended once per lane after the fork.
        let receipts: Mutex<Vec<R>> = Mutex::new(Vec::new());
        // Each lane is one forked child of the caller's token: `fork` gives
        // lane `i` its share of the budget, and every share is at least
        // `item_threads` because `lanes * item_threads <= budget`.
        let workspace = exec.workspace();
        let policy = widths.item_policy();
        let item_budget = NonZeroUsize::new(item_threads).unwrap_or(NonZeroUsize::MIN);
        exec.token().fork(lanes, |lane, token| {
            let mut lane_exec = Exec::from_token(token.limit(item_budget), workspace, policy);
            let (lo, hi) = lane_range(lane);
            let mut lane_receipts: Vec<R> = Vec::new();
            let mut guard = lane_chunks[lane]
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let (base_offset, chunk) = &mut *guard;
            let base_offset = *base_offset;
            let mut lane_cursor = 0usize;
            for source in &items[lo..hi] {
                let (out_start, out_len) = source.out_range();
                let current = core::mem::take(chunk);
                let d = if out_len == 0 {
                    let (slice, tail) = current.split_at_mut(0);
                    *chunk = tail;
                    slice
                } else {
                    let start = out_start - base_offset;
                    let (_, rest) = current.split_at_mut(start - lane_cursor);
                    let (slice, tail) = rest.split_at_mut(out_len);
                    *chunk = tail;
                    lane_cursor = start + out_len;
                    slice
                };
                match run(source.job_item(plans, lhs, rhs, d), &mut lane_exec) {
                    Ok(receipt) => lane_receipts.push(receipt),
                    Err(error) => {
                        let mut slot = failure.lock().unwrap_or_else(|poison| poison.into_inner());
                        // Recording by id, not by arrival, keeps the reported error
                        // independent of lane scheduling.
                        keep_lowest(&mut slot, source.id(), error);
                    }
                }
            }
            if !lane_receipts.is_empty() {
                receipts
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .append(&mut lane_receipts);
            }
        });
        match failure
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner())
        {
            Some((_, error)) => Err(error),
            None => Ok((
                route,
                receipts
                    .into_inner()
                    .unwrap_or_else(|poison| poison.into_inner()),
            )),
        }
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
    /// With batched strided runs (#78), `Auto` grants the items the part of
    /// that width the lanes leave unused, under the child's own `Auto` policy:
    /// a run is one item whose child spreads its batch axis itself (tcontract's
    /// packed driver claims batch entries on lanes), so one lane holding a long
    /// run would otherwise run it on one thread. The route then reports the
    /// granted item width; the child may use fewer threads. Without batched
    /// runs, `Auto` items stay serial as before.
    ///
    /// Every forced strategy goes through [`Exec::batch_decision`] on the
    /// original host budget, which clamps but never falls back.
    fn lane_decision(&self, exec: &Exec<'_>) -> Result<Widths> {
        if let BatchStrategy::Auto = exec.batch_policy().strategy {
            let serial_ns =
                2.0 * self.meta.work as f64 * ns_per_flop() * if T::IS_COMPLEX { 4.0 } else { 1.0 };
            let width = exec
                .width_for(serial_ns, GROUPED_SECTIONS)
                .min(exec.budget())
                .max(1);
            let lanes = width.min(self.meta.item_count()).max(1);
            if self.meta.runs.is_empty() {
                return Ok(Widths::new(lanes, 1));
            }
            return Ok(Widths {
                lanes,
                item_threads: (width / lanes).max(1),
                item_policy: Some(BatchPolicy::default()),
            });
        }
        let query = BatchQuery::new(self.meta.item_count(), self.meta.max_item_work as f64)
            .with_item_parallel(true);
        let decision = exec.batch_decision(&query);
        Ok(Widths::new(decision.lanes, decision.item_threads))
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
fn child_context<'b>(exec: &'b mut Exec<'_>, widths: Widths) -> Exec<'b> {
    let workspace = exec.workspace();
    if widths.item_threads <= 1 {
        return workspace.map_or(Exec::serial(), Exec::serial_with_workspace);
    }
    let budget = NonZeroUsize::new(widths.item_threads).unwrap_or(NonZeroUsize::MIN);
    Exec::from_token(exec.token().limit(budget), workspace, widths.item_policy())
}

/// The resolved widths of one grouped execution and the batch policy every
/// item's child context runs under.
#[derive(Clone, Copy, Debug)]
struct Widths {
    lanes: usize,
    item_threads: usize,
    /// `None` is the historical item policy, [`child_policy`], built only where
    /// a child context needs it (a default policy reads the host model).
    item_policy: Option<BatchPolicy>,
}

impl Widths {
    /// Widths with the historical item policy.
    fn new(lanes: usize, item_threads: usize) -> Self {
        Self {
            lanes,
            item_threads,
            item_policy: None,
        }
    }

    fn item_policy(&self) -> BatchPolicy {
        self.item_policy
            .unwrap_or_else(|| child_policy(self.item_threads))
    }
}

/// Where the item loops read their items from: the prepared jobs themselves
/// (the per-job route, read exactly as before #78) or the item list with
/// batched runs.
trait ItemSource: Sync {
    /// The original job id an error names.
    fn id(&self) -> usize;
    /// The item's output `(start, len)` in the shared output buffer.
    fn out_range(&self) -> (usize, usize);
    /// The child's view of the item: its windows of the shared inputs, the
    /// origins inside them, and its own disjoint output slice `out`.
    fn job_item<'p, 'd, T: Scalar, D>(
        &self,
        plans: &'p [Arc<Plan<T>>],
        lhs: &'p [T],
        rhs: &'p [T],
        out: &'d mut [D],
    ) -> JobItem<'p, 'd, T, D>;
}

impl ItemSource for PreparedJob {
    #[inline(always)]
    fn id(&self) -> usize {
        self.meta.id
    }

    #[inline(always)]
    fn out_range(&self) -> (usize, usize) {
        (self.meta.out_offset, self.meta.out_len)
    }

    #[inline(always)]
    fn job_item<'p, 'd, T: Scalar, D>(
        &self,
        plans: &'p [Arc<Plan<T>>],
        lhs: &'p [T],
        rhs: &'p [T],
        out: &'d mut [D],
    ) -> JobItem<'p, 'd, T, D> {
        let meta = &self.meta;
        JobItem {
            id: meta.id,
            plan: &plans[self.plan as usize],
            lhs: (&lhs[meta.lhs_offset..meta.lhs_offset + meta.lhs_len], 0),
            rhs: (&rhs[meta.rhs_offset..meta.rhs_offset + meta.rhs_len], 0),
            out: (out, 0),
        }
    }
}

impl ItemSource for Item {
    #[inline(always)]
    fn id(&self) -> usize {
        self.id
    }

    #[inline(always)]
    fn out_range(&self) -> (usize, usize) {
        (self.out.start, self.out.len)
    }

    #[inline(always)]
    fn job_item<'p, 'd, T: Scalar, D>(
        &self,
        plans: &'p [Arc<Plan<T>>],
        lhs: &'p [T],
        rhs: &'p [T],
        out: &'d mut [D],
    ) -> JobItem<'p, 'd, T, D> {
        JobItem {
            id: self.id,
            plan: &plans[self.plan as usize],
            lhs: (&lhs[self.lhs.start..self.lhs.end()], self.lhs.origin),
            rhs: (&rhs[self.rhs.start..self.rhs.end()], self.rhs.origin),
            out: (out, self.out.origin),
        }
    }
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

struct JobItem<'p, 'd, T: Scalar, D> {
    id: usize,
    plan: &'p Plan<T>,
    lhs: (&'p [T], usize),
    rhs: (&'p [T], usize),
    out: (&'d mut [D], usize),
}

/// A window origin as tcontract's slice origin. Origins lie inside windows of
/// buffers whose byte lengths `prepare_jobs` bounded by `isize::MAX`.
fn origin(offset: usize) -> isize {
    offset as isize
}

fn run_item<T: Scalar>(
    item: JobItem<'_, '_, T, T>,
    exec: &mut Exec<'_>,
    alpha: T,
    beta: T,
    source: SliceAccumulationSource<'_, T>,
) -> Result<()> {
    item.plan
        .execute_slices_accum(
            exec,
            alpha,
            (item.lhs.0, origin(item.lhs.1)),
            (item.rhs.0, origin(item.rhs.1)),
            beta,
            source,
            (item.out.0, origin(item.out.1)),
        )
        .map_err(|source| Error::Contract {
            step: item.id,
            source,
        })
}

fn run_item_uninit<'d, T: Scalar>(
    item: JobItem<'_, 'd, T, MaybeUninit<T>>,
    exec: &mut Exec<'_>,
    alpha: T,
    beta: T,
    source: SliceAccumulationSource<'_, T>,
) -> Result<&'d mut [T]> {
    item.plan
        .execute_uninit_slices(
            exec,
            alpha,
            (item.lhs.0, origin(item.lhs.1)),
            (item.rhs.0, origin(item.rhs.1)),
            beta,
            source,
            (item.out.0, origin(item.out.1)),
        )
        .map_err(|source| Error::Contract {
            step: item.id,
            source,
        })
}

fn scalar_kind(dtype: DType) -> ScalarKind {
    match dtype {
        DType::F32 => ScalarKind::F32,
        DType::F64 => ScalarKind::F64,
        DType::C32 => ScalarKind::C32,
        DType::C64 => ScalarKind::C64,
    }
}

/// The batched strided runs of `jobs`, in original order (#78).
///
/// The partition is [`GemmBatch::runs`] of the jobs as compact descriptors:
/// maximal runs, greedy from the left. A run is batched when it has at least
/// two jobs, a non-empty output, and its output span
/// `[first.out_offset, last.out_offset + rows * cols)` holds no other job's
/// output, so it can own one disjoint slice of the output buffer. `sorted` is
/// the prepared output order of the same jobs (non-empty outputs first, by
/// offset); it is derived from `jobs`, so the result is a function of `jobs`
/// and the element size alone.
fn batched_runs(
    jobs: &[GroupedGemmJob],
    scalar: ScalarKind,
    sorted: &[PreparedJob],
) -> Vec<GemmRun> {
    let ops = GemmOps {
        a: Trans::N,
        b: Trans::N,
    };
    let descriptors = jobs
        .iter()
        .map(|job| {
            let offsets = [job.lhs_offset, job.rhs_offset, job.out_offset];
            GemmJob::compact(ops, offsets, job.rows, job.cols, job.contracted)
        })
        .collect();
    // `prepare_jobs` already checked the same extents, byte spans and the
    // output overlap rule for compact jobs, so the descriptor accepts what it
    // accepted; were it ever to refuse, no run is batched and every job keeps
    // the per-job route.
    let Ok(batch) = GemmBatch::new(scalar, ops, descriptors) else {
        return Vec::new();
    };
    let nonempty = &sorted[..sorted.partition_point(|job| job.meta.out_len != 0)];
    let batched = |run: &&GemmRun| {
        let first = &jobs[run.first];
        if run.len < 2 || first.rows == 0 || first.cols == 0 {
            return false;
        }
        // Output strides are positive, so the run's outputs ascend from its
        // first job to its last; exclusive when exactly its own jobs start in
        // that range.
        let lo = first.out_offset;
        let hi = jobs[run.first + run.len - 1].out_offset;
        let inside = nonempty.partition_point(|job| job.meta.out_offset <= hi)
            - nonempty.partition_point(|job| job.meta.out_offset < lo);
        inside == run.len
    };
    let mut runs = Vec::with_capacity(batch.runs().iter().filter(batched).count());
    runs.extend(batch.runs().iter().filter(batched).copied());
    runs
}

/// Store `runs`, their distinct children and the execution items in
/// `prepared`.
///
/// Children are deduplicated by `(geometry, len, stride)` and ordered by that
/// key, so the same job set gives the same children in any job order; each
/// names the lowest first id among its runs. Items are in output order: a run
/// is one item at its first job's output, and its jobs are consecutive in that
/// order because its output span holds no other job's output.
fn plan_runs(
    prepared: &mut PreparedGroupedJobs,
    jobs: &[GroupedGemmJob],
    runs: Vec<GemmRun>,
) -> Result<()> {
    let nonempty = prepared.jobs.partition_point(|job| job.meta.out_len != 0);
    // A run's first job has a non-empty output, so it is found by offset.
    let geometry = |run: &GemmRun| {
        let offset = jobs[run.first].out_offset;
        let index = prepared.jobs[..nonempty]
            .binary_search_by_key(&offset, |job| job.meta.out_offset)
            .unwrap_or_else(|_| unreachable!("a batched run starts at a non-empty output"));
        prepared.jobs[index].plan
    };
    let key = |run: &GemmRun| (geometry(run), run.len, run.stride, run.first);
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_unstable_by_key(|&r| key(&runs[r]));
    let mut run_children: Vec<RunChild> = Vec::new();
    let mut child_of = vec![0u32; runs.len()];
    for &r in &order {
        let run = &runs[r];
        let (geometry, len, stride, id) = key(run);
        let same = run_children
            .last()
            .is_some_and(|c| (c.geometry, c.len, c.stride) == (geometry, len, stride));
        if !same {
            run_children.push(RunChild {
                geometry,
                len,
                stride,
                id,
            });
        }
        child_of[r] = child_index(id, prepared.children.len() + run_children.len() - 1)?;
    }

    // Merge the runs, by first output offset, into the output-ordered jobs.
    order.sort_unstable_by_key(|&r| jobs[runs[r].first].out_offset);
    let folded: usize = runs.iter().map(|run| run.len - 1).sum();
    let mut items = Vec::with_capacity(prepared.jobs.len() - folded);
    let mut max_item_work = prepared.max_job_work;
    let mut next = order.iter().peekable();
    let mut skip = 0usize;
    for job in &prepared.jobs {
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let starts_run = next.peek().is_some_and(|&&r| job.meta.id == runs[r].first);
        if !starts_run {
            items.push(Item::job(job));
            continue;
        }
        let Some(&r) = next.next() else {
            unreachable!("peeked above")
        };
        let run = &runs[r];
        let first = &jobs[run.first];
        let work = (first.rows as u128) * (first.contracted as u128) * (first.cols as u128);
        // INVARIANT: at most the total work, which `prepare_jobs` checked.
        max_item_work = max_item_work.max(work * run.len as u128);
        items.push(Item {
            id: run.first,
            plan: child_of[r],
            lhs: Window::run(first.lhs_offset, job.meta.lhs_len, run.stride[0], run.len),
            rhs: Window::run(first.rhs_offset, job.meta.rhs_len, run.stride[1], run.len),
            out: Window::run(first.out_offset, job.meta.out_len, run.stride[2], run.len),
        });
        skip = run.len - 1;
    }
    prepared.runs = runs;
    prepared.run_children = run_children;
    prepared.items = items;
    prepared.max_item_work = max_item_work;
    Ok(())
}

fn child_index(job: usize, index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| {
        grouped(GroupedError::Overflow {
            job,
            what: "child index",
        })
    })
}

fn spec(dims: &[usize], strides: &[isize], op: Op) -> tcontract::Result<OperandSpec> {
    Ok(OperandSpec::new(LayoutSpec::new(dims, strides, 0)?).with_op(op))
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
mod tests;
