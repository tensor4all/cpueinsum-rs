//! N-ary einsum: a plan of binary steps over an intermediate arena.

use strided_view::{StridedView, StridedViewMut};
use tcontract::{OutputContract, Plan, PlanConfig};
use tcontract_exec::{Exec, Workspace, WorkspaceReq};

use crate::backend::{StepBackend, Tcontract};
use crate::batch::VendorLoopRoute;
use crate::binary::step_problem;
use crate::error::{Error, Result, ShapeError};
use crate::layout::Layout;
use crate::scalar::Scalar;
use crate::spec::{EinsumSpec, EinsumSpecRef};

/// Where a step reads an operand from.
#[derive(Clone, Copy, Debug)]
enum Src {
    /// The caller's input view with this index.
    Input(usize),
    /// The intermediate with this index.
    Temp(usize),
    /// The rank-0 one of a single-input plan.
    One,
}

/// Where a step writes its result.
#[derive(Clone, Copy, Debug)]
enum Dst {
    /// The caller's output view.
    Output,
    /// The intermediate with this index.
    Temp(usize),
}

/// A column-major intermediate at a fixed place in the scratch buffer.
#[derive(Clone, Debug)]
struct Temp {
    offset: usize,
    len: usize,
    dims: Vec<usize>,
    strides: Vec<isize>,
}

impl Temp {
    fn end(&self) -> usize {
        self.offset + self.len
    }

    fn range(&self) -> core::ops::Range<usize> {
        self.offset..self.end()
    }

    fn layout(&self) -> Layout<'_> {
        // INVARIANT: dims and strides are built with equal lengths.
        Layout::new(&self.dims, &self.strides).expect("temp layout lengths agree")
    }
}

/// Who runs a step.
///
/// The tcontract plan stays inline, as it was before backends: boxing it would add
/// a pointer chase to every step of the hot path.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum Kernel<T: Scalar, S> {
    /// tcontract.
    Tcontract(Plan<T>),
    /// The plan's backend, with the step's work space (an intermediate slot
    /// live only at this step) and, for a step the backend runs as a serial
    /// loop of per-item calls, the tcontract alternative that a forced lane
    /// strategy runs instead.
    Backend {
        step: S,
        work: Option<usize>,
        native: Option<Plan<T>>,
    },
}

#[derive(Debug)]
struct Step<T: Scalar, S> {
    a: Src,
    b: Src,
    dst: Dst,
    kernel: Kernel<T, S>,
}

/// An owned layout of a caller operand, checked again at execution.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Owned {
    dims: Vec<usize>,
    strides: Vec<isize>,
}

impl Owned {
    fn of(l: Layout<'_>) -> Self {
        Self {
            dims: l.dims().to_vec(),
            strides: l.strides().to_vec(),
        }
    }

    fn matches(&self, dims: &[usize], strides: &[isize]) -> bool {
        self.dims == dims && self.strides == strides
    }
}

/// A planned einsum: one plan per step of the given order (tcontract's,
/// or the [`StepBackend`]'s for a step it takes), and the placement of every
/// intermediate in one scratch buffer.
///
/// Everything that depends only on the description and the operand layouts is
/// done here once; [`execute_into`](Self::execute_into) only builds views and
/// runs the steps. Intermediates are column-major and share one buffer: an
/// intermediate lives from the step that writes it to the step that reads it,
/// and storage is reused first fit once it is dead. A single input is
/// contracted with a rank-0 one, which covers diagonals, sums and
/// permutations of one operand.
///
/// The backend `B` defaults to [`Tcontract`], which leaves every step to
/// tcontract; [`with_backend`](Self::with_backend) plans with another.
///
/// # Examples
///
/// ```
/// use cpueinsum::strided_view::{StridedView, StridedViewMut};
/// use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Layout, Scratch};
/// // d[i, l] = sum_{j,k} a[i, j] b[j, k] c[k, l], all 2x2 column-major.
/// let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [3, 2]]).unwrap();
/// let cm = Layout::new(&[2, 2], &[1, 2]).unwrap();
/// let plan = EinsumPlan::<f64>::new(&spec, &[cm, cm, cm], cm).unwrap();
/// assert_eq!(plan.scratch_len(), 4);
///
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let id = [1.0, 0.0, 0.0, 1.0];
/// let mut d = [0.0; 4];
/// let views = [
///     StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     StridedView::new(&id, &[2, 2], &[1, 2], 0).unwrap(),
///     StridedView::new(&id, &[2, 2], &[1, 2], 0).unwrap(),
/// ];
/// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
/// let mut scratch = Scratch::new();
/// plan.execute_into(&Exec::serial(), &views, &mut dv, &mut scratch).unwrap();
/// assert_eq!(d, a);
/// ```
#[derive(Debug)]
pub struct EinsumPlan<T: Scalar, B: StepBackend<T> = Tcontract> {
    backend: B,
    inputs: Vec<Owned>,
    output: Owned,
    temps: Vec<Temp>,
    steps: Vec<Step<T, B::Step>>,
    scratch_len: usize,
}

impl<T: Scalar> EinsumPlan<T> {
    /// Plan `spec` for inputs and an output of these layouts, every step on
    /// tcontract.
    ///
    /// # Errors
    ///
    /// As [`with_backend`](Self::with_backend).
    pub fn new(spec: &EinsumSpec, inputs: &[Layout<'_>], output: Layout<'_>) -> Result<Self> {
        Self::with_backend(Tcontract, spec, inputs, output)
    }
}

impl<T: Scalar, B: StepBackend<T>> EinsumPlan<T, B> {
    /// Plan `spec` for inputs and an output of these layouts, offering every
    /// step to `backend` first, with every tcontract step under the default
    /// [`PlanConfig`].
    ///
    /// # Errors
    ///
    /// As [`compile`](Self::compile).
    pub fn with_backend(
        backend: B,
        spec: &EinsumSpec,
        inputs: &[Layout<'_>],
        output: Layout<'_>,
    ) -> Result<Self> {
        Self::build(backend, spec, inputs, output, &PlanConfig::default())
    }

    /// Compile a plan from a borrowed [`EinsumSpecRef`], copying the
    /// description once, and plan every tcontract step under `config`.
    ///
    /// # Errors
    ///
    /// [`ShapeError`] when the number of inputs differs from the description,
    /// an operand's rank differs from its number of labels, a label has two
    /// extents, or an intermediate's size overflows; [`Error::Contract`] when
    /// tcontract rejects a step (for example an output whose axes
    /// alias).
    ///
    /// # Examples
    ///
    /// ```
    /// use cpueinsum::{EinsumPlan, EinsumSpecRef, Layout, Tcontract};
    /// use cpueinsum::tcontract::PlanConfig;
    /// let inputs: &[&[i64]] = &[&[0, 1], &[1, 2]];
    /// let spec = EinsumSpecRef { inputs, output: &[0, 2], path: &[[0, 1]] };
    /// let cm = Layout::new(&[2, 2], &[1, 2]).unwrap();
    /// let plan = EinsumPlan::<f64, Tcontract>::compile(
    ///     spec, &[cm, cm], cm, Tcontract, &PlanConfig::default()).unwrap();
    /// assert_eq!(plan.steps(), 1);
    /// ```
    pub fn compile(
        spec: EinsumSpecRef<'_>,
        inputs: &[Layout<'_>],
        output: Layout<'_>,
        backend: B,
        config: &PlanConfig,
    ) -> Result<Self> {
        let owned = EinsumSpec::new(spec.inputs, spec.output, spec.path)?;
        Self::build(backend, &owned, inputs, output, config)
    }

    /// Plan `spec` for inputs and an output of these layouts, offering every
    /// step to `backend` first.
    ///
    /// # Errors
    ///
    /// [`ShapeError`] when the number of inputs differs from the description,
    /// an operand's rank differs from its number of labels, a label has two
    /// extents, or an intermediate's size overflows; [`Error::Contract`] when
    /// tcontract rejects a step (for example an output whose axes
    /// alias).
    fn build(
        backend: B,
        spec: &EinsumSpec,
        inputs: &[Layout<'_>],
        output: Layout<'_>,
        config: &PlanConfig,
    ) -> Result<Self> {
        let n = spec.inputs().len();
        if inputs.len() != n {
            return Err(ShapeError::InputCount {
                expected: n,
                got: inputs.len(),
            }
            .into());
        }

        // Dense label ids and one extent per label.
        let mut labels: Vec<i64> = spec.inputs().iter().flatten().copied().collect();
        labels.sort_unstable();
        labels.dedup();
        let id = |l: i64| labels.binary_search(&l).expect("label collected above");
        let mut extent = vec![usize::MAX; labels.len()];
        let mut set_extents = |operand: usize, ls: &[i64], layout: Layout<'_>| -> Result<()> {
            if layout.dims().len() != ls.len() {
                return Err(ShapeError::Rank {
                    operand,
                    rank: layout.dims().len(),
                    labels: ls.len(),
                }
                .into());
            }
            for (&l, &e) in ls.iter().zip(layout.dims()) {
                let slot = &mut extent[id(l)];
                if *slot == usize::MAX {
                    *slot = e;
                } else if *slot != e {
                    return Err(ShapeError::Extent {
                        label: l,
                        first: *slot,
                        second: e,
                    }
                    .into());
                }
            }
            Ok(())
        };
        for (k, (ls, &layout)) in spec.inputs().iter().zip(inputs).enumerate() {
            set_extents(k, ls, layout)?;
        }
        set_extents(n, spec.output(), output)?;

        let mut in_output = vec![false; labels.len()];
        for &l in spec.output() {
            in_output[id(l)] = true;
        }

        // live[label] = number of live operands that carry the label.
        let mut live = vec![0usize; labels.len()];
        let mut operand_labels: Vec<Vec<i64>> = spec.inputs().to_vec();
        let mut mark = vec![false; labels.len()];
        let mut adjust = |ls: &[i64], live: &mut [usize], up: bool| {
            for &l in ls {
                let i = id(l);
                if !mark[i] {
                    mark[i] = true;
                    if up {
                        live[i] += 1;
                    } else {
                        live[i] -= 1;
                    }
                }
            }
            for &l in ls {
                mark[id(l)] = false;
            }
        };
        for ls in spec.inputs() {
            adjust(ls, &mut live, true);
        }

        let source = |operand: usize, temp_of: &[usize]| {
            if operand < n {
                Src::Input(operand)
            } else {
                Src::Temp(temp_of[operand - n])
            }
        };

        let mut temps: Vec<Temp> = Vec::new();
        // temp_of[k] = the intermediate written by step k.
        let mut temp_of: Vec<usize> = Vec::with_capacity(n.saturating_sub(1));
        // last_use[t] = the step that reads intermediate t.
        let mut last_use: Vec<usize> = Vec::new();
        let mut steps = Vec::with_capacity(n);
        let mut scratch_len = 0usize;
        let path = spec.path();

        // The kernel of one step; a work slot is placed beside everything
        // live at the step, the step's own result included.
        let kernel = |step: usize,
                      problem,
                      temps: &mut Vec<Temp>,
                      last_use: &mut Vec<usize>,
                      scratch_len: &mut usize|
         -> Result<Kernel<T, B::Step>> {
            let Some(s) = backend.plan(&problem) else {
                return Plan::<T>::from_problem(problem, config)
                    .map(Kernel::Tcontract)
                    .map_err(|source| Error::Contract { step, source });
            };
            // The batch policy is an executor property resolved at execution,
            // so a step the backend runs as a serial per-item loop keeps its
            // tcontract alternative ready: a forced lane strategy runs that
            // instead of the loop. Planning both costs one plan per such step
            // and keeps execution branch-only.
            let native = if B::vendor_loop(&s) {
                Some(
                    Plan::<T>::from_problem(problem, config)
                        .map_err(|source| Error::Contract { step, source })?,
                )
            } else {
                None
            };
            let len = B::work_len(&s);
            let work = (len > 0).then(|| {
                let offset = first_fit(temps, last_use, step, len);
                temps.push(Temp {
                    offset,
                    len,
                    dims: Vec::new(),
                    strides: Vec::new(),
                });
                last_use.push(step);
                *scratch_len = (*scratch_len).max(offset + len);
                temps.len() - 1
            });
            Ok(Kernel::Backend {
                step: s,
                work,
                native,
            })
        };

        if n == 1 {
            // One input: contract with a rank-0 operand holding one.
            let problem = step_problem::<T>(
                0,
                inputs[0],
                &spec.inputs()[0],
                RANK0,
                &[],
                output,
                spec.output(),
            )?;
            let kernel = kernel(0, problem, &mut temps, &mut last_use, &mut scratch_len)?;
            steps.push(Step {
                a: Src::Input(0),
                b: Src::One,
                dst: Dst::Output,
                kernel,
            });
        }

        for (step, &[x, y]) in path.iter().enumerate() {
            let last = step + 1 == path.len();
            adjust(&operand_labels[x], &mut live, false);
            adjust(&operand_labels[y], &mut live, false);
            let (sx, sy) = (source(x, &temp_of), source(y, &temp_of));
            for s in [sx, sy] {
                if let Src::Temp(t) = s {
                    last_use[t] = step;
                }
            }

            let (result, dst) = if last {
                (spec.output().to_vec(), Dst::Output)
            } else {
                let mut kept: Vec<i64> = Vec::new();
                for &l in operand_labels[x].iter().chain(&operand_labels[y]) {
                    let i = id(l);
                    if (in_output[i] || live[i] > 0) && !kept.contains(&l) {
                        kept.push(l);
                    }
                }
                let dims: Vec<usize> = kept.iter().map(|&l| extent[id(l)]).collect();
                let (strides, len) = column_major(&dims)?;
                let offset = first_fit(&temps, &last_use, step, len);
                let t = Temp {
                    offset,
                    len,
                    dims,
                    strides,
                };
                scratch_len = scratch_len.max(t.end());
                temps.push(t);
                last_use.push(usize::MAX);
                temp_of.push(temps.len() - 1);
                (kept, Dst::Temp(temps.len() - 1))
            };

            let layout_of = |s: Src| match s {
                Src::Input(k) => inputs[k],
                Src::Temp(t) => temps[t].layout(),
                Src::One => RANK0,
            };
            let dst_layout = match dst {
                Dst::Output => output,
                Dst::Temp(t) => temps[t].layout(),
            };
            let problem = step_problem::<T>(
                step,
                layout_of(sx),
                &operand_labels[x],
                layout_of(sy),
                &operand_labels[y],
                dst_layout,
                &result,
            )?;
            let kernel = kernel(step, problem, &mut temps, &mut last_use, &mut scratch_len)?;
            steps.push(Step {
                a: sx,
                b: sy,
                dst,
                kernel,
            });
            adjust(&result, &mut live, true);
            operand_labels.push(result);
        }

        Ok(Self {
            backend,
            inputs: inputs.iter().map(|&l| Owned::of(l)).collect(),
            output: Owned::of(output),
            temps,
            steps,
            scratch_len,
        })
    }

    /// Elements of scratch the plan needs for its intermediates.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Elements of the caller-owned intermediate arena the plan needs.
    ///
    /// This is the same element count as [`scratch_len`](Self::scratch_len): a
    /// plan keeps every intermediate and every [`StepBackend`](crate::StepBackend)
    /// work slot in one element-disjoint arena. It is deliberately **not** a
    /// scalar for the whole workspace, because the native tcontract steps need
    /// a separate, component-wise
    /// [`WorkspaceReq`](tcontract_exec::WorkspaceReq); see
    /// [`native_workspace_req`](Self::native_workspace_req).
    ///
    /// A caller that executes a plan many times sizes a [`Scratch`] once with
    /// `Scratch::with_len(plan.workspace_len())` and reuses it, so the steady
    /// state allocates nothing for the arena.
    pub fn workspace_len(&self) -> usize {
        self.scratch_len
    }

    /// The component-wise workspace the native tcontract steps need on `exec`
    /// for an `output` storage contract.
    ///
    /// Each native step's [`Plan::workspace_req`](tcontract::Plan::workspace_req)
    /// is folded field by field with a maximum: the steps run one after another,
    /// so their arenas are reused, and a large A block of one step must not stand
    /// in for a missing B panel of another. A backend step runs from the element
    /// arena and contributes nothing here, except the tcontract alternative a
    /// forced lane strategy runs under `exec`'s policy, which contributes what a
    /// native step of the same problem would. The result is a conservative bound:
    /// prepare a [`Workspace`](tcontract_exec::Workspace) with it and warm every
    /// worker the executor may use (`Workspace::prepare_for`) before executing
    /// through [`Exec::serial_with_workspace`] or a pool, or tcontract grows the
    /// arena on first use.
    pub fn native_workspace_req(&self, exec: &Exec<'_>, output: OutputContract) -> WorkspaceReq {
        let mut req = WorkspaceReq::default();
        let forced_lanes = matches!(
            VendorLoopRoute::resolve(exec.batch_policy().strategy),
            VendorLoopRoute::Native
        );
        for step in &self.steps {
            let native = match &step.kernel {
                Kernel::Tcontract(plan) => Some(plan),
                Kernel::Backend { native, .. } if forced_lanes => native.as_ref(),
                Kernel::Backend { .. } => None,
            };
            if let Some(plan) = native {
                let r = plan.workspace_req(exec, output);
                req.a_bytes = req.a_bytes.max(r.a_bytes);
                req.tile_bytes = req.tile_bytes.max(r.tile_bytes);
                req.worker_scatter = req.worker_scatter.max(r.worker_scatter);
                req.b_bytes = req.b_bytes.max(r.b_bytes);
                req.team_scatter = req.team_scatter.max(r.team_scatter);
                req.barriers = req.barriers.max(r.barriers);
            }
        }
        req
    }

    /// Number of binary steps (one for a single input).
    pub fn steps(&self) -> usize {
        self.steps.len()
    }

    /// Number of steps the backend took (the rest run on tcontract).
    ///
    /// This is the planning-time split. A step the backend runs as a serial loop
    /// of per-item calls reports that through
    /// [`StepBackend::vendor_loop`](crate::StepBackend::vendor_loop), so the
    /// executor's batch policy may still run its tcontract alternative: a forced
    /// lane strategy does, without changing this count.
    pub fn backend_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(s.kernel, Kernel::Backend { .. }))
            .count()
    }

    /// The backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Run the plan: the overwrite form, reading no previous value of `out`.
    ///
    /// `scratch` grows to [`scratch_len`](Self::scratch_len) when it is
    /// shorter, and is otherwise reused as is; its contents are never read
    /// before they are written.
    ///
    /// # Errors
    ///
    /// [`ShapeError::InputCount`] or [`ShapeError::Mismatch`] when the views
    /// differ from the layouts the plan was built for (nothing is written);
    /// [`Error::Contract`] or [`Error::Backend`] when a lower layer fails a
    /// step. Every step's batch-policy admission is resolved after the shape
    /// check and before the scratch buffer grows or any step runs: a forced lane
    /// strategy a native route cannot honour is refused as [`Error::Contract`]
    /// there, so no intermediate, output or scratch element is written first.
    pub fn execute_into(
        &self,
        exec: &Exec<'_>,
        inputs: &[StridedView<'_, T>],
        out: &mut StridedViewMut<'_, T>,
        scratch: &mut Scratch<T>,
    ) -> Result<()> {
        self.check_shapes(inputs, out)?;
        self.admit(exec)?;
        let buf = scratch.reserve(self.scratch_len);
        self.run_steps(exec, inputs, out, buf)
    }

    /// A bounded, allocation-free replay from a caller-owned workspace and arena.
    ///
    /// `work` holds the plan's intermediates and backend work (exactly
    /// [`workspace_len`](Self::workspace_len) elements) and `workspace` is sized
    /// and warmed for the plan's native steps (see
    /// [`native_workspace_req`](Self::native_workspace_req)). Both are checked
    /// before the first write: a short arena returns [`Error::Workspace`] and a
    /// native workspace that does not cover the requirement returns
    /// [`Error::Contract`] carrying tcontract's `Workspace` shortage, so nothing
    /// is written and the arena is never grown. This is the serial bounded entry;
    /// a caller with a pool lends its own workspace through [`Exec`].
    ///
    /// # Errors
    ///
    /// [`ShapeError`] for views that do not match the plan, [`Error::Contract`]
    /// for a refused batch policy or a native workspace that does not cover the
    /// requirement, [`Error::Workspace`] for a short arena, and
    /// [`Error::Backend`] for a backend step. The batch-policy admission runs
    /// right after the shape check (see
    /// [`execute_into`](Self::execute_into)), so a refusal precedes the arena and
    /// workspace checks and nothing is written.
    pub fn execute_bounded(
        &self,
        workspace: &Workspace,
        inputs: &[StridedView<'_, T>],
        out: &mut StridedViewMut<'_, T>,
        work: &mut [T],
    ) -> Result<()> {
        self.check_shapes(inputs, out)?;
        let exec = Exec::serial_with_workspace(workspace.as_provider());
        self.admit(&exec)?;
        if work.len() < self.scratch_len {
            return Err(Error::Workspace {
                needed: self.scratch_len,
                have: work.len(),
            });
        }
        let required = self.native_workspace_req(&exec, OutputContract::Initialized);
        if !required.covered_by(workspace.req()) || workspace.slots() < 1 {
            return Err(Error::Contract {
                step: 0,
                source: tcontract::Error::Workspace {
                    needed: required.bytes(),
                    have: workspace.have(),
                },
            });
        }
        self.run_steps(&exec, inputs, out, &mut work[..self.scratch_len])
    }

    fn check_shapes(
        &self,
        inputs: &[StridedView<'_, T>],
        out: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        if inputs.len() != self.inputs.len() {
            return Err(ShapeError::InputCount {
                expected: self.inputs.len(),
                got: inputs.len(),
            }
            .into());
        }
        for (k, (v, l)) in inputs.iter().zip(&self.inputs).enumerate() {
            if !l.matches(v.dims(), v.strides()) {
                return Err(ShapeError::Mismatch { operand: k }.into());
            }
        }
        if !self.output.matches(out.dims(), out.strides()) {
            return Err(ShapeError::Mismatch {
                operand: self.inputs.len(),
            }
            .into());
        }
        Ok(())
    }

    fn run_steps(
        &self,
        exec: &Exec<'_>,
        inputs: &[StridedView<'_, T>],
        out: &mut StridedViewMut<'_, T>,
        buf: &mut [T],
    ) -> Result<()> {
        let one = [T::ONE];

        // Intermediates are passed as slices of the scratch buffer, not views:
        // a view allocates its extents and strides on every construction, a
        // cost of the order of a small step itself.
        for (k, step) in self.steps.iter().enumerate() {
            let range = |t: usize| self.temps[t].range();
            let dst = match step.dst {
                Dst::Temp(t) => Some(range(t)),
                Dst::Output => None,
            };
            let work = match step.kernel {
                Kernel::Backend { work: Some(w), .. } => Some(range(w)),
                _ => None,
            };
            let (dbuf, wbuf, rest) = carve(buf, dst, work);
            // INVARIANT: the arena never places a step's result or work over
            // an intermediate the step reads, so each source lies wholly in
            // one read-only part of the scratch.
            let side = |s: Src| -> (&[T], isize) {
                match s {
                    Src::Input(i) => (inputs[i].data(), inputs[i].offset()),
                    Src::One => (&one, 0),
                    Src::Temp(t) => (rest.get(range(t)), 0),
                }
            };
            let (a, b) = (side(step.a), side(step.b));
            let d = match dbuf {
                Some(dbuf) => (dbuf, 0),
                None => {
                    let origin = out.offset();
                    (out.data_mut(), origin)
                }
            };
            match &step.kernel {
                Kernel::Tcontract(plan) => plan
                    .execute_slices(exec, T::ONE, a, b, d)
                    .map_err(|source| Error::Contract { step: k, source })?,
                Kernel::Backend {
                    step: s, native, ..
                } => match self.backend_route(k, s, exec) {
                    VendorLoopRoute::Loop => self
                        .backend
                        .execute(s, exec, a, b, d, wbuf.unwrap_or_default())
                        .map_err(|source| Error::Backend { step: k, source })?,
                    VendorLoopRoute::Native => native
                        .as_ref()
                        // INVARIANT: `native` is planned exactly for the steps
                        // `vendor_loop` reports, which is also the only way the
                        // route resolves to `Native`.
                        .expect("a vendor-loop step plans its native alternative")
                        .execute_slices(exec, T::ONE, a, b, d)
                        .map_err(|source| Error::Contract { step: k, source })?,
                },
            }
        }
        Ok(())
    }

    /// The route of the backend step `step`, resolved before any write.
    ///
    /// A step that is a serial loop of per-item calls is subject to the shared
    /// [`VendorLoopRoute`] rule, so a forced lane strategy runs its tcontract
    /// alternative. A step that resolves the batch policy itself runs on the
    /// backend under every policy.
    fn backend_route(&self, _k: usize, step: &B::Step, exec: &Exec<'_>) -> VendorLoopRoute {
        if !B::vendor_loop(step) {
            return VendorLoopRoute::Loop;
        }
        VendorLoopRoute::resolve(exec.batch_policy().strategy)
    }

    /// Resolve every step's batch-policy admission under `exec`, writing nothing.
    ///
    /// A plan resolves its whole route before the first step, so a refusal can
    /// never follow a partially written intermediate or output:
    ///
    /// * a backend step that is a serial loop of per-item calls yields to the
    ///   shared [`VendorLoopRoute`] rule, which sends a forced lane strategy to
    ///   the native alternative;
    /// * every native step, and every backend step the policy routes to its
    ///   tcontract alternative, resolves
    ///   [`Plan::execution_route`](tcontract::Plan::execution_route) for the same
    ///   initialized contract and overwrite scale execution uses, which refuses a
    ///   forced lane strategy the native route cannot honour (tcontract's packed
    ///   driver rejects lanes *and* item threads) before any write.
    ///
    /// A backend step that resolves the batch policy itself (the default of
    /// [`StepBackend::vendor_loop`](crate::StepBackend::vendor_loop)) needs no
    /// native admission and keeps running.
    fn admit(&self, exec: &Exec<'_>) -> Result<()> {
        for (k, step) in self.steps.iter().enumerate() {
            let native = match &step.kernel {
                Kernel::Tcontract(plan) => plan,
                Kernel::Backend {
                    step: s, native, ..
                } => match self.backend_route(k, s, exec) {
                    VendorLoopRoute::Loop => continue,
                    // INVARIANT: `native` is planned exactly for the steps
                    // `vendor_loop` reports, the only way to route to `Native`.
                    VendorLoopRoute::Native => native
                        .as_ref()
                        .expect("a vendor-loop step plans its native alternative"),
                },
            };
            native
                .execution_route(exec, T::ONE, OutputContract::Initialized)
                .map_err(|source| Error::Contract { step: k, source })?;
        }
        Ok(())
    }
}

const RANK0: Layout<'static> = Layout::rank0();

/// Column-major strides and the element count of `dims`.
fn column_major(dims: &[usize]) -> Result<(Vec<isize>, usize)> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut len = 1usize;
    for &e in dims {
        strides.push(isize::try_from(len).map_err(|_| ShapeError::Overflow)?);
        len = len.checked_mul(e).ok_or(ShapeError::Overflow)?;
    }
    isize::try_from(len).map_err(|_| ShapeError::Overflow)?;
    Ok((strides, len))
}

/// The read-only parts of a buffer around at most two carved-out ranges, each
/// with its start.
struct Rest<'b, T> {
    parts: [(usize, &'b [T]); 3],
}

impl<'b, T> Rest<'b, T> {
    /// The elements of `r`, which lies wholly in one part (or is empty).
    fn get(&self, r: core::ops::Range<usize>) -> &'b [T] {
        if r.is_empty() {
            return &[];
        }
        for &(start, part) in &self.parts {
            if r.start >= start && r.end <= start + part.len() {
                return &part[r.start - start..r.end - start];
            }
        }
        unreachable!("an operand overlaps a step's result or work space")
    }
}

/// Split `buf` into the mutable ranges `x` and `y` (disjoint) and the
/// read-only rest.
#[allow(clippy::type_complexity)]
fn carve<T>(
    buf: &mut [T],
    x: Option<core::ops::Range<usize>>,
    y: Option<core::ops::Range<usize>>,
) -> (Option<&mut [T]>, Option<&mut [T]>, Rest<'_, T>) {
    let empty = (0, &[][..]);
    // Order the two ranges; `swap` records whether y comes first.
    let (lo, hi, swap) = match (x, y) {
        (Some(x), Some(y)) if y.start < x.start => (Some(y), Some(x), true),
        (Some(x), y) => (Some(x), y, false),
        (None, Some(y)) => (Some(y), None, true),
        (None, None) => (None, None, false),
    };
    let Some(lo) = lo else {
        return (
            None,
            None,
            Rest {
                parts: [(0, buf), empty, empty],
            },
        );
    };
    let (p0, tail) = buf.split_at_mut(lo.start);
    let (m0, tail) = tail.split_at_mut(lo.len());
    let (p1, m1, p2, p2_start) = match hi {
        Some(hi) => {
            let (p1, tail) = tail.split_at_mut(hi.start - lo.end);
            let (m1, p2) = tail.split_at_mut(hi.len());
            (p1, Some(m1), p2, hi.end)
        }
        None => (tail, None, &mut [][..], lo.end),
    };
    let rest = Rest {
        parts: [(0, &*p0), (lo.end, &*p1), (p2_start, &*p2)],
    };
    if swap {
        match m1 {
            Some(m1) => (Some(m1), Some(m0), rest),
            None => (None, Some(m0), rest),
        }
    } else {
        (Some(m0), m1, rest)
    }
}

/// The lowest offset where `len` elements fit beside every intermediate still
/// live at `step` (read at `step` or later).
fn first_fit(temps: &[Temp], last_use: &[usize], step: usize, len: usize) -> usize {
    let mut busy: Vec<(usize, usize)> = temps
        .iter()
        .zip(last_use)
        .filter(|&(t, &u)| u >= step && t.len > 0)
        .map(|(t, _)| (t.offset, t.end()))
        .collect();
    busy.sort_unstable();
    let mut at = 0;
    for (start, end) in busy {
        if start >= at + len {
            break;
        }
        at = at.max(end);
    }
    at
}

/// Reusable storage for the intermediates of an [`EinsumPlan`].
///
/// One `Scratch` can serve plans of different sizes; it only grows.
///
/// # Examples
///
/// ```
/// let s = cpueinsum::Scratch::<f64>::new();
/// assert_eq!(s.len(), 0);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Scratch<T> {
    buf: Vec<T>,
}

impl<T: Scalar> Scratch<T> {
    /// Empty scratch.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Scratch of `len` elements.
    pub fn with_len(len: usize) -> Self {
        Self {
            buf: vec![T::default(); len],
        }
    }

    /// Elements held.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether no element is held.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn reserve(&mut self, len: usize) -> &mut [T] {
        if self.buf.len() < len {
            self.buf.resize(len, T::default());
        }
        &mut self.buf[..len]
    }
}

/// Plan and run an einsum in one call, with fresh scratch.
///
/// To execute the same shapes many times, build an [`EinsumPlan`] once and
/// keep a [`Scratch`].
///
/// # Errors
///
/// As [`EinsumPlan::new`] and [`EinsumPlan::execute_into`].
///
/// # Examples
///
/// ```
/// use cpueinsum::strided_view::{StridedView, StridedViewMut};
/// use cpueinsum::{einsum_into, EinsumSpec, Exec};
/// // The trace of a 2x2 matrix.
/// let spec = EinsumSpec::new(&[&[0, 0]], &[], &[]).unwrap();
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let mut d = [0.0];
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let mut dv = StridedViewMut::new(&mut d, &[], &[], 0).unwrap();
/// einsum_into(&Exec::serial(), &spec, &[av], &mut dv).unwrap();
/// assert_eq!(d, [5.0]);
/// ```
pub fn einsum_into<T: Scalar>(
    exec: &Exec<'_>,
    spec: &EinsumSpec,
    inputs: &[StridedView<'_, T>],
    out: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let layouts: Vec<Layout<'_>> = inputs.iter().map(Layout::of).collect();
    let plan = EinsumPlan::<T>::new(spec, &layouts, Layout::of_mut(out))?;
    let mut scratch = Scratch::with_len(plan.scratch_len());
    plan.execute_into(exec, inputs, out, &mut scratch)
}
