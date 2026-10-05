//! N-ary einsum: a plan of binary steps over an intermediate arena.

use strided_view::{StridedView, StridedViewMut};
use tprims_contract::Plan;
use tprims_exec::Exec;

use crate::binary::plan_step;
use crate::error::{Error, Result, ShapeError};
use crate::layout::Layout;
use crate::scalar::Scalar;
use crate::spec::EinsumSpec;

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

    fn layout(&self) -> Layout<'_> {
        // INVARIANT: dims and strides are built with equal lengths.
        Layout::new(&self.dims, &self.strides).expect("temp layout lengths agree")
    }
}

#[derive(Debug)]
struct Step<T: Scalar> {
    a: Src,
    b: Src,
    dst: Dst,
    plan: Plan<T>,
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

/// A planned einsum: one tprims-contract plan per step of the given order,
/// and the placement of every intermediate in one scratch buffer.
///
/// Everything that depends only on the description and the operand layouts is
/// done here once; [`execute_into`](Self::execute_into) only builds views and
/// runs the steps. Intermediates are column-major and share one buffer: an
/// intermediate lives from the step that writes it to the step that reads it,
/// and storage is reused first fit once it is dead. A single input is
/// contracted with a rank-0 one, which covers diagonals, sums and
/// permutations of one operand.
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
pub struct EinsumPlan<T: Scalar> {
    inputs: Vec<Owned>,
    output: Owned,
    temps: Vec<Temp>,
    steps: Vec<Step<T>>,
    scratch_len: usize,
}

impl<T: Scalar> EinsumPlan<T> {
    /// Plan `spec` for inputs and an output of these layouts.
    ///
    /// # Errors
    ///
    /// [`ShapeError`] when the number of inputs differs from the description,
    /// an operand's rank differs from its number of labels, a label has two
    /// extents, or an intermediate's size overflows; [`Error::Contract`] when
    /// tprims-contract rejects a step (for example an output whose axes
    /// alias).
    pub fn new(spec: &EinsumSpec, inputs: &[Layout<'_>], output: Layout<'_>) -> Result<Self> {
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

        if n == 1 {
            // One input: contract with a rank-0 operand holding one.
            let plan = plan_step::<T>(
                0,
                inputs[0],
                &spec.inputs()[0],
                RANK0,
                &[],
                output,
                spec.output(),
            )?;
            steps.push(Step {
                a: Src::Input(0),
                b: Src::One,
                dst: Dst::Output,
                plan,
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
            let plan = plan_step::<T>(
                step,
                layout_of(sx),
                &operand_labels[x],
                layout_of(sy),
                &operand_labels[y],
                dst_layout,
                &result,
            )?;
            steps.push(Step {
                a: sx,
                b: sy,
                dst,
                plan,
            });
            adjust(&result, &mut live, true);
            operand_labels.push(result);
        }

        Ok(Self {
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

    /// Number of binary steps (one for a single input).
    pub fn steps(&self) -> usize {
        self.steps.len()
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
    /// [`Error::Contract`] when a lower layer fails a step.
    pub fn execute_into(
        &self,
        exec: &Exec<'_>,
        inputs: &[StridedView<'_, T>],
        out: &mut StridedViewMut<'_, T>,
        scratch: &mut Scratch<T>,
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
        let buf = scratch.reserve(self.scratch_len);
        let one = [T::ONE];

        for (k, step) in self.steps.iter().enumerate() {
            let wrap = |source| Error::Contract { step: k, source };
            match step.dst {
                Dst::Output => {
                    let a = self.view(step.a, inputs, buf, &one);
                    let b = self.view(step.b, inputs, buf, &one);
                    step.plan
                        .execute_into(exec, T::ONE, &a, &b, out)
                        .map_err(wrap)?;
                }
                Dst::Temp(t) => {
                    let dt = &self.temps[t];
                    let (left, rest) = buf.split_at_mut(dt.offset);
                    let (dbuf, right) = rest.split_at_mut(dt.len);
                    let (left, right) = (&*left, &*right);
                    // INVARIANT: the arena never places a step's result over
                    // an intermediate the step reads, so each source lies
                    // wholly in `left` or wholly in `right`.
                    let side = |s: Src| -> StridedView<'_, T> {
                        match s {
                            Src::Temp(u) => {
                                let st = &self.temps[u];
                                let data = if st.end() <= dt.offset {
                                    &left[st.offset..st.end()]
                                } else {
                                    &right[st.offset - dt.end()..st.end() - dt.end()]
                                };
                                temp_view(data, st)
                            }
                            Src::Input(i) => inputs[i].clone(),
                            Src::One => temp_view(&one, &RANK0_TEMP),
                        }
                    };
                    let a = side(step.a);
                    let b = side(step.b);
                    // INVARIANT: `dbuf` is exactly the column-major extent of `dt`.
                    let mut d = StridedViewMut::new(dbuf, &dt.dims, &dt.strides, 0)
                        .expect("intermediate view is in bounds");
                    step.plan
                        .execute_into(exec, T::ONE, &a, &b, &mut d)
                        .map_err(wrap)?;
                }
            }
        }
        Ok(())
    }

    /// The view of a source when the step writes the caller's output.
    fn view<'v>(
        &self,
        s: Src,
        inputs: &[StridedView<'v, T>],
        buf: &'v [T],
        one: &'v [T; 1],
    ) -> StridedView<'v, T> {
        match s {
            Src::Input(i) => inputs[i].clone(),
            Src::One => temp_view(one, &RANK0_TEMP),
            Src::Temp(t) => {
                let st = &self.temps[t];
                temp_view(&buf[st.offset..st.end()], st)
            }
        }
    }
}

const RANK0: Layout<'static> = Layout::rank0();

static RANK0_TEMP: Temp = Temp {
    offset: 0,
    len: 1,
    dims: Vec::new(),
    strides: Vec::new(),
};

fn temp_view<'v, T>(data: &'v [T], t: &Temp) -> StridedView<'v, T> {
    // INVARIANT: `data` is exactly the column-major extent of `t`.
    StridedView::new(data, &t.dims, &t.strides, 0).expect("intermediate view is in bounds")
}

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
