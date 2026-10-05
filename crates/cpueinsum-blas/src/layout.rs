//! Planning one step: the GEMM, and what is packed into work space.
//!
//! The index groups M (A, D), N (B, D) and K (A, B) are each put in one order
//! and fused per operand as tenferro's CPU `dot_general` does (via
//! tprims-contract's faer strategy, which ported it): an operand whose groups
//! fuse and whose fused matrix has a unit stride on one axis with a valid
//! leading dimension is handed to BLAS in place. Any other operand is packed
//! into column-major work space first (and the output copied back after).
//! Batch axes are looped over, never fused.

use cpueinsum::tprims_contract::api::{is_injective_layout, OperandId, Problem, RoleAxis};

use crate::ffi::{BlasScalar, Gemm, Trans};

/// A copy between an operand and its packed form in work space.
///
/// Axes are in the packed (column-major) order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pack {
    /// Start in the step's work space.
    pub(crate) offset: usize,
    /// Elements.
    pub(crate) len: usize,
    pub(crate) dims: Vec<usize>,
    /// Strides in the caller's operand.
    pub(crate) outer: Vec<isize>,
    /// Strides in work space (column-major).
    pub(crate) inner: Vec<isize>,
}

/// Where the GEMM finds one of its three matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Side {
    /// The step operand it comes from (A, B or D).
    pub(crate) from: OperandId,
    /// `None`: in place in the caller's memory.
    pub(crate) pack: Option<Pack>,
    /// Stride of each batch axis where the GEMM reads it.
    pub(crate) h: Vec<isize>,
}

/// A step taken by [`Blas`](crate::Blas): one GEMM per batch item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlasStep {
    pub(crate) gemm: Gemm,
    pub(crate) left: Side,
    pub(crate) right: Side,
    pub(crate) out: Side,
    /// Batch extents (extent-one axes dropped).
    pub(crate) h: Vec<usize>,
    /// Inclusive element spans of A, B and D relative to their origins.
    pub(crate) spans: [(i128, i128); 3],
    pub(crate) work_len: usize,
}

impl BlasStep {
    /// Whether an operand is packed: `[A, B, D]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate blas_src as _;
    /// use cpueinsum::{EinsumPlan, EinsumSpec, Layout};
    /// use cpueinsum_blas::Blas;
    /// let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    /// let cm = Layout::new(&[64, 64], &[1, 64]).unwrap();
    /// let plan = EinsumPlan::<f64, Blas>::with_backend(Blas::default(), &spec, &[cm, cm], cm).unwrap();
    /// assert_eq!(plan.backend_steps(), 1);
    /// ```
    pub fn packed(&self) -> [bool; 3] {
        let mut out = [false; 3];
        for s in [&self.left, &self.right, &self.out] {
            let i = match s.from {
                OperandId::A => 0,
                OperandId::B => 1,
                _ => 2,
            };
            out[i] = s.pack.is_some();
        }
        out
    }

    /// Multiply-accumulates of one GEMM of the batch.
    pub fn gemm_macs(&self) -> u64 {
        let g = &self.gemm;
        [g.m, g.n, g.k].iter().map(|&x| x as u64).product()
    }
}

/// Whether `order` fuses the group in operand `o`: the strides of the
/// non-unit axes must chain. Returns the fused stride (one for an empty
/// group).
fn fuse(group: &[RoleAxis], order: &[usize], o: OperandId) -> Option<isize> {
    let mut first: Option<isize> = None;
    let mut expect = 0isize;
    for &e in order {
        let d = group[e].extent();
        if d == 1 {
            continue;
        }
        let s = group[e].stride(o);
        match first {
            None => first = Some(s),
            Some(_) if s != expect => return None,
            Some(_) => {}
        }
        expect = s.checked_mul(isize::try_from(d).ok()?)?;
    }
    Some(first.unwrap_or(1))
}

/// The order of a group: the first carrier's stride order that fuses in every
/// carrier, else one that fuses in the first carrier, else the first.
fn choose(group: &[RoleAxis], carriers: &[OperandId]) -> Vec<usize> {
    let sorted = |o: OperandId| {
        let mut v: Vec<usize> = (0..group.len()).collect();
        v.sort_by_key(|&e| group[e].stride(o).unsigned_abs());
        v
    };
    let candidates: Vec<Vec<usize>> = carriers.iter().map(|&o| sorted(o)).collect();
    let fits_all = |ord: &[usize]| carriers.iter().all(|&o| fuse(group, ord, o).is_some());
    if let Some(c) = candidates.iter().find(|c| fits_all(c)) {
        return c.clone();
    }
    if let Some(c) = candidates
        .iter()
        .find(|c| fuse(group, c, carriers[0]).is_some())
    {
        return c.clone();
    }
    candidates.into_iter().next().unwrap_or_default()
}

fn extent(group: &[RoleAxis]) -> usize {
    group.iter().map(RoleAxis::extent).product()
}

/// The leading dimension of a `rows x cols` matrix read as stored
/// (column-major), if it is one BLAS accepts.
pub(crate) fn no_trans(rows: usize, cols: usize, rs: isize, cs: isize) -> Option<i32> {
    if rows > 1 && rs != 1 {
        return None;
    }
    let min = rows.max(1);
    // A single column has no stride between columns; any valid one serves.
    let ld = if cols <= 1 {
        min
    } else {
        usize::try_from(cs).ok()?
    };
    (ld >= min).then_some(())?;
    i32::try_from(ld).ok()
}

/// How BLAS reads a `rows x cols` input with strides `(rs, cs)`, and its
/// leading dimension; conjugation needs the transposed reading.
pub(crate) fn orient(
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
    conj: bool,
) -> Option<(Trans, i32)> {
    let t = no_trans(cols, rows, cs, rs).map(|ld| (if conj { Trans::C } else { Trans::T }, ld));
    if conj {
        return t;
    }
    no_trans(rows, cols, rs, cs).map(|ld| (Trans::No, ld)).or(t)
}

/// One group in a chosen order.
struct Group<'p> {
    axes: &'p [RoleAxis],
    order: Vec<usize>,
    extent: usize,
}

impl Group<'_> {
    fn new<'p>(axes: &'p [RoleAxis], carriers: &[OperandId]) -> Group<'p> {
        Group {
            axes,
            order: choose(axes, carriers),
            extent: extent(axes),
        }
    }

    fn stride(&self, o: OperandId) -> Option<isize> {
        fuse(self.axes, &self.order, o)
    }

    /// `(extent, stride in o)` of each axis, in order.
    fn axes(&self, o: OperandId) -> impl Iterator<Item = (usize, isize)> + '_ {
        self.order
            .iter()
            .map(move |&e| (self.axes[e].extent(), self.axes[e].stride(o)))
    }
}

/// Lays out the work space as packs are added.
struct Work {
    len: usize,
}

impl Work {
    /// A column-major pack of `axes` (`(extent, stride in the operand)`, the
    /// matrix's axes first), and the strides of the batch axes in it.
    fn pack(&mut self, axes: Vec<(usize, isize)>, h: &[(usize, isize)]) -> (Pack, Vec<isize>) {
        let mut dims = Vec::with_capacity(axes.len() + h.len());
        let mut outer = Vec::with_capacity(dims.capacity());
        let mut inner = Vec::with_capacity(dims.capacity());
        let mut hs = Vec::with_capacity(h.len());
        let mut s = 1usize;
        for (i, &(e, o)) in axes.iter().chain(h).enumerate() {
            if i >= axes.len() {
                hs.push(s as isize);
            }
            dims.push(e);
            outer.push(o);
            inner.push(s as isize);
            s *= e;
        }
        let pack = Pack {
            offset: self.len,
            len: s,
            dims,
            outer,
            inner,
        };
        self.len += s;
        (pack, hs)
    }
}

/// A GEMM input: `rows x cols` from groups of operand `o`.
#[allow(clippy::too_many_arguments)]
fn input(
    o: OperandId,
    rows: &Group<'_>,
    cols: &Group<'_>,
    conj: bool,
    h: &[RoleAxis],
    work: &mut Work,
) -> Option<(Side, Trans, i32)> {
    let direct = rows
        .stride(o)
        .zip(cols.stride(o))
        .and_then(|(rs, cs)| orient(rows.extent, cols.extent, rs, cs, conj));
    if let Some((t, ld)) = direct {
        let side = Side {
            from: o,
            pack: None,
            h: h.iter().map(|x| x.stride(o)).collect(),
        };
        return Some((side, t, ld));
    }
    let hax: Vec<_> = h.iter().map(|x| (x.extent(), x.stride(o))).collect();
    // Packed column-major, or transposed so that conjugation is a ConjTrans.
    let (axes, t, ld): (Vec<_>, _, _) = if conj {
        (
            cols.axes(o).chain(rows.axes(o)).collect(),
            Trans::C,
            cols.extent.max(1),
        )
    } else {
        (
            rows.axes(o).chain(cols.axes(o)).collect(),
            Trans::No,
            rows.extent.max(1),
        )
    };
    let (pack, hs) = work.pack(axes, &hax);
    let side = Side {
        from: o,
        pack: Some(pack),
        h: hs,
    };
    Some((side, t, i32::try_from(ld).ok()?))
}

/// Plan `p` as batched GEMMs, or `None` when BLAS should not take it.
pub(crate) fn plan<T: BlasScalar>(p: &Problem) -> Option<BlasStep> {
    use OperandId::{A, B, D};
    let r = p.roles();
    if p.k_empty() || p.out_empty() || p.all_batch() || !p.c_matches_d() {
        return None;
    }
    if !matches!(p.c_spec(), cpueinsum::tprims_contract::api::CSpec::Absent) {
        return None;
    }
    // A reduction over an axis one input lacks is not a matrix product.
    if r.k().iter().any(|x| !(x.in_a() && x.in_b())) {
        return None;
    }
    let dl = p.d().layout();
    if !is_injective_layout(dl.dims(), dl.strides()) {
        return None;
    }
    let span = |o| p.span(o).map(|s| (s.lo(), s.hi()));
    let spans = [span(A)?, span(B)?, span(D)?];

    let c = |conj: bool| T::COMPLEX && conj;
    let conj_d = c(p.d().op().is_conj());
    // conj(A B) = conj(A) conj(B).
    let ca = c(p.a().op().is_conj()) != conj_d;
    let cb = c(p.b().op().is_conj()) != conj_d;

    let m = Group::new(r.m(), &[D, A]);
    let n = Group::new(r.n(), &[D, B]);
    let k = Group::new(r.k(), &[A, B]);
    let h: Vec<RoleAxis> = r.h().iter().copied().filter(|x| x.extent() != 1).collect();
    let dim = |e: usize| i32::try_from(e).ok();
    let (mi, ni, ki) = (dim(m.extent)?, dim(n.extent)?, dim(k.extent)?);

    let mut work = Work { len: 0 };
    // The output: as stored, or transposed (D^T = B^T A^T), or packed.
    let dm = m.stride(D);
    let dn = n.stride(D);
    let direct = |rows: &Group<'_>, cols: &Group<'_>, rs: Option<isize>, cs: Option<isize>| {
        no_trans(rows.extent, cols.extent, rs?, cs?)
    };
    let d_h = || h.iter().map(|x| x.stride(D)).collect::<Vec<_>>();
    let (swap, out, ldc) = if let Some(ld) = direct(&m, &n, dm, dn) {
        let out = Side {
            from: D,
            pack: None,
            h: d_h(),
        };
        (false, out, ld)
    } else if let Some(ld) = direct(&n, &m, dn, dm) {
        let out = Side {
            from: D,
            pack: None,
            h: d_h(),
        };
        (true, out, ld)
    } else {
        let hax: Vec<_> = h.iter().map(|x| (x.extent(), x.stride(D))).collect();
        let (pack, hs) = work.pack(m.axes(D).chain(n.axes(D)).collect(), &hax);
        let out = Side {
            from: D,
            pack: Some(pack),
            h: hs,
        };
        (false, out, dim(m.extent.max(1))?)
    };

    let (left, ta, lda, right, tb, ldb, gm, gn) = if swap {
        // D^T (N x M) = B^T (N x K) A^T (K x M).
        let (l, ta, lda) = input(B, &n, &k, cb, &h, &mut work)?;
        let (r, tb, ldb) = input(A, &k, &m, ca, &h, &mut work)?;
        (l, ta, lda, r, tb, ldb, ni, mi)
    } else {
        let (l, ta, lda) = input(A, &m, &k, ca, &h, &mut work)?;
        let (r, tb, ldb) = input(B, &k, &n, cb, &h, &mut work)?;
        (l, ta, lda, r, tb, ldb, mi, ni)
    };
    let gemm = Gemm {
        ta,
        tb,
        m: gm,
        n: gn,
        k: ki,
        lda,
        ldb,
        ldc,
    };
    Some(BlasStep {
        gemm,
        left,
        right,
        out,
        h: h.iter().map(RoleAxis::extent).collect(),
        spans,
        work_len: work.len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_trans_layouts() {
        // Column-major 3x4.
        assert_eq!(no_trans(3, 4, 1, 3), Some(3));
        // Padded leading dimension.
        assert_eq!(no_trans(3, 4, 1, 5), Some(5));
        // Leading dimension below the row count.
        assert_eq!(no_trans(3, 4, 1, 2), None);
        // Row-major is not column-major.
        assert_eq!(no_trans(3, 4, 4, 1), None);
        // A single row or column ignores the irrelevant stride.
        assert_eq!(no_trans(1, 4, 7, 1), Some(1));
        assert_eq!(no_trans(3, 1, 1, 0), Some(3));
        // Negative leading dimension.
        assert_eq!(no_trans(3, 4, 1, -3), None);
    }

    #[test]
    fn orientation() {
        assert_eq!(orient(3, 4, 1, 3, false), Some((Trans::No, 3)));
        assert_eq!(orient(3, 4, 4, 1, false), Some((Trans::T, 4)));
        // Conjugation exists only transposed.
        assert_eq!(orient(3, 4, 4, 1, true), Some((Trans::C, 4)));
        assert_eq!(orient(3, 4, 1, 3, true), None);
        // Neither axis unit stride.
        assert_eq!(orient(3, 4, 2, 6, false), None);
    }
}
