//! Replaying one recorded call on each route: buffers, plans, the checked twin
//! and the timer.

use core::mem::MaybeUninit;
use std::fmt;
use std::time::{Duration, Instant};

use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{BatchPolicy, BatchStrategy, Exec, GroupedGemmJob};
use cpueinsum_blas::BlasScalar;
use num_complex::Complex64;

/// Why the checked twin may demand exact equality.
///
/// Recorded verbatim in the export metadata.
pub const EXACTNESS: &str = "Inputs are small integers: lhs re = (i mod 7) - 3, rhs re = \
    (i mod 5) - 2, and for complex128 lhs im = (i mod 3) - 1, rhs im = ((i + 1) mod 3) - 1, with i \
    the element index in the shared buffer. Every product component is an integer of magnitude at \
    most 6 (float64) or 7 (complex128: |ar br| + |ai bi| <= 6 + 1), so every partial sum of a job \
    with contraction length k is an integer of magnitude at most 7k. alpha = 1 and beta is 0 or \
    fresh. While 7k < 2^53 every sum, in any order, blocking or with FMA, is exact, so every route \
    must produce exactly the same values. Values are compared with IEEE equality: the sign of a \
    zero is not compared, because (-3) * 0 = -0.0 starts one summation order and +0.0 another.";

/// Largest magnitude of one product component the fill can produce.
pub const PRODUCT_BOUND: f64 = 7.0;

/// The rows a cell can declare, in default order.
pub const ROWS: [&str; 5] = ["sequential", "outer", "auto", "inner", "blas"];

/// A declared route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    /// `cpueinsum::GroupedPlan` under `BatchStrategy::Sequential`.
    Sequential,
    /// ... under `Outer`.
    Outer,
    /// ... under `Auto` (today's selector).
    Auto,
    /// ... under `Inner`.
    Inner,
    /// `cpueinsum_blas::GroupedPlan` under `Sequential` (the vendor loop).
    Blas,
}

impl Row {
    /// The row a command-line name selects.
    pub fn of(name: &str) -> Option<Row> {
        Some(match name {
            "sequential" => Row::Sequential,
            "outer" => Row::Outer,
            "auto" => Row::Auto,
            "inner" => Row::Inner,
            "blas" => Row::Blas,
            _ => return None,
        })
    }

    fn strategy(self) -> BatchStrategy {
        match self {
            Row::Sequential | Row::Blas => BatchStrategy::Sequential,
            Row::Outer => BatchStrategy::Outer,
            Row::Auto => BatchStrategy::Auto,
            Row::Inner => BatchStrategy::Inner,
        }
    }
}

/// The route a call actually took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observed {
    /// A native row.
    Native(cpueinsum::GroupedRoute),
    /// The vendor row.
    Blas(cpueinsum_blas::GroupedRoute),
}

impl Observed {
    /// The vendor plan declined the group and ran its native alternative.
    pub fn vendor_declined(self) -> bool {
        matches!(
            self,
            Observed::Blas(cpueinsum_blas::GroupedRoute::Native(_))
        )
    }
}

impl fmt::Display for Observed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Observed::Native(r) => write!(f, "{r:?}"),
            Observed::Blas(r) => write!(f, "{r:?}"),
        }
    }
}

/// An element type the replay drives.
pub trait Elem: BlasScalar + Copy + PartialEq + fmt::Debug {
    /// The integer-valued left input at buffer position `i`.
    fn lhs(i: usize) -> Self;
    /// The integer-valued right input at buffer position `i`.
    fn rhs(i: usize) -> Self;
    /// A value no route writes, to catch an unwritten slot.
    fn nan() -> Self;
    /// `acc + a * b`.
    fn mul_add(acc: Self, a: Self, b: Self) -> Self;
    /// The largest component magnitude (NaN propagates).
    fn max_abs(self) -> f64;
}

fn re_lhs(i: usize) -> f64 {
    (i % 7) as f64 - 3.0
}
fn re_rhs(i: usize) -> f64 {
    (i % 5) as f64 - 2.0
}

impl Elem for f64 {
    fn lhs(i: usize) -> Self {
        re_lhs(i)
    }
    fn rhs(i: usize) -> Self {
        re_rhs(i)
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn mul_add(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
    fn max_abs(self) -> f64 {
        self.abs()
    }
}

impl Elem for Complex64 {
    fn lhs(i: usize) -> Self {
        Complex64::new(re_lhs(i), (i % 3) as f64 - 1.0)
    }
    fn rhs(i: usize) -> Self {
        Complex64::new(re_rhs(i), ((i + 1) % 3) as f64 - 1.0)
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn mul_add(acc: Self, a: Self, b: Self) -> Self {
        acc + a * b
    }
    fn max_abs(self) -> f64 {
        if self.re.is_nan() || self.im.is_nan() {
            f64::NAN
        } else {
            self.re.abs().max(self.im.abs())
        }
    }
}

/// `C = A B` by the definition, column by column; the checked twin's
/// independent reference for `sequential`.
pub fn naive<T: Elem>(jobs: &[GroupedGemmJob], a: &[T], b: &[T], out_len: usize) -> Vec<T> {
    let mut out = vec![T::nan(); out_len];
    let mut col = Vec::new();
    for job in jobs {
        let (m, k) = (job.rows, job.contracted);
        for j in 0..job.cols {
            col.clear();
            col.resize(m, T::default());
            for p in 0..k {
                let bpj = b[job.rhs_offset + p + k * j];
                let a_col = &a[job.lhs_offset + m * p..][..m];
                for (c, &aip) in col.iter_mut().zip(a_col) {
                    *c = T::mul_add(*c, aip, bpj);
                }
            }
            out[job.out_offset + m * j..][..m].copy_from_slice(&col);
        }
    }
    out
}

/// The first position where `got` differs from `want` (IEEE equality, so a
/// NaN always differs), as a note.
pub fn mismatch<T: Elem>(got: &[T], want: &[T]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("length {} != {}", got.len(), want.len()));
    }
    got.iter()
        .zip(want)
        .position(|(g, w)| g != w)
        .map(|i| format!("element {i}: {:?} != {:?}", got[i], want[i]))
}

/// One call's buffers and both prepared plans, built once and shared by every
/// row of the call.
pub struct Cell<T: Elem> {
    jobs: Vec<GroupedGemmJob>,
    fresh: bool,
    a: Vec<T>,
    b: Vec<T>,
    out: Vec<T>,
    fresh_out: Vec<MaybeUninit<T>>,
    work: Vec<T>,
    native: cpueinsum::GroupedPlan<T>,
    blas: cpueinsum_blas::GroupedPlan<T>,
}

impl<T: Elem> Cell<T> {
    /// Prepare `jobs`; `fresh` replays a never-read destination.
    pub fn new(jobs: Vec<GroupedGemmJob>, fresh: bool) -> Result<Self, String> {
        let config = PlanConfig::default();
        let native = cpueinsum::GroupedPlan::<T>::new(&jobs, &config).map_err(|e| e.to_string())?;
        let blas =
            cpueinsum_blas::GroupedPlan::<T>::new(&jobs, &config).map_err(|e| e.to_string())?;
        let out_len = native.required_out_len();
        if fresh {
            native
                .validate_uninit(T::default(), out_len)
                .map_err(|e| format!("fresh call: {e}"))?;
        }
        Ok(Cell {
            a: (0..native.required_lhs_len()).map(T::lhs).collect(),
            b: (0..native.required_rhs_len()).map(T::rhs).collect(),
            out: vec![T::nan(); out_len],
            fresh_out: vec![MaybeUninit::new(T::nan()); if fresh { out_len } else { 0 }],
            work: vec![T::default(); blas.work_len()],
            jobs,
            fresh,
            native,
            blas,
        })
    }

    /// The definition's result for this call.
    pub fn reference(&self) -> Vec<T> {
        naive(&self.jobs, &self.a, &self.b, self.out.len())
    }

    /// Run `row` once. With `keep`, the output is first reset to NaN and then
    /// returned.
    pub fn call(
        &mut self,
        row: Row,
        exec: &mut Exec<'_>,
        keep: bool,
    ) -> Result<(Observed, Option<Vec<T>>), String> {
        let mut exec = exec
            .reborrow()
            .with_batch_policy(BatchPolicy::new(row.strategy()));
        let (one, zero) = (T::ONE, T::default());
        let (a, b, work) = (&self.a[..], &self.b[..], &mut self.work[..]);
        if self.fresh {
            if keep {
                self.fresh_out.fill(MaybeUninit::new(T::nan()));
            }
            let storage = &mut self.fresh_out[..];
            let (written, route) = if row == Row::Blas {
                let (w, r) = self
                    .blas
                    .execute_uninit(&mut exec, one, a, b, zero, storage, work)
                    .map_err(|e| e.to_string())?;
                (w, Observed::Blas(r))
            } else {
                let (w, r) = self
                    .native
                    .execute_uninit(&mut exec, one, a, b, zero, storage)
                    .map_err(|e| e.to_string())?;
                (w, Observed::Native(r))
            };
            return Ok((route, keep.then(|| written.to_vec())));
        }
        if keep {
            self.out.fill(T::nan());
        }
        let out = &mut self.out[..];
        let route = if row == Row::Blas {
            self.blas
                .execute_into_accum(&mut exec, one, a, b, zero, out, work)
                .map(Observed::Blas)
                .map_err(|e| e.to_string())?
        } else {
            self.native
                .execute_into_accum(&mut exec, one, a, b, zero, out)
                .map(Observed::Native)
                .map_err(|e| e.to_string())?
        };
        Ok((route, keep.then(|| self.out.clone())))
    }
}

/// Time `call`: prime for `prime_ms` of wall time, set the calls per sample so
/// a sample lasts at least 1 ms, then take `reps` samples of that many
/// back-to-back calls. Returns the per-call time of each sample in
/// nanoseconds, the calls per sample, and the route, which must not change.
pub fn time(
    reps: usize,
    prime_ms: u64,
    mut call: impl FnMut() -> Result<Observed, String>,
) -> Result<(Vec<f64>, u64, Observed), String> {
    // One untimed call even without priming: plan children, scratch and the
    // pool's workspace are first touched here, never in a sample.
    let route = call()?;
    let mut checked = || -> Result<(), String> {
        let r = call()?;
        if r == route {
            Ok(())
        } else {
            Err(format!("route changed from {route} to {r}"))
        }
    };
    let start = Instant::now();
    let mut calls = 0u32;
    while start.elapsed() < Duration::from_millis(prime_ms) {
        checked()?;
        calls += 1;
    }
    let single = if calls > 0 {
        start.elapsed() / calls
    } else {
        let s = Instant::now();
        checked()?;
        s.elapsed()
    };
    let inner = (1_000_000u128.div_ceil(single.as_nanos().max(1))).max(1) as u64;
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        for _ in 0..inner {
            checked()?;
        }
        samples.push(s.elapsed().as_nanos() as f64 / inner as f64);
    }
    Ok((samples, inner, route))
}

/// The median; an even count averages the two central samples.
pub fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => f64::NAN,
        _ if n % 2 == 1 => v[n / 2],
        _ => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

#[cfg(test)]
mod tests;
