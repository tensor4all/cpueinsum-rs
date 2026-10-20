//! Cross-implementation diagnostics.
//!
//! A campaign page that compares two implementations has to show they compute the
//! same thing, and the harness cannot do that by comparing its own rows when a suite
//! carries one row (one strategy) or when the other implementation is a different
//! process (a Julia or a torch arm). This module defines the values both sides
//! compute from their own output and the file they exchange.
//!
//! The four values are cheap (one pass over the output, outside the timed region) and
//! order-independent, which is what a cross-language comparison needs: implementations
//! do not produce bit-identical floats, but they agree on these to a relative tolerance.
//!
//! `weighted` is the one that matters for these benchmarks. `sum`, `sum_sq` and
//! `max_abs` are invariant under a permutation of the output, and the mistakes these
//! campaigns look for are exactly permutations - a transposed result, a stride read
//! the wrong way, a batch lane written into the wrong slot. A weight that depends on
//! the element's linear index is not invariant under any of them.

use crate::elem::Elem;
use crate::harness::Record;
use std::collections::HashMap;
use std::fs;
use std::io;

/// The values one implementation reports for one case.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Diagnostics {
    /// Real part of `sum_i x_i`.
    pub sum_re: f64,
    /// Imaginary part of `sum_i x_i` (zero for a real dtype).
    pub sum_im: f64,
    /// `sum_i |x_i|^2`, the squared Frobenius norm.
    pub sum_sq: f64,
    /// `max_i |x_i|`.
    pub max_abs: f64,
    /// Real part of `sum_i w_i x_i`.
    pub weighted_re: f64,
    /// Imaginary part of `sum_i w_i x_i`.
    pub weighted_im: f64,
}

/// The fixed weight for linear index `i`, in `[0, 1)`.
///
/// SplitMix64 of `i` folded to 53 bits: the same value in every implementation, and
/// independent enough of the index that a permutation cannot preserve the weighted sum
/// by accident. The constant and the shift are part of the cross-arm contract; a Julia
/// runner reproduces them exactly.
pub fn weight(i: usize) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

impl Diagnostics {
    /// Measure `out` in row-major order, which is the order both sides must use.
    pub fn of<T: Elem>(out: &[T]) -> Self {
        let mut d = Diagnostics::default();
        for (i, &x) in out.iter().enumerate() {
            let (re, im) = (Elem::re(x), Elem::im(x));
            d.sum_re += re;
            d.sum_im += im;
            d.sum_sq += re * re + im * im;
            d.max_abs = d.max_abs.max(x.mag());
            let w = weight(i);
            d.weighted_re += w * re;
            d.weighted_im += w * im;
        }
        d
    }

    /// The six comparisons, and the largest relative disagreement among them.
    ///
    /// `max(|a|, |b|, 1)` in the denominator keeps a near-zero value from making the
    /// tolerance meaningless. Returns `None` when every value agrees.
    pub fn disagree(&self, other: &Self, tol: f64) -> Option<(&'static str, f64)> {
        let pairs = [
            ("sum_re", self.sum_re, other.sum_re),
            ("sum_im", self.sum_im, other.sum_im),
            ("sum_sq", self.sum_sq, other.sum_sq),
            ("max_abs", self.max_abs, other.max_abs),
            ("weighted_re", self.weighted_re, other.weighted_re),
            ("weighted_im", self.weighted_im, other.weighted_im),
        ];
        let mut worst: Option<(&'static str, f64)> = None;
        for (name, a, b) in pairs {
            let ratio = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
            if ratio > tol && worst.map(|(_, w)| ratio > w).unwrap_or(true) {
                worst = Some((name, ratio));
            }
        }
        worst
    }
}

/// The key a case is compared under: everything that decides *what* was computed,
/// and nothing that decides who computed it. The engine row is deliberately absent:
/// two arms name their rows differently (`opt_size`, `omeinsum`, `torch`) and mean the
/// same work.
pub fn case_key(r: &Record) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        r.regime, r.family, r.dtype, r.m, r.n, r.k, r.batch
    )
}

const HEADER: &str = "regime,family,dtype,m,n,k,batch,row,sum_re,sum_im,sum_sq,max_abs,weighted_re,weighted_im,status\n";

/// Write one line per measured cell, in the order the run measured them.
pub fn write(path: &str, records: &[Record]) -> io::Result<()> {
    let mut text = String::from(HEADER);
    for r in records {
        if let Some(d) = r.diagnostics {
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}\n",
                r.regime,
                r.family,
                r.dtype,
                r.m,
                r.n,
                r.k,
                r.batch,
                r.row,
                d.sum_re,
                d.sum_im,
                d.sum_sq,
                d.max_abs,
                d.weighted_re,
                d.weighted_im,
                r.status.as_str()
            ));
        }
    }
    fs::write(path, text)
}

/// Read a diagnostics file written by another arm, keyed by [`case_key`].
///
/// A file with the wrong header is refused rather than half-read: a reference that
/// silently contributes no cases would turn a cross-implementation check into none.
pub fn read(path: &str) -> Result<HashMap<String, Diagnostics>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("cannot read the reference {path}: {e}"))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    if header.trim() != HEADER.trim() {
        return Err(format!(
            "the reference {path} does not carry the diagnostics header this harness writes: {header:?}"
        ));
    }
    let mut out = HashMap::new();
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 15 {
            return Err(format!(
                "the reference {path} line {} has {} fields",
                n + 2,
                f.len()
            ));
        }
        let num = |i: usize| -> f64 { f[i].parse::<f64>().unwrap_or(f64::NAN) };
        let key = format!(
            "{}|{}|{}|{}|{}|{}|{}",
            f[0], f[1], f[2], f[3], f[4], f[5], f[6]
        );
        out.insert(
            key,
            Diagnostics {
                sum_re: num(8),
                sum_im: num(9),
                sum_sq: num(10),
                max_abs: num(11),
                weighted_re: num(12),
                weighted_im: num(13),
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
