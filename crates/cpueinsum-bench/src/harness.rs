//! The run/verify grid: timing, cross-row correctness comparison, and the two
//! evidence CSVs.
//!
//! A *declared cell* is one `(declared shape, family, dtype, batch, engine
//! row)` combination. Every cell gets a record, `ok`, `skipped` (with the
//! reason) or `failed`; the census CSV and the timed CSV are written from the
//! same records, so their cell sets and statuses cannot drift. The engine rows
//! are executed in the order [`Config::rows`] gives, which is the declared row
//! list rotated by `--rotate-rows`, so a whole-process repeat re-measures every
//! case under a different row schedule.

use std::fmt::Write as _;
use std::fs;
use std::time::{Duration, Instant};

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Layout, Scratch};
use num_complex::{Complex32, Complex64};

use crate::corpus::{Instance, Translation};
use crate::elem::{Dtype, Elem};

/// Whether the grid measures or checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Time every cell, keeping the chronological per-call samples.
    Run,
    /// Run every cell once and compare the rows against each other.
    Verify,
}

/// Which summary of the chronological samples a row reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Statistic {
    /// The fastest call.
    #[default]
    Best,
    /// The median call; an even count averages the two central samples.
    Median,
}

impl Statistic {
    /// Parse `best|median`.
    pub fn parse(s: &str) -> Option<Statistic> {
        match s.trim() {
            "best" => Some(Statistic::Best),
            "median" => Some(Statistic::Median),
            _ => None,
        }
    }

    /// Apply the statistic to the samples, in milliseconds.
    pub fn of(self, samples: &[f64]) -> f64 {
        match self {
            Statistic::Best => samples.iter().copied().fold(f64::INFINITY, f64::min),
            Statistic::Median => {
                if samples.is_empty() {
                    return 0.0;
                }
                let mut v = samples.to_vec();
                v.sort_by(f64::total_cmp);
                let n = v.len();
                if n % 2 == 1 {
                    v[n / 2]
                } else {
                    (v[n / 2 - 1] + v[n / 2]) / 2.0
                }
            }
        }
    }
}

/// A cell's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Measured (or, for `verify`, checked).
    Ok,
    /// Declared but not run, with the reason in `note`.
    Skipped,
    /// Reached but wrong or unable to finish. The run exits non-zero.
    Failed,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Skipped => "skipped",
            Status::Failed => "failed",
        }
    }
}

/// One declared cell and its outcome, in execution order.
pub struct Record {
    /// Run's regime label.
    pub regime: String,
    /// Instance name.
    pub family: String,
    /// Declared dtype label.
    pub dtype: String,
    /// Always empty: this corpus has no order axis.
    pub order: String,
    /// The identity `m` derived from the declared size spec.
    pub m: String,
    /// The identity `n` derived from the declared size spec.
    pub n: String,
    /// The identity `k` derived from the declared size spec.
    pub k: String,
    /// The unit batch, echoed.
    pub batch: String,
    /// The engine row (`opt_size` or `opt_flops`).
    pub row: String,
    /// The requested thread budget, echoed.
    pub threads: usize,
    /// The cell's outcome.
    pub status: Status,
    /// Empty for `ok`; the reason for `skipped`/`failed`.
    pub note: String,
    /// The requested statistic of `samples_ms`, for an `ok` cell.
    pub total_ms: Option<f64>,
    /// The chronological per-call wall times, in milliseconds.
    pub samples_ms: Vec<f64>,
    /// Operand count of the instance, when it is known.
    pub case_m: String,
    /// Total input elements of the instance.
    pub case_n: String,
    /// Output element count of the instance.
    pub case_k: String,
}

impl Record {
    fn declared(
        config: &Config,
        dtype: &str,
        instance: &Instance,
        shape: &str,
        batch: u32,
        row: &str,
    ) -> Record {
        let (m, n, k) = declared_shape(shape, &instance.name);
        Record {
            regime: config.regime.clone(),
            family: instance.name.clone(),
            dtype: dtype.to_string(),
            order: String::new(),
            m,
            n,
            k,
            batch: batch.to_string(),
            row: row.to_string(),
            threads: config.threads,
            status: Status::Skipped,
            note: String::new(),
            total_ms: None,
            samples_ms: Vec::new(),
            case_m: instance.operands.to_string(),
            case_n: instance.input_len.to_string(),
            case_k: instance.output_len.to_string(),
        }
    }
}

/// The parsed run/verify request, shared by the grid.
pub struct Config {
    /// Regime label echoed into every cell.
    pub regime: String,
    /// Declared size specs (`--n`); this corpus has no size knob, so typically
    /// the sentinel `1`.
    pub n_specs: Vec<String>,
    /// Declared dtype labels.
    pub dtypes: Vec<String>,
    /// Declared batches; this corpus has unit batch.
    pub batches: Vec<u32>,
    /// The engine rows in execution order (the declared rows rotated).
    pub rows: Vec<String>,
    /// The reported statistic.
    pub statistic: Statistic,
    /// Timed repetitions per cell.
    pub reps: usize,
    /// Untimed priming per cell, in milliseconds.
    pub prime_ms: u64,
    /// The requested thread budget.
    pub threads: usize,
    /// Measure or check.
    pub mode: Mode,
}

/// The identity `(m, n, k)` the toolkit derives from a declared size spec: a
/// 1-dim spec is a square, a 2-dim spec is `m x n`, and a 3-dim spec is
/// `m x k x n`. Only a family literally named `gemm` carries a contracting `k`
/// in the 1-dim form. The sentinel this corpus uses resolves to `1,1,""`.
pub fn declared_shape(spec: &str, family: &str) -> (String, String, String) {
    let dims: Vec<&str> = spec.split('x').collect();
    match dims[..] {
        [m, k, n] => (m.to_string(), n.to_string(), k.to_string()),
        [m, n] => (m.to_string(), n.to_string(), String::new()),
        [m] if family == "gemm" => (m.to_string(), m.to_string(), m.to_string()),
        [m] => (m.to_string(), m.to_string(), String::new()),
        _ => (String::new(), String::new(), String::new()),
    }
}

/// Run the declared grid, returning every record and whether every reached cell
/// passed.
pub fn run_grid(instances: &[Instance], config: &Config, exec: &Exec<'_>) -> (Vec<Record>, bool) {
    let mut records = Vec::new();
    let mut all_ok = true;
    for dtype in &config.dtypes {
        match Dtype::of(dtype) {
            Some(Dtype::F32) => {
                grid::<f32>(instances, config, dtype, exec, &mut records, &mut all_ok)
            }
            Some(Dtype::F64) => {
                grid::<f64>(instances, config, dtype, exec, &mut records, &mut all_ok)
            }
            Some(Dtype::C32) => {
                grid::<Complex32>(instances, config, dtype, exec, &mut records, &mut all_ok)
            }
            Some(Dtype::C64) => {
                grid::<Complex64>(instances, config, dtype, exec, &mut records, &mut all_ok)
            }
            None => unsupported_dtype(instances, config, dtype, &mut records),
        }
    }
    (records, all_ok)
}

fn unsupported_dtype(
    instances: &[Instance],
    config: &Config,
    dtype: &str,
    records: &mut Vec<Record>,
) {
    let note = format!("unsupported dtype {dtype}");
    for shape in &config.n_specs {
        for instance in instances {
            for &batch in &config.batches {
                for row in &config.rows {
                    let mut record = Record::declared(config, dtype, instance, shape, batch, row);
                    record.note = note.clone();
                    records.push(record);
                }
            }
        }
    }
}

fn grid<T: Elem>(
    instances: &[Instance],
    config: &Config,
    dtype: &str,
    exec: &Exec<'_>,
    records: &mut Vec<Record>,
    all_ok: &mut bool,
) {
    for shape in &config.n_specs {
        for instance in instances {
            let skip = |records: &mut Vec<Record>, batch: u32, row: &str, note: &str| {
                let mut record = Record::declared(config, dtype, instance, shape, batch, row);
                record.note = note.to_string();
                records.push(record);
            };
            let translation = match &instance.translation {
                Ok(t) => t,
                Err(note) => {
                    for &batch in &config.batches {
                        for row in &config.rows {
                            skip(records, batch, row, note);
                        }
                    }
                    continue;
                }
            };
            let data = FamilyData::<T>::new(translation);
            for &batch in &config.batches {
                let mut reference: Option<Vec<T>> = None;
                for row in &config.rows {
                    let spec = match instance.spec(row) {
                        Ok(spec) => spec,
                        Err(note) => {
                            skip(records, batch, row, &note);
                            continue;
                        }
                    };
                    let mut record = Record::declared(config, dtype, instance, shape, batch, row);
                    match config.mode {
                        Mode::Run => match time_cell(translation, &data, &spec, exec, config) {
                            Ok(samples) => {
                                record.status = Status::Ok;
                                record.total_ms = Some(config.statistic.of(&samples));
                                record.samples_ms = samples;
                            }
                            Err(note) => {
                                record.status = Status::Failed;
                                record.note = note;
                                *all_ok = false;
                            }
                        },
                        Mode::Verify => match verify_cell(translation, &data, &spec, exec) {
                            Ok(out) => match &reference {
                                None => {
                                    reference = Some(out);
                                    record.status = Status::Ok;
                                }
                                Some(want) => {
                                    let error = rel_error(&out, want);
                                    if error.is_finite() && error <= T::TOLERANCE {
                                        record.status = Status::Ok;
                                    } else {
                                        record.status = Status::Failed;
                                        record.note =
                                            format!("MISMATCH({row}) rel_err={error:.2e}");
                                        *all_ok = false;
                                    }
                                }
                            },
                            Err(note) => {
                                record.status = Status::Failed;
                                record.note = note;
                                *all_ok = false;
                            }
                        },
                    }
                    records.push(record);
                }
            }
        }
    }
}

/// The input buffers for one instance: the same deterministic data for every
/// dtype and both engine rows, so a cross-row comparison compares the rows and
/// not the inputs.
struct FamilyData<T: Elem> {
    inputs: Vec<Vec<T>>,
}

impl<T: Elem> FamilyData<T> {
    fn new(translation: &Translation) -> Self {
        let inputs = translation
            .dims
            .iter()
            .map(|dims| {
                let len: usize = dims.iter().product();
                (0..len).map(T::sample).collect()
            })
            .collect();
        FamilyData { inputs }
    }
}

fn build_views<'a, T: Elem>(
    translation: &'a Translation,
    data: &'a FamilyData<T>,
) -> Result<Vec<StridedView<'a, T>>, String> {
    translation
        .dims
        .iter()
        .zip(&translation.strides)
        .zip(&data.inputs)
        .map(|((dims, strides), buffer)| {
            StridedView::new(buffer, dims, strides, 0).map_err(|e| e.to_string())
        })
        .collect()
}

fn build_plan<T: Elem>(
    spec: &EinsumSpec,
    translation: &Translation,
) -> Result<EinsumPlan<T>, String> {
    let layouts: Vec<Layout<'_>> = translation
        .dims
        .iter()
        .zip(&translation.strides)
        .map(|(dims, strides)| Layout::new(dims, strides).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let out =
        Layout::new(&translation.out_dims, &translation.out_strides).map_err(|e| e.to_string())?;
    EinsumPlan::<T>::new(spec, &layouts, out).map_err(|e| e.to_string())
}

fn execute<T: Elem>(
    plan: &EinsumPlan<T>,
    exec: &Exec<'_>,
    views: &[StridedView<'_, T>],
    translation: &Translation,
    out: &mut [T],
    scratch: &mut Scratch<T>,
) -> Result<(), String> {
    let mut view = StridedViewMut::new(out, &translation.out_dims, &translation.out_strides, 0)
        .map_err(|e| e.to_string())?;
    plan.execute_into(exec, views, &mut view, scratch)
        .map_err(|e| e.to_string())
}

fn time_cell<T: Elem>(
    translation: &Translation,
    data: &FamilyData<T>,
    spec: &EinsumSpec,
    exec: &Exec<'_>,
    config: &Config,
) -> Result<Vec<f64>, String> {
    let plan = build_plan::<T>(spec, translation)?;
    let views = build_views(translation, data)?;
    let mut out = vec![T::default(); translation.out_len];
    let mut scratch = Scratch::<T>::new();
    let mut error: Option<String> = None;
    let samples = timed(config.reps, config.prime_ms, || {
        if error.is_some() {
            return;
        }
        if let Err(e) = execute(&plan, exec, &views, translation, &mut out, &mut scratch) {
            error = Some(e);
        }
    });
    match error {
        Some(e) => Err(format!("execution failed after priming: {e}")),
        None => Ok(samples),
    }
}

fn verify_cell<T: Elem>(
    translation: &Translation,
    data: &FamilyData<T>,
    spec: &EinsumSpec,
    exec: &Exec<'_>,
) -> Result<Vec<T>, String> {
    let plan = build_plan::<T>(spec, translation)?;
    let views = build_views(translation, data)?;
    let mut out = vec![T::default(); translation.out_len];
    let mut scratch = Scratch::<T>::new();
    execute(&plan, exec, &views, translation, &mut out, &mut scratch)?;
    Ok(out)
}

/// Run `f` until a wall-clock priming deadline has passed, then `reps` timed
/// calls, returning every call's wall time in milliseconds in chronological
/// order. Priming is time-based rather than a call count so a fast and a slow
/// cell are warmed for the same duration.
fn timed(reps: usize, prime_ms: u64, mut f: impl FnMut()) -> Vec<f64> {
    if prime_ms > 0 {
        let until = Instant::now() + Duration::from_millis(prime_ms);
        while Instant::now() < until {
            f();
        }
    }
    let n = reps.max(1);
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples
}

/// Relative Frobenius error `||got - want|| / ||want||`.
fn rel_error<T: Elem>(got: &[T], want: &[T]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&g, &w) in got.iter().zip(want) {
        let d = g.dist(w);
        num += d * d;
        let m = w.mag();
        den += m * m;
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// One CSV field, quoted when it carries a comma or a quote.
fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_line(text: &mut String, fields: &[String]) {
    let _ = writeln!(
        text,
        "{}",
        fields
            .iter()
            .map(|field| csv_field(field))
            .collect::<Vec<_>>()
            .join(",")
    );
}

/// Write one census row per declared cell, including declared skips.
pub fn write_census(path: &str, records: &[Record]) -> std::io::Result<()> {
    let mut text = String::from("regime,family,dtype,order,m,n,k,batch,row,threads,status,note\n");
    for r in records {
        write_line(
            &mut text,
            &[
                r.regime.clone(),
                r.family.clone(),
                r.dtype.clone(),
                r.order.clone(),
                r.m.clone(),
                r.n.clone(),
                r.k.clone(),
                r.batch.clone(),
                r.row.clone(),
                r.threads.to_string(),
                r.status.as_str().to_string(),
                r.note.clone(),
            ],
        );
    }
    fs::write(path, text)
}

/// Write the timed evidence, one row per declared cell, carrying the
/// chronological per-call samples and the requested statistic.
pub fn write_timed(path: &str, records: &[Record]) -> std::io::Result<()> {
    let mut text = String::from(
        "regime,family,dtype,order,m,n,k,batch,row,threads,status,note,total_ms,per_item_us,samples_ms,rayon_threads,vendor_threads,case_m,case_n,case_k\n",
    );
    for r in records {
        let batch: f64 = r.batch.parse().unwrap_or(1.0);
        let (total, per_item, samples) = match (r.status, r.total_ms) {
            (Status::Ok, Some(total)) => (
                format!("{total:.9}"),
                format!("{:.9}", total * 1000.0 / batch),
                format!(
                    "[{}]",
                    r.samples_ms
                        .iter()
                        .map(|v| format!("{v:.12}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
            _ => (String::new(), String::new(), String::new()),
        };
        let threads = r.threads.to_string();
        write_line(
            &mut text,
            &[
                r.regime.clone(),
                r.family.clone(),
                r.dtype.clone(),
                r.order.clone(),
                r.m.clone(),
                r.n.clone(),
                r.k.clone(),
                r.batch.clone(),
                r.row.clone(),
                threads.clone(),
                r.status.as_str().to_string(),
                r.note.clone(),
                total,
                per_item,
                samples,
                threads.clone(),
                threads,
                r.case_m.clone(),
                r.case_n.clone(),
                r.case_k.clone(),
            ],
        );
    }
    fs::write(path, text)
}

/// The dtype labels declared by the loaded corpus, sorted and deduplicated;
/// the default `--dtype` list.
pub fn corpus_dtypes(instances: &[Instance]) -> Vec<String> {
    let mut labels: Vec<String> = instances.iter().map(|i| i.dtype.clone()).collect();
    labels.sort();
    labels.dedup();
    labels
}
