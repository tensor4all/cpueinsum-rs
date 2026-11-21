//! `cpueinsum-replay` -- the timing-v2 harness that replays the grouped GEMM
//! calls of a `symtn-gemm-trace/1` corpus (tensor4all/cpueinsum-benchmark#18)
//! on each `GroupedPlan` route.
//!
//! Subcommands, in the campaign recorder's harness contract:
//!
//! * `run` -- time every declared cell (call x route at one width) and write
//!   the timed CSV.
//! * `verify` -- run every declared cell once with integer-valued data and
//!   compare it exactly: `sequential` against the definition, every other row
//!   against `sequential` at the same width. Prints
//!   `all comparisons within tolerance` last, only when every comparison passed.
//! * `info` -- the build and the linked vendor as `key=value` lines.
//! * `export` -- fold the timed CSVs of one host into one JSON file for the
//!   route fit: one row per timed cell, with the per-call summary, the capture
//!   digests and the campaign commit as metadata.
//!
//! A family is `<capture stem>:rNNN`, call `NNN` of that capture. Each width
//! replays the capture `--width-capture` names for it (a 4-thread capture at
//! width 4, a full-core capture at full width); a family of another capture is
//! a skipped cell. The rows are `sequential`, `outer`, `auto`, `inner`
//! (`cpueinsum::GroupedPlan` under that `BatchStrategy`) and `blas`
//! (`cpueinsum_blas::GroupedPlan` on the linked vendor). This crate is separate
//! from `cpueinsum-bench` because the `blas` row links a provider, which the
//! einsum harness's binary (and its recorded identity) must not.

// The `blas` row's provider.
extern crate blas_src as _;

mod corpus;
mod export;
mod replay;
mod vendor;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;

use cpueinsum::{ArenaProvider, BatchPolicy, Exec};
use num_complex::Complex64;

use crate::corpus::{Call, Capture, Dtype};
use crate::replay::{Cell, Elem, Row};

fn die(message: &str) -> ExitCode {
    eprintln!("cpueinsum-replay: {message}");
    ExitCode::from(2)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Run,
    Verify,
}

struct Args {
    mode: Mode,
    threads: usize,
    corpus_dir: String,
    width_capture: BTreeMap<usize, String>,
    families: Vec<String>,
    dtypes: Vec<Dtype>,
    n_specs: Vec<String>,
    rows: Vec<Row>,
    row_names: Vec<String>,
    rotate: usize,
    reps: usize,
    prime_ms: u64,
    regime: String,
    csv: Option<String>,
    census_csv: Option<String>,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    match argv.get(1).map(String::as_str) {
        Some("info") => info(&argv[2..]),
        Some("export") => match export::main(&argv[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => die(&message),
        },
        Some("run" | "verify") => match parse(&argv) {
            Ok(args) => measure(&args),
            Err(message) => die(&message),
        },
        other => die(&format!(
            "expected run, verify, info or export, got {other:?}"
        )),
    }
}

fn info(rest: &[String]) -> ExitCode {
    let threads = match rest {
        [] => None,
        [flag, value] if flag == "--threads" => match value.parse::<usize>() {
            Ok(t) if t > 0 => Some(t),
            _ => {
                return die(&format!(
                    "--threads {value:?}: expected a positive whole number"
                ))
            }
        },
        _ => return die("info takes only --threads"),
    };
    let mut effective = String::new();
    if let Some(t) = threads {
        if let Err(message) = vendor::set_threads(t) {
            return die(&message);
        }
        match threadteam::Pool::builder().threads(t).build() {
            Ok(pool) if pool.width().get() == t => effective = t.to_string(),
            Ok(pool) => {
                return die(&format!(
                    "requested {t} threads, the pool has {}",
                    pool.width()
                ))
            }
            Err(e) => return die(&format!("cannot create a {t}-worker pool: {e}")),
        }
    }
    println!("harness     : cpueinsum-replay");
    println!(
        "target      : {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("provider    : cpueinsum {}", cpueinsum::VERSION);
    println!("provider=cpueinsum");
    println!("provider_version={}", cpueinsum::VERSION);
    println!("corpus_format={}", corpus::FORMAT);
    println!(
        "threads={}",
        threads.map(|t| t.to_string()).unwrap_or_default()
    );
    println!("threads_effective={effective}");
    for (key, value) in vendor::identity() {
        println!("{key}={value}");
    }
    ExitCode::SUCCESS
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mode = if argv[1] == "run" {
        Mode::Run
    } else {
        Mode::Verify
    };
    let mut args = Args {
        mode,
        threads: 1,
        corpus_dir: String::new(),
        width_capture: BTreeMap::new(),
        families: Vec::new(),
        dtypes: Vec::new(),
        n_specs: vec!["1".into()],
        rows: Vec::new(),
        row_names: Vec::new(),
        rotate: 0,
        reps: 5,
        prime_ms: 250,
        regime: "default".into(),
        csv: None,
        census_csv: None,
    };
    let mut i = 2;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--threads" => args.threads = whole(value, flag)?,
            "--corpus-dir" => args.corpus_dir = value.clone(),
            "--width-capture" => {
                for entry in list(value, flag)? {
                    let (w, stem) = entry
                        .split_once('=')
                        .ok_or_else(|| format!("{flag} {entry:?}: expected WIDTH=STEM"))?;
                    if args
                        .width_capture
                        .insert(whole(w, flag)?, stem.into())
                        .is_some()
                    {
                        return Err(format!("{flag}: width {w} named twice"));
                    }
                }
            }
            "--family" => args.families = list(value, flag)?,
            "--dtype" => {
                args.dtypes = list(value, flag)?
                    .iter()
                    .map(|d| Dtype::of_label(d).ok_or_else(|| format!("unsupported dtype {d:?}")))
                    .collect::<Result<_, _>>()?;
            }
            "--n" => args.n_specs = list(value, flag)?,
            "--batch" => {
                if list(value, flag)?.iter().any(|b| b != "1") {
                    return Err(
                        "--batch: a recorded call has unit batch; every value must be 1".into(),
                    );
                }
            }
            "--rows" => args.row_names = list(value, flag)?,
            "--rotate-rows" => args.rotate = whole(value, flag)?,
            "--statistic" => {
                if value != "median" {
                    return Err("--statistic: this harness reports the median only".into());
                }
            }
            "--reps" => args.reps = whole(value, flag)?,
            "--prime-ms" => args.prime_ms = whole(value, flag)? as u64,
            "--regime" => args.regime = value.clone(),
            "--csv" => args.csv = Some(value.clone()),
            "--census-csv" => args.census_csv = Some(value.clone()),
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 2;
    }
    dedup(&mut args.families);
    dedup(&mut args.n_specs);
    dedup(&mut args.row_names);
    args.dtypes.dedup();
    if args.threads == 0 {
        return Err("--threads must be positive".into());
    }
    if args.corpus_dir.is_empty() {
        return Err("--corpus-dir DIR is required".into());
    }
    if !args.width_capture.contains_key(&args.threads) {
        return Err(format!(
            "--width-capture names no capture for width {}; each width replays a declared capture",
            args.threads
        ));
    }
    if args.n_specs.iter().any(|n| n != "1") {
        return Err(
            "--n: a recorded call has no size knob; every value must be the sentinel 1".into(),
        );
    }
    if args.reps == 0 || (mode == Mode::Run && args.reps < 2) {
        return Err("--reps must be at least 2 for a run and 1 for a verify".into());
    }
    if args.row_names.is_empty() {
        args.row_names = replay::ROWS.iter().map(|r| r.to_string()).collect();
    }
    args.rows = args
        .row_names
        .iter()
        .map(|r| {
            Row::of(r).ok_or_else(|| {
                format!(
                    "unknown row {r:?}; this binary carries {}",
                    replay::ROWS.join(",")
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let len = args.rows.len();
    args.rows.rotate_left(args.rotate % len);
    args.row_names.rotate_left(args.rotate % len);
    if mode == Mode::Run && args.csv.is_none() {
        return Err("run needs --csv PATH".into());
    }
    if mode == Mode::Verify && args.csv.is_none() && args.census_csv.is_none() {
        return Err("verify needs --csv PATH or --census-csv PATH".into());
    }
    Ok(args)
}

fn whole(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} {value:?}: expected a whole number"))
}

fn list(value: &str, flag: &str) -> Result<Vec<String>, String> {
    let values: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    if values.is_empty() {
        return Err(format!("{flag} needs at least one value"));
    }
    Ok(values)
}

fn dedup(values: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    values.retain(|v| seen.insert(v.clone()));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Ok,
    Skipped,
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
struct Record {
    family: String,
    dtype: &'static str,
    n: String,
    row: String,
    status: Status,
    note: String,
    samples_ns: Vec<f64>,
    route: String,
    inner: u64,
    max_abs: Option<f64>,
    capture: String,
    sha256: String,
    jobs: usize,
    work: u128,
    out_len: usize,
    dominant: (usize, usize, usize),
}

struct Grid<'a> {
    args: &'a Args,
    records: Vec<Record>,
    all_ok: bool,
    compared: bool,
}

fn measure(args: &Args) -> ExitCode {
    // Before any BLAS call; on macOS this may re-execute the process.
    match vendor::set_threads(args.threads) {
        Ok(Some(got)) if got != args.threads => {
            return die(&format!(
                "requested {} vendor threads, the vendor reports {got}",
                args.threads
            ))
        }
        Ok(_) => {}
        Err(message) => return die(&message),
    }
    let captures = match corpus::load_dir(Path::new(&args.corpus_dir)) {
        Ok(c) => c,
        Err(message) => return die(&message),
    };
    let mut index = BTreeMap::new();
    for (c, capture) in captures.iter().enumerate() {
        for call in &capture.calls {
            index.insert(capture.family(call.index), (c, call.index));
        }
    }
    let stem = &args.width_capture[&args.threads];
    if !captures.iter().any(|c| &c.stem == stem) {
        return die(&format!(
            "--width-capture names {stem:?}, which is not in the corpus"
        ));
    }
    let families: Vec<String> = if args.families.is_empty() {
        index.keys().cloned().collect()
    } else {
        args.families.clone()
    };
    if let Some(unknown) = families.iter().find(|f| !index.contains_key(*f)) {
        return die(&format!("unknown family {unknown:?}"));
    }
    let dtypes = if args.dtypes.is_empty() {
        vec![Dtype::F64, Dtype::C64]
    } else {
        args.dtypes.clone()
    };

    // The pool, its team and the budget the runtime actually granted, held
    // across verification and timing; a clamped request fails.
    let t = args.threads;
    let pool = match threadteam::Pool::builder().threads(t).build() {
        Ok(pool) if pool.width().get() == t => pool,
        Ok(pool) => {
            return die(&format!(
                "requested {t} threads, the pool has {}",
                pool.width()
            ))
        }
        Err(e) => return die(&format!("cannot create a {t}-worker pool: {e}")),
    };
    let team = match pool.into_team(t) {
        Ok(team) if team.size().get() == t => team,
        Ok(team) => {
            return die(&format!(
                "requested {t} threads, the team has {}",
                team.size()
            ))
        }
        Err(e) => return die(&format!("cannot form a {t}-thread team: {e}")),
    };
    let arena = ArenaProvider::new();
    let mut exec = Exec::from_token(team.exec(t), Some(&arena), BatchPolicy::default());
    if exec.budget() != t || (t > 1 && !exec.token_ref().carries_team()) {
        return die(&format!(
            "requested a {t}-thread team budget, got {}",
            exec.budget()
        ));
    }

    let mut grid = Grid {
        args,
        records: Vec::new(),
        all_ok: true,
        compared: false,
    };
    for &dtype in &dtypes {
        for n in &args.n_specs {
            for family in &families {
                let (c, i) = index[family];
                let capture = &captures[c];
                let call = &capture.calls[i];
                let base = grid.declared(capture, call, family, dtype, n);
                let admitted = if call.dtype != dtype {
                    Err(format!("the call is {}", call.dtype.label()))
                } else if &capture.stem != stem {
                    Err(format!("width {t} replays capture {stem}"))
                } else {
                    call.admit()
                };
                let jobs = match admitted {
                    Ok(jobs) => jobs,
                    Err(note) => {
                        grid.skip_all(&base, &note);
                        continue;
                    }
                };
                match dtype {
                    Dtype::F64 => grid.family::<f64>(&base, call, jobs, &mut exec),
                    Dtype::C64 => grid.family::<Complex64>(&base, call, jobs, &mut exec),
                }
            }
        }
    }
    finish(grid)
}

impl Grid<'_> {
    fn declared(
        &self,
        capture: &Capture,
        call: &Call,
        family: &str,
        dtype: Dtype,
        n: &str,
    ) -> Record {
        let summary = call.summary();
        Record {
            family: family.into(),
            dtype: dtype.label(),
            n: n.into(),
            row: String::new(),
            status: Status::Skipped,
            note: String::new(),
            samples_ns: Vec::new(),
            route: String::new(),
            inner: 0,
            max_abs: None,
            capture: capture.stem.clone(),
            sha256: capture.sha256.clone(),
            jobs: summary.jobs,
            work: summary.work,
            out_len: call
                .jobs
                .iter()
                .map(|j| j[2] + j[3] * j[4])
                .max()
                .unwrap_or(0),
            dominant: summary.dominant,
        }
    }

    fn push(&mut self, base: &Record, row: &str, status: Status, note: String) -> &mut Record {
        if status == Status::Failed {
            self.all_ok = false;
        }
        self.records.push(Record {
            row: row.into(),
            status,
            note,
            family: base.family.clone(),
            n: base.n.clone(),
            capture: base.capture.clone(),
            sha256: base.sha256.clone(),
            samples_ns: Vec::new(),
            route: String::new(),
            ..*base
        });
        self.records.last_mut().expect("just pushed")
    }

    fn skip_all(&mut self, base: &Record, note: &str) {
        for r in 0..self.args.rows.len() {
            let name = self.args.row_names[r].clone();
            self.push(base, &name, Status::Skipped, note.into());
        }
    }

    fn family<T: Elem>(
        &mut self,
        base: &Record,
        call: &Call,
        jobs: Vec<cpueinsum::GroupedGemmJob>,
        exec: &mut Exec<'_>,
    ) {
        let mut cell = match Cell::<T>::new(jobs, call.beta.is_none()) {
            Ok(cell) => cell,
            Err(note) => {
                for r in 0..self.args.rows.len() {
                    let name = self.args.row_names[r].clone();
                    self.push(base, &name, Status::Failed, format!("prepare: {note}"));
                }
                return;
            }
        };
        let args = self.args;
        // verify: `sequential` against the definition first; every row is then
        // compared with that output.
        let reference = if args.mode == Mode::Verify {
            let want = cell.reference();
            Some(match cell.call(Row::Sequential, exec, true) {
                Ok((_, Some(got))) => match replay::mismatch(&got, &want) {
                    None => Ok(got),
                    Some(at) => Err(format!("MISMATCH(sequential vs definition) {at}")),
                },
                Ok((_, None)) => Err("no output".into()),
                Err(e) => Err(format!("execution failed: {e}")),
            })
        } else {
            None
        };
        for (row, name) in args.rows.iter().zip(&args.row_names) {
            let first = match cell.call(*row, exec, args.mode == Mode::Verify) {
                Ok(first) => first,
                Err(e) => {
                    self.push(base, name, Status::Failed, format!("execution failed: {e}"));
                    continue;
                }
            };
            if first.0.vendor_declined() {
                let note = format!("vendor declined, ran native ({})", first.0);
                self.push(base, name, Status::Skipped, note);
                continue;
            }
            match &reference {
                Some(reference) => {
                    self.compared = true;
                    let got = first.1.unwrap_or_default();
                    let max_abs = got.iter().map(|v| v.max_abs()).fold(0.0, f64::max);
                    let outcome = match reference {
                        Err(e) => Err(e.clone()),
                        Ok(want) => match replay::mismatch(&got, want) {
                            None => Ok(()),
                            Some(at) => Err(format!("MISMATCH({name} vs sequential) {at}")),
                        },
                    };
                    let (status, note) = match outcome {
                        Ok(()) => (Status::Ok, String::new()),
                        Err(note) => (Status::Failed, note),
                    };
                    let record = self.push(base, name, status, note);
                    record.route = first.0.to_string();
                    record.max_abs = Some(max_abs);
                }
                None => {
                    let timed = replay::time(args.reps, args.prime_ms, || {
                        cell.call(*row, exec, false).map(|(r, _)| r)
                    });
                    match timed {
                        Ok((samples, inner, route)) => {
                            let record = self.push(base, name, Status::Ok, String::new());
                            record.samples_ns = samples;
                            record.inner = inner;
                            record.route = route.to_string();
                        }
                        Err(e) => {
                            self.push(base, name, Status::Failed, format!("execution failed: {e}"));
                        }
                    }
                }
            }
        }
    }
}

fn finish(grid: Grid<'_>) -> ExitCode {
    let args = grid.args;
    let records = &grid.records;
    if args.mode == Mode::Verify && !grid.compared {
        return die("verify made no comparison: every declared cell was skipped");
    }
    let count = |s: Status| records.iter().filter(|r| r.status == s).count();
    println!(
        "{}: {} declared cell(s), {} ok, {} skipped, {} failed",
        if args.mode == Mode::Verify {
            "verify"
        } else {
            "run"
        },
        records.len(),
        count(Status::Ok),
        count(Status::Skipped),
        count(Status::Failed),
    );
    for r in records.iter().filter(|r| r.status == Status::Failed) {
        println!("FAILED {} {} {} {}", r.row, r.family, r.dtype, r.note);
    }
    if let Some(max) = records.iter().filter_map(|r| r.max_abs).reduce(f64::max) {
        println!("verify: max |value| {max}");
    }
    let verify = args.mode == Mode::Verify;
    let outputs = [(&args.csv, verify), (&args.census_csv, true)];
    for (path, census) in outputs {
        if let Some(path) = path {
            let text = if census {
                census_csv(args, records)
            } else {
                timed_csv(args, records)
            };
            if let Err(e) = std::fs::write(path, text) {
                return die(&format!("{path}: {e}"));
            }
            println!("wrote {path}");
        }
    }
    match (verify, grid.all_ok) {
        (true, true) => {
            println!("all comparisons within tolerance");
            ExitCode::SUCCESS
        }
        (_, true) => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

/// The census and timed CSVs share the recorder's leading columns. `m,n,k`
/// are the identity the recorder derives from the declared size spec (the
/// sentinel `1`); the call's own geometry is in the trailing columns.
const LEAD: &str = "regime,family,dtype,order,m,n,k,batch,row,threads,status,note";

fn lead(args: &Args, r: &Record) -> Vec<String> {
    vec![
        args.regime.clone(),
        r.family.clone(),
        r.dtype.into(),
        String::new(),
        r.n.clone(),
        r.n.clone(),
        String::new(),
        "1".into(),
        r.row.clone(),
        args.threads.to_string(),
        r.status.as_str().into(),
        r.note.clone(),
    ]
}

fn census_csv(args: &Args, records: &[Record]) -> String {
    let mut text = format!("{LEAD},route_observed,max_abs\n");
    for r in records {
        let mut fields = lead(args, r);
        fields.push(r.route.clone());
        fields.push(r.max_abs.map(|v| v.to_string()).unwrap_or_default());
        line(&mut text, &fields);
    }
    text
}

/// Header of the timed CSV; `export` reads these columns by name.
pub const TIMED_HEADER: &str = "regime,family,dtype,order,m,n,k,batch,row,threads,status,note,\
total_ms,per_item_us,samples_ms,rayon_threads,vendor_threads,case_m,case_n,case_k,capture,\
capture_sha256,job_m,job_n,job_k,route_observed,inner_calls";

fn timed_csv(args: &Args, records: &[Record]) -> String {
    let mut text = format!("{TIMED_HEADER}\n");
    for r in records {
        let ok = r.status == Status::Ok && !r.samples_ns.is_empty();
        let total_ms = replay::median(&r.samples_ns) / 1e6;
        let mut fields = lead(args, r);
        fields.extend([
            if ok {
                format!("{total_ms:.9}")
            } else {
                String::new()
            },
            if ok {
                format!("{:.9}", total_ms * 1e3)
            } else {
                String::new()
            },
            if ok {
                let ms: Vec<String> = r
                    .samples_ns
                    .iter()
                    .map(|v| format!("{:.12}", v / 1e6))
                    .collect();
                format!("[{}]", ms.join(","))
            } else {
                String::new()
            },
            args.threads.to_string(),
            args.threads.to_string(),
            // case_m/n/k: jobs, total m*n*k, output elements.
            r.jobs.to_string(),
            r.work.to_string(),
            r.out_len.to_string(),
            r.capture.clone(),
            r.sha256.clone(),
            r.dominant.0.to_string(),
            r.dominant.1.to_string(),
            r.dominant.2.to_string(),
            r.route.clone(),
            if ok {
                r.inner.to_string()
            } else {
                String::new()
            },
        ]);
        line(&mut text, &fields);
    }
    text
}

fn line(text: &mut String, fields: &[String]) {
    let quoted: Vec<String> = fields
        .iter()
        .map(|f| {
            if f.contains([',', '"', '\n']) {
                format!("\"{}\"", f.replace('"', "\"\""))
            } else {
                f.clone()
            }
        })
        .collect();
    let _ = writeln!(text, "{}", quoted.join(","));
}
