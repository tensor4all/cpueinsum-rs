//! `cpueinsum-bench` -- the timing-v2 benchmark harness over the
//! `tenferro-einsum-benchmark` instance corpus.
//!
//! Subcommands:
//!
//! * `run` -- time every declared cell, one measurement per declared
//!   contraction path (`opt_size` and `opt_flops`), writing the timed evidence
//!   CSV with the chronological per-call samples.
//! * `verify` -- run every declared cell once, cross-check the two rows, and
//!   write the census. Prints `all comparisons within tolerance` as its final
//!   line only when every comparison passed.
//! * `info` -- the machine, the `key=value` evidence keys, and nothing timed.
//!
//! This crate exists to publish the comparison between the corpus's two
//! declared contraction paths; cpueinsum supplies the integer-label, SSA-order
//! execution. A bad flag, an unavailable row, an unknown family or an empty
//! selection fails closed.

mod corpus;
mod diagnostics;
mod elem;
mod harness;
mod json;
mod report;

use std::path::Path;
use std::process::ExitCode;

use cpueinsum::{Exec, Pool};

use crate::corpus::{Instance, ENGINE_ROWS};
use crate::harness::{Config, Mode, Statistic};
use crate::report::{print_environment, print_evidence_keys};

/// Exit with a clear message; a bad flag must never be ignored.
fn die(message: &str) -> ExitCode {
    eprintln!("cpueinsum-bench: {message}");
    ExitCode::from(2)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Run,
    Verify,
    Info,
}

struct Args {
    command: Command,
    threads: Option<usize>,
    instances: Option<String>,
    n_specs: Vec<String>,
    families: Vec<String>,
    case: Option<String>,
    dtypes: Vec<String>,
    batches: Vec<u32>,
    rows: Vec<String>,
    engines: Vec<String>,
    rotate: usize,
    statistic: Statistic,
    reps: usize,
    prime_ms: u64,
    csv: Option<String>,
    census_csv: Option<String>,
    diagnostics_csv: Option<String>,
    reference_csv: Option<String>,
    regime: String,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse(&argv) {
        Ok(args) => args,
        Err(message) => return die(&message),
    };
    match args.command {
        Command::Info => match print_info(args.threads) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => die(&message),
        },
        Command::Run | Command::Verify => measure(&args),
    }
}

fn print_info(requested: Option<usize>) -> Result<(), String> {
    print_environment();
    let effective = match requested {
        Some(threads) => Some(observed_pool_size(threads)?),
        None => None,
    };
    print_evidence_keys(requested, effective);
    Ok(())
}

/// Refuse a request above Rayon's soft limit, which a pool build would silently
/// clamp.
fn check_rayon_limit(requested: usize) -> Result<(), String> {
    let max = rayon::max_num_threads();
    if requested > max {
        return Err(format!(
            "requested {requested} thread(s) but this Rayon build clamps a pool to {max}"
        ));
    }
    Ok(())
}

/// The worker count a pool of `requested` threads actually gets. Rayon
/// soft-limits a pool, so a request above that limit would be silently clamped
/// and must be refused instead of reported as effective.
fn observed_pool_size(requested: usize) -> Result<usize, String> {
    check_rayon_limit(requested)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(requested)
        .build()
        .map_err(|e| format!("cannot create a {requested}-worker pool: {e}"))?;
    let size = pool.current_num_threads();
    if size != requested {
        return Err(format!(
            "requested {requested} thread(s) but the rayon pool has {size} worker(s)"
        ));
    }
    Ok(size)
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let command = match argv.get(1).map(String::as_str) {
        Some("run") => Command::Run,
        Some("verify") => Command::Verify,
        Some("info") => Command::Info,
        other => return Err(format!("expected run, verify or info, got {other:?}")),
    };
    let mut args = Args {
        command,
        threads: None,
        instances: None,
        n_specs: Vec::new(),
        families: Vec::new(),
        case: None,
        dtypes: Vec::new(),
        batches: vec![1],
        rows: Vec::new(),
        engines: Vec::new(),
        rotate: 0,
        statistic: Statistic::Median,
        reps: 5,
        prime_ms: 250,
        csv: None,
        census_csv: None,
        diagnostics_csv: None,
        reference_csv: None,
        regime: "default".to_string(),
    };
    let mut rows_given = false;
    let mut engines_given = false;
    let mut i = 2;
    while i < argv.len() {
        let flag = argv[i].as_str();
        if command == Command::Info && flag != "--threads" {
            return Err(format!("info takes only --threads, got {flag:?}"));
        }
        match flag {
            "--threads" => args.threads = Some(parse_usize(&next(argv, i, flag)?, flag)?),
            "--instances" => args.instances = Some(next(argv, i, flag)?),
            "--n" => {
                args.n_specs = list(&next(argv, i, flag)?, flag)?;
                dedup(&mut args.n_specs);
            }
            "--family" => {
                args.families = list(&next(argv, i, flag)?, flag)?;
                dedup(&mut args.families);
            }
            "--case" => args.case = Some(next(argv, i, flag)?),
            "--dtype" => {
                args.dtypes = list(&next(argv, i, flag)?, flag)?;
                dedup(&mut args.dtypes);
            }
            "--batch" => {
                let mut batches = list(&next(argv, i, flag)?, flag)?
                    .iter()
                    .map(|v| v.parse::<u32>().map_err(|_| format!("bad --batch {v:?}")))
                    .collect::<Result<Vec<_>, _>>()?;
                dedup_values(&mut batches);
                args.batches = batches;
            }
            "--rows" => {
                args.rows = list(&next(argv, i, flag)?, flag)?;
                dedup(&mut args.rows);
                rows_given = true;
            }
            "--engines" => {
                args.engines = list(&next(argv, i, flag)?, flag)?;
                dedup(&mut args.engines);
                engines_given = true;
            }
            "--rotate-rows" => args.rotate = parse_usize(&next(argv, i, flag)?, flag)?,
            "--statistic" => {
                args.statistic = Statistic::parse(&next(argv, i, flag)?)
                    .ok_or_else(|| "--statistic must be best or median".to_string())?;
            }
            "--reps" => args.reps = parse_usize(&next(argv, i, flag)?, flag)?,
            "--prime-ms" => args.prime_ms = parse_u64(&next(argv, i, flag)?, flag)?,
            "--csv" => args.csv = Some(next(argv, i, flag)?),
            "--census-csv" => args.census_csv = Some(next(argv, i, flag)?),
            "--diagnostics-csv" => args.diagnostics_csv = Some(next(argv, i, flag)?),
            "--reference-csv" => args.reference_csv = Some(next(argv, i, flag)?),
            "--regime" => args.regime = next(argv, i, flag)?,
            // The corpus has fixed extents; the sibling harness accepts --size
            // so one campaign command line fits both, and this harness does too.
            "--size" => {
                let _ = next(argv, i, flag)?;
            }
            other => return Err(format!("unknown option {other:?}")),
        }
        // Every accepted flag takes exactly one value.
        i += 2;
    }
    if let Some(0) = args.threads {
        return Err("--threads must be positive".to_string());
    }
    if command == Command::Info {
        return Ok(args);
    }
    if args.reps == 0 {
        return Err("--reps must be positive".to_string());
    }
    if command == Command::Run && args.reps < 2 {
        return Err("--reps must be at least 2 for a timing-v2 run".to_string());
    }
    if args.batches.iter().any(|&b| b != 1) {
        return Err("--batch: this corpus has unit batch; every value must be 1".to_string());
    }
    let known = ENGINE_ROWS;
    for row in args.rows.iter().chain(args.engines.iter()) {
        if !known.contains(&row.as_str()) {
            return Err(format!(
                "unknown or unavailable row {row:?}; this binary carries {}",
                known.join(",")
            ));
        }
    }
    if rows_given {
        if engines_given {
            args.rows.retain(|row| args.engines.contains(row));
        }
    } else if engines_given {
        args.rows = args.engines.clone();
    } else {
        args.rows = known.iter().map(|s| (*s).to_string()).collect();
    }
    if args.rows.is_empty() {
        return Err("the selection names no engine row".to_string());
    }
    if command == Command::Verify && args.rows.len() < 2 && args.reference_csv.is_none() {
        return Err(
            "verify needs at least two engine rows: a single-row selection has no independent \
             result to compare against. Pass --reference-csv with another arm's diagnostics to \
             check a single row against an independent implementation"
                .to_string(),
        );
    }
    if args.command == Command::Run && args.csv.is_none() {
        return Err("run needs --csv PATH".to_string());
    }
    if args.command == Command::Verify && args.csv.is_none() && args.census_csv.is_none() {
        return Err("verify needs --csv PATH or --census-csv PATH".to_string());
    }
    Ok(args)
}

fn measure(args: &Args) -> ExitCode {
    let dir = args.instances.as_deref().unwrap_or("");
    if dir.is_empty() {
        return die("--instances DIR is required");
    }
    let loaded = match Instance::load_dir(Path::new(dir)) {
        Ok(instances) => instances,
        Err(message) => return die(&message),
    };
    let names: Vec<&str> = loaded.iter().map(|i| i.name.as_str()).collect();
    for family in &args.families {
        if !names.contains(&family.as_str()) {
            return die(&format!("unknown family {family:?}"));
        }
    }
    let selected: Vec<Instance> = loaded
        .into_iter()
        .filter(|instance| {
            (args.families.is_empty() || args.families.contains(&instance.name))
                && args
                    .case
                    .as_deref()
                    .is_none_or(|c| instance.name.contains(c))
        })
        .collect();
    if selected.is_empty() {
        return die("the selection names no instance");
    }

    let dtypes = if args.dtypes.is_empty() {
        harness::corpus_dtypes(&selected)
    } else {
        args.dtypes.clone()
    };
    let n_specs = if args.n_specs.is_empty() {
        vec!["1".to_string()]
    } else {
        args.n_specs.clone()
    };

    // The declared rows in execution order: the selection rotated. The reported
    // row name is unchanged; only the schedule moves.
    let mut rows = args.rows.clone();
    if !rows.is_empty() {
        let len = rows.len();
        rows.rotate_left(args.rotate % len);
    }

    // Build the pool and the executor first, then report the counts the runtime
    // actually granted, never the request. A clamped pool or budget fails.
    let requested = args.threads.unwrap_or(1);
    if let Err(message) = check_rayon_limit(requested) {
        return die(&message);
    }
    let thread_pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(requested)
        .build()
    {
        Ok(pool) => pool,
        Err(e) => return die(&format!("cannot create a {requested}-worker pool: {e}")),
    };
    let pool = Pool::borrow(&thread_pool);
    let rayon_threads = pool.size();
    let exec = match Exec::rayon(&pool).with_budget(requested) {
        Ok(exec) => exec,
        Err(e) => return die(&format!("cannot set a {requested}-thread budget: {e}")),
    };
    let vendor_threads = exec.budget();
    if let Err(message) = harness::thread_readback(requested, rayon_threads, vendor_threads) {
        return die(&message);
    }

    let reference = match args.reference_csv.as_deref() {
        Some(path) => match diagnostics::read(path) {
            Ok(map) => Some(map),
            Err(message) => return die(&message),
        },
        None => None,
    };

    let config = Config {
        regime: args.regime.clone(),
        n_specs,
        dtypes,
        batches: args.batches.clone(),
        rows,
        statistic: args.statistic,
        reps: args.reps,
        prime_ms: args.prime_ms,
        threads: requested,
        rayon_threads,
        vendor_threads,
        mode: match args.command {
            Command::Verify => Mode::Verify,
            _ => Mode::Run,
        },
        reference,
    };

    let (records, all_ok, compared) = harness::run_grid(&selected, &config, &exec);
    // A verify run that made no cross-row comparison checked nothing; refuse it
    // even though every declared cell is "ok" or "skipped".
    if args.command == Command::Verify && !compared {
        return die(
            "verify made no comparison: every declared row was skipped or only one reached \
             execution, so there is no second independent result to check",
        );
    }
    println!(
        "{}: {} declared cell(s), {} ok, {} skipped, {} failed",
        match args.command {
            Command::Verify => "verify",
            _ => "run",
        },
        records.len(),
        records
            .iter()
            .filter(|r| r.status == harness::Status::Ok)
            .count(),
        records
            .iter()
            .filter(|r| r.status == harness::Status::Skipped)
            .count(),
        records
            .iter()
            .filter(|r| r.status == harness::Status::Failed)
            .count(),
    );
    if args.command == Command::Verify && !all_ok {
        for record in records
            .iter()
            .filter(|r| r.status == harness::Status::Failed)
        {
            println!(
                "MISMATCH {} {} {} {} {}",
                record.row, record.family, record.dtype, record.n, record.note
            );
        }
    }

    if let Some(path) = args.diagnostics_csv.as_deref() {
        if let Err(e) = diagnostics::write(path, &records) {
            return die(&format!("cannot write the diagnostics to {path}: {e}"));
        }
    }
    if let Err(message) = write_evidence(args, &records) {
        return die(&message);
    }
    if args.command == Command::Verify {
        if all_ok {
            println!("all comparisons within tolerance");
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    } else if all_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn write_evidence(args: &Args, records: &[harness::Record]) -> Result<(), String> {
    let write = |path: &Option<String>, census: bool| -> Result<(), String> {
        if let Some(path) = path {
            let result = if census {
                harness::write_census(path, records)
            } else {
                harness::write_timed(path, records)
            };
            result.map_err(|e| format!("{path}: {e}"))?;
            println!("wrote {path}");
        }
        Ok(())
    };
    // verify writes the census wherever its caller points it; run's `--csv` is
    // the timed evidence and `--census-csv`, when given, is the census.
    let verify = args.command == Command::Verify;
    write(&args.csv, verify)?;
    write(&args.census_csv, true)?;
    Ok(())
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// A list-valued flag must name at least one value; an explicitly empty request
/// is refused rather than silently becoming a default or an empty selection.
fn list(value: &str, flag: &str) -> Result<Vec<String>, String> {
    let values = split_list(value);
    if values.is_empty() {
        return Err(format!("{flag} needs at least one value"));
    }
    Ok(values)
}

/// Collapse repeated values, keeping the first occurrence, so a repeated flag
/// declares each cell exactly once.
fn dedup(values: &mut Vec<String>) {
    let mut seen = std::collections::BTreeSet::new();
    values.retain(|v| seen.insert(v.clone()));
}

fn dedup_values<T: Ord + Copy>(values: &mut Vec<T>) {
    let mut seen = std::collections::BTreeSet::new();
    values.retain(|v| seen.insert(*v));
}

fn next(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} {value:?}: expected a whole number"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{flag} {value:?}: expected a whole number"))
}
