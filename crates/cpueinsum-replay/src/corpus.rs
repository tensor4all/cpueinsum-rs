//! Loading a `symtn-gemm-trace/1` capture and summarising its calls.
//!
//! A capture is one gzip JSON-lines file: a provenance line, then one line per
//! distinct grouped call `C_j = alpha op0(A_j) op1(B_j) + beta C_j`, with jobs
//! `[a_offset, b_offset, c_offset, m, n, k, lda, ldb, ldc]` in column-major
//! form (the tensor4all/tcontract-rs#209 descriptor). The format is documented
//! next to the corpus, in `gemm-trace-corpus/README.md` of the capture branch.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use cpueinsum::GroupedGemmJob;
use cpueinsum_bench::json::{self, Json};
use sha2::{Digest, Sha256};

/// The one format this harness reads.
pub const FORMAT: &str = "symtn-gemm-trace/1";
const JOB_FIELDS: [&str; 9] = [
    "a_offset", "b_offset", "c_offset", "m", "n", "k", "lda", "ldb", "ldc",
];

/// The element type a call declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// `F64` in the corpus, `float64` on the command line.
    F64,
    /// `C64` in the corpus, `complex128` on the command line.
    C64,
}

impl Dtype {
    /// The campaign's dtype label.
    pub fn label(self) -> &'static str {
        match self {
            Dtype::F64 => "float64",
            Dtype::C64 => "complex128",
        }
    }

    /// The type a command-line label names.
    pub fn of_label(label: &str) -> Option<Dtype> {
        match label {
            "float64" | "f64" => Some(Dtype::F64),
            "complex128" | "c64" => Some(Dtype::C64),
            _ => None,
        }
    }
}

/// One recorded call.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// Position among the capture's call lines, from 0.
    pub index: usize,
    /// Element type.
    pub dtype: Dtype,
    /// Identical calls this record stands for.
    pub multiplicity: u64,
    /// `model chi=<states> <scalar> <step>`, for the export only.
    pub label: String,
    /// Operand ops, one of `N`/`T`/`C` each.
    pub ops: Vec<String>,
    /// `[re, im]`.
    pub alpha: [f64; 2],
    /// `[re, im]`, or `None` for a fresh (never read) destination.
    pub beta: Option<[f64; 2]>,
    /// The caller's strided-run partition; empty for fresh calls.
    pub runs: Vec<usize>,
    /// `[a_off, b_off, c_off, m, n, k, lda, ldb, ldc]`.
    pub jobs: Vec<[usize; 9]>,
}

/// One capture file.
#[derive(Debug)]
pub struct Capture {
    /// The file name without `.jsonl.gz`; families are `<stem>:rNNN`.
    pub stem: String,
    /// Where it was read from.
    pub path: PathBuf,
    /// SHA-256 of the compressed file, lower-case hex.
    pub sha256: String,
    /// The engine width the calls were captured at.
    pub engine_threads: usize,
    /// The capturing symtn commit.
    pub symtn_commit: String,
    /// The symtn base the capture branch is on.
    pub symtn_base: String,
    /// The capture's dense provider.
    pub provider: String,
    /// The recorded calls in file order.
    pub calls: Vec<Call>,
}

impl Capture {
    /// The family name of call `index`.
    pub fn family(&self, index: usize) -> String {
        format!("{}:r{index:03}", self.stem)
    }
}

/// Every family whose call has `dtype`, in name order: the default family list
/// of that dtype, which declares no dtype-mismatch cell.
pub fn families(captures: &[Capture], dtype: Dtype) -> Vec<String> {
    let mut out: Vec<String> = captures
        .iter()
        .flat_map(|c| {
            c.calls
                .iter()
                .filter(|call| call.dtype == dtype)
                .map(|call| c.family(call.index))
        })
        .collect();
    out.sort();
    out
}

/// Every `*.jsonl.gz` file of `dir`, sorted by name.
pub fn load_dir(dir: &Path) -> Result<Vec<Capture>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".jsonl.gz"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("{}: no .jsonl.gz capture", dir.display()));
    }
    paths.iter().map(|p| load(p)).collect()
}

/// Read and check one capture file.
pub fn load(path: &Path) -> Result<Capture, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let stem = name.strip_suffix(".jsonl.gz").unwrap_or(name).to_string();
    let sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    parse(&text, stem, path.to_path_buf(), sha256).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse the decompressed text of a capture.
pub fn parse(text: &str, stem: String, path: PathBuf, sha256: String) -> Result<Capture, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let head = json::parse(lines.next().ok_or("empty capture")?)?;
    if head.get("kind").and_then(Json::as_str) != Some("provenance") {
        return Err("the first line is not the provenance record".into());
    }
    let format = head.get("format").and_then(Json::as_str);
    if format != Some(FORMAT) {
        return Err(format!("format {format:?}, this harness reads {FORMAT:?}"));
    }
    let fields: Vec<&str> = head
        .get("job_fields")
        .and_then(Json::as_arr)
        .unwrap_or_default()
        .iter()
        .filter_map(Json::as_str)
        .collect();
    if fields != JOB_FIELDS {
        return Err(format!("job_fields {fields:?}, expected {JOB_FIELDS:?}"));
    }
    if head.get("layout").and_then(Json::as_str) != Some("column-major") {
        return Err("layout must be column-major".into());
    }
    let text_field = |key: &str| {
        head.get(key)
            .and_then(Json::as_str)
            .map(str::to_string)
            .ok_or(format!("provenance has no {key}"))
    };
    let mut capture = Capture {
        stem,
        path,
        sha256,
        engine_threads: whole(head.get("engine_threads"), "engine_threads")?,
        symtn_commit: text_field("symtn_commit")?,
        symtn_base: text_field("symtn_base")?,
        provider: text_field("provider")?,
        calls: Vec::new(),
    };
    for (index, line) in lines.enumerate() {
        let call =
            parse_call(&json::parse(line)?, index).map_err(|e| format!("call {index}: {e}"))?;
        capture.calls.push(call);
    }
    Ok(capture)
}

fn whole(value: Option<&Json>, what: &str) -> Result<usize, String> {
    match value.and_then(Json::as_f64) {
        Some(v) if v >= 0.0 && v.fract() == 0.0 && v < 9.0e15 => Ok(v as usize),
        _ => Err(format!("{what} is not a whole number")),
    }
}

fn pair(value: &Json, what: &str) -> Result<[f64; 2], String> {
    match value.as_arr() {
        Some([re, im]) => match (re.as_f64(), im.as_f64()) {
            (Some(re), Some(im)) => Ok([re, im]),
            _ => Err(format!("{what} is not [re, im]")),
        },
        _ => Err(format!("{what} is not [re, im]")),
    }
}

fn parse_call(v: &Json, index: usize) -> Result<Call, String> {
    if v.get("kind").and_then(Json::as_str) != Some("call") {
        return Err("kind is not call".into());
    }
    let dtype = match v.get("dtype").and_then(Json::as_str) {
        Some("F64") => Dtype::F64,
        Some("C64") => Dtype::C64,
        other => return Err(format!("unsupported dtype {other:?}")),
    };
    let text = |key: &str| v.get(key).and_then(Json::as_str).unwrap_or("?").to_string();
    let label = format!(
        "{} chi={} {} {}",
        text("model"),
        v.get("chi_states").and_then(Json::as_f64).unwrap_or(0.0),
        text("scalar"),
        text("step"),
    );
    let ops = v
        .get("ops")
        .and_then(Json::as_arr)
        .ok_or("no ops")?
        .iter()
        .map(|o| o.as_str().map(str::to_string).ok_or("op is not a string"))
        .collect::<Result<Vec<_>, _>>()?;
    let alpha = pair(v.get("alpha").ok_or("no alpha")?, "alpha")?;
    let beta = match v.get("beta").ok_or("no beta")? {
        Json::Null => None,
        b => Some(pair(b, "beta")?),
    };
    let runs = v
        .get("runs")
        .and_then(Json::as_arr)
        .ok_or("no runs")?
        .iter()
        .map(|r| whole(Some(r), "a run length"))
        .collect::<Result<Vec<_>, _>>()?;
    let jobs = v
        .get("jobs")
        .and_then(Json::as_arr)
        .ok_or("no jobs")?
        .iter()
        .map(|job| {
            let fields = job
                .as_arr()
                .filter(|f| f.len() == 9)
                .ok_or("a job is not 9 fields")?;
            let mut out = [0usize; 9];
            for (slot, field) in out.iter_mut().zip(fields) {
                *slot = whole(Some(field), "a job field")?;
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Call {
        index,
        dtype,
        multiplicity: whole(v.get("multiplicity"), "multiplicity")? as u64,
        label,
        ops,
        alpha,
        beta,
        runs,
        jobs,
    })
}

/// The per-call features the export carries. Work is `m*n*k` per job, the
/// same measure the corpus README's summary uses (not scaled for complex).
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// Number of jobs.
    pub jobs: usize,
    /// Distinct `(m, n, k)`.
    pub distinct_geometries: usize,
    /// `sum m*n*k`.
    pub work: u128,
    /// The job size at which the cumulative work, jobs sorted by size, first
    /// reaches half the total.
    pub work_weighted_median: u128,
    /// `max m*n*k / sum m*n*k` (0 for an all-empty call).
    pub largest_job_share: f64,
    /// Fraction of the work inside a recorded strided run of length at least 2
    /// (the runs partition the jobs in order); `None` when the call records no
    /// partition (fresh calls) or has no work.
    pub run_coverage: Option<f64>,
    /// The same fraction counted in jobs instead of work; `None` without a
    /// partition.
    pub run_job_coverage: Option<f64>,
    /// `(m, n, k)` of the largest job (the first one on a tie).
    pub dominant: (usize, usize, usize),
}

impl Call {
    /// Summarise the job list.
    pub fn summary(&self) -> Summary {
        let work: Vec<u128> = self
            .jobs
            .iter()
            .map(|j| j[3] as u128 * j[4] as u128 * j[5] as u128)
            .collect();
        let total: u128 = work.iter().sum();
        let mut sorted = work.clone();
        sorted.sort_unstable();
        let mut acc = 0u128;
        let mut median = 0;
        for w in sorted {
            acc += w;
            median = w;
            if 2 * acc >= total {
                break;
            }
        }
        let mut d = 0;
        for i in 1..work.len() {
            if work[i] > work[d] {
                d = i;
            }
        }
        let best = work.get(d).copied().unwrap_or(0);
        let dominant = self.jobs.get(d).map_or((0, 0, 0), |j| (j[3], j[4], j[5]));
        let geometries: BTreeSet<_> = self.jobs.iter().map(|j| (j[3], j[4], j[5])).collect();
        let in_runs: usize = self.runs.iter().filter(|&&r| r >= 2).sum();
        let (mut start, mut work_in_runs) = (0, 0u128);
        for &r in &self.runs {
            if r >= 2 {
                work_in_runs += work.iter().skip(start).take(r).sum::<u128>();
            }
            start += r;
        }
        Summary {
            jobs: self.jobs.len(),
            distinct_geometries: geometries.len(),
            work: total,
            work_weighted_median: median,
            largest_job_share: if total == 0 {
                0.0
            } else {
                best as f64 / total as f64
            },
            run_coverage: (!self.runs.is_empty() && total > 0)
                .then(|| work_in_runs as f64 / total as f64),
            run_job_coverage: (!self.runs.is_empty() && !self.jobs.is_empty())
                .then(|| in_runs as f64 / self.jobs.len() as f64),
            dominant,
        }
    }

    /// The jobs as cpueinsum descriptors, or why this call cannot be replayed.
    ///
    /// `GroupedGemmJob` is compact column-major with no leading dimension and
    /// no operand op, and the checked twin relies on `alpha = 1` and
    /// `beta in {0, fresh}` (see `replay::EXACTNESS`). None of the refusals
    /// occurs in the current corpus; they are reported as skipped cells, never
    /// dropped.
    pub fn admit(&self) -> Result<Vec<GroupedGemmJob>, String> {
        if self.ops.iter().any(|o| o != "N") {
            return Err(format!("ops {:?}: only N,N is replayed", self.ops));
        }
        if self.alpha != [1.0, 0.0] {
            return Err(format!(
                "alpha {:?}: only alpha = 1 is replayed",
                self.alpha
            ));
        }
        if self.beta.is_some_and(|b| b != [0.0, 0.0]) {
            return Err(format!(
                "beta {:?}: only beta = 0 or fresh is replayed",
                self.beta
            ));
        }
        self.jobs
            .iter()
            .map(|&[a, b, c, m, n, k, lda, ldb, ldc]| {
                if lda != m || ldb != k || ldc != m {
                    return Err(format!(
                        "non-compact job (m={m} k={k} lda={lda} ldb={ldb} ldc={ldc}): \
                         GroupedGemmJob has no leading dimension"
                    ));
                }
                // The record gives n before k; GroupedGemmJob takes contracted before cols.
                Ok(GroupedGemmJob::new(a, b, c, m, k, n))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
