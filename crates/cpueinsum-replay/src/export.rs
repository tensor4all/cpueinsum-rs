//! `export`: one host's timed CSVs folded into one JSON file for the route fit.
//!
//! ```text
//! cpueinsum-replay export --corpus-dir DIR --host-class CLASS --campaign-commit SHA \
//!     --out FILE.json CSV...
//! ```
//!
//! Every `ok` row of a timed CSV (the `run` output) becomes one row
//! `call_id, route, width, host_class, dtype, m, n, k, measured_ns` (plus the
//! samples, the observed route and the source file). `m, n, k` are the call's
//! largest job; the full job list is the call's corpus record, named by
//! `call_id`. The metadata carries the per-call summary of every replayed
//! capture, the captures' SHA-256, the campaign commit, which capture each
//! width replayed, and the exactness argument of the checked twin with the
//! bound on |value| and, when census CSVs (`verify` output) are passed too,
//! the largest |value| verify observed.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use crate::corpus::{self, Capture};
use crate::replay;

/// The export's own format tag.
pub const FORMAT: &str = "cpueinsum-replay-export/1";

struct Row {
    call_id: String,
    route: String,
    width: usize,
    dtype: String,
    mnk: [usize; 3],
    samples_ns: Vec<f64>,
    route_observed: String,
    inner_calls: u64,
    source: String,
}

/// Parse the arguments, read every CSV and write the JSON file.
pub fn main(argv: &[String]) -> Result<(), String> {
    let (mut dir, mut host, mut commit, mut out) = (None, None, None, None);
    let mut csvs = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let slot = match argv[i].as_str() {
            "--corpus-dir" => &mut dir,
            "--host-class" => &mut host,
            "--campaign-commit" => &mut commit,
            "--out" => &mut out,
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag:?}")),
            path => {
                csvs.push(path.to_string());
                i += 1;
                continue;
            }
        };
        *slot = Some(
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", argv[i]))?,
        );
        i += 2;
    }
    let need = |v: Option<String>, flag: &str| v.ok_or(format!("export needs {flag}"));
    let (dir, host, commit, out) = (
        need(dir, "--corpus-dir")?,
        need(host, "--host-class")?,
        need(commit, "--campaign-commit")?,
        need(out, "--out")?,
    );
    if csvs.is_empty() {
        return Err("export needs at least one timed CSV".into());
    }
    let captures = corpus::load_dir(Path::new(&dir))?;
    let mut rows = Vec::new();
    let mut observed_max: Option<f64> = None;
    for path in &csvs {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        read_csv(&text, path, &captures, &mut rows, &mut observed_max)
            .map_err(|e| format!("{path}: {e}"))?;
    }
    if rows.is_empty() {
        return Err("the CSVs hold no ok timed row".into());
    }
    let json = render(&captures, &rows, &host, &commit, observed_max)?;
    std::fs::write(&out, json).map_err(|e| format!("{out}: {e}"))?;
    println!("wrote {out}: {} row(s)", rows.len());
    Ok(())
}

fn read_csv(
    text: &str,
    path: &str,
    captures: &[Capture],
    rows: &mut Vec<Row>,
    observed_max: &mut Option<f64>,
) -> Result<(), String> {
    let mut lines = text.lines();
    let header = split(lines.next().ok_or("empty CSV")?);
    let col = |name: &str| header.iter().position(|h| h == name);
    let get = |fields: &[String], name: &str| -> Result<String, String> {
        col(name)
            .and_then(|i| fields.get(i).cloned())
            .ok_or(format!("no column {name}"))
    };
    let census = col("max_abs").is_some();
    if !census && header.join(",") != crate::TIMED_HEADER {
        return Err("neither a timed CSV of this harness nor a census".into());
    }
    let source = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    for line in lines.filter(|l| !l.is_empty()) {
        let f = split(line);
        if f.len() != header.len() {
            return Err(format!(
                "{} fields under a {}-column header",
                f.len(),
                header.len()
            ));
        }
        if get(&f, "status")? != "ok" {
            continue;
        }
        if census {
            if let Ok(v) = get(&f, "max_abs")?.parse::<f64>() {
                *observed_max = Some(observed_max.map_or(v, |m| m.max(v)));
            }
            continue;
        }
        let stem = get(&f, "capture")?;
        let capture = captures
            .iter()
            .find(|c| c.stem == stem)
            .ok_or(format!("capture {stem} is not in the corpus"))?;
        if capture.sha256 != get(&f, "capture_sha256")? {
            return Err(format!("capture {stem} changed since it was timed"));
        }
        let samples = get(&f, "samples_ms")?;
        let samples_ns = samples
            .trim_matches(['[', ']'])
            .split(',')
            .map(|v| v.parse::<f64>().map(|ms| ms * 1e6))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| format!("bad samples_ms {samples:?}"))?;
        rows.push(Row {
            call_id: get(&f, "family")?,
            route: get(&f, "row")?,
            width: number(&get(&f, "threads")?)?,
            dtype: get(&f, "dtype")?,
            mnk: [
                number(&get(&f, "job_m")?)?,
                number(&get(&f, "job_n")?)?,
                number(&get(&f, "job_k")?)?,
            ],
            samples_ns,
            route_observed: get(&f, "route_observed")?,
            inner_calls: number(&get(&f, "inner_calls")?)? as u64,
            source: source.clone(),
        });
    }
    Ok(())
}

fn number(v: &str) -> Result<usize, String> {
    v.parse()
        .map_err(|_| format!("{v:?} is not a whole number"))
}

/// One CSV line as written by this harness (RFC 4180 quoting).
pub fn split(line: &str) -> Vec<String> {
    let mut fields = vec![String::new()];
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                chars.next();
                fields.last_mut().unwrap().push('"');
            }
            ('"', _) => quoted = !quoted,
            (',', false) => fields.push(String::new()),
            (c, _) => fields.last_mut().unwrap().push(c),
        }
    }
    fields
}

fn esc(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".into()
    }
}

fn render(
    captures: &[Capture],
    rows: &[Row],
    host: &str,
    commit: &str,
    observed_max: Option<f64>,
) -> Result<String, String> {
    let mut width_capture: BTreeMap<usize, BTreeSet<&str>> = BTreeMap::new();
    for row in rows {
        let stem = row.call_id.rsplit_once(':').map_or("", |(s, _)| s);
        width_capture.entry(row.width).or_default().insert(stem);
    }
    let mut used = BTreeSet::new();
    let mut widths = Vec::new();
    for (w, stems) in &width_capture {
        if stems.len() != 1 {
            return Err(format!("width {w} replayed several captures: {stems:?}"));
        }
        let stem = *stems.iter().next().expect("one stem");
        used.insert(stem);
        widths.push(format!("{}: {}", esc(&w.to_string()), esc(stem)));
    }
    let used: Vec<&Capture> = captures
        .iter()
        .filter(|c| used.contains(c.stem.as_str()))
        .collect();
    let k_max = used
        .iter()
        .flat_map(|c| {
            c.calls
                .iter()
                .flat_map(|call| call.jobs.iter().map(|j| j[5]))
        })
        .max()
        .unwrap_or(0);
    let bound = replay::PRODUCT_BOUND * k_max as f64;
    let limit = 9_007_199_254_740_992.0; // 2^53
    if bound >= limit {
        return Err(format!(
            "|value| bound {bound} reaches 2^53; the twin is not exact"
        ));
    }

    let mut s = String::new();
    let _ = writeln!(s, "{{\n  \"format\": {},\n  \"metadata\": {{", esc(FORMAT));
    let _ = writeln!(s, "    \"campaign_commit\": {},", esc(commit));
    let _ = writeln!(s, "    \"host_class\": {},", esc(host));
    let _ = writeln!(s, "    \"harness\": \"cpueinsum-replay\",");
    let _ = writeln!(s, "    \"cpueinsum_version\": {},", esc(cpueinsum::VERSION));
    let _ = writeln!(s, "    \"width_capture\": {{{}}},", widths.join(", "));
    let _ = writeln!(s, "    \"captures\": [");
    for (i, c) in used.iter().enumerate() {
        let _ = writeln!(
            s,
            "      {{\"stem\": {}, \"file\": {}, \"sha256\": {}, \"format\": {}, \"symtn_commit\": {}, \
             \"symtn_base\": {}, \"provider\": {}, \"engine_threads\": {}, \"records\": {}}}{}",
            esc(&c.stem),
            esc(&c.path.file_name().unwrap_or_default().to_string_lossy()),
            esc(&c.sha256),
            esc(corpus::FORMAT),
            esc(&c.symtn_commit),
            esc(&c.symtn_base),
            esc(&c.provider),
            c.engine_threads,
            c.calls.len(),
            if i + 1 < used.len() { "," } else { "" }
        );
    }
    let _ = writeln!(s, "    ],");
    let _ = writeln!(
        s,
        "    \"exactness\": {{\"comparison\": {}, \"argument\": {}, \"max_abs_input\": 3, \
         \"max_abs_product_component\": {}, \"k_max\": {k_max}, \"max_abs_value_bound\": {}, \
         \"exact_integer_limit\": {}, \"max_abs_value_observed\": {}}},",
        esc("every route against sequential at the same width, sequential against the definition; IEEE equality"),
        esc(replay::EXACTNESS),
        num(replay::PRODUCT_BOUND),
        num(bound),
        num(limit),
        observed_max.map_or("null".into(), num),
    );
    let _ = writeln!(
        s,
        "    \"definitions\": {{\"m_n_k\": {}, \"work\": {}, \"work_weighted_median\": {}, \
         \"largest_job_share\": {}, \"run_coverage\": {}, \"measured_ns\": {}}},",
        esc("m, n, k of the call's largest job by m*n*k, the first on a tie"),
        esc("sum over jobs of m*n*k, not scaled for complex"),
        esc("the job m*n*k at which the cumulative work, jobs sorted by m*n*k, first reaches half"),
        esc("max job m*n*k / work"),
        esc("fraction of the jobs inside a recorded strided run of length >= 2; null for a call without a recorded partition (fresh calls)"),
        esc("median of the samples; a sample is the wall time per call of back-to-back calls lasting at least 1 ms"),
    );
    let _ = writeln!(s, "    \"calls\": [");
    let total: usize = used.iter().map(|c| c.calls.len()).sum();
    let mut n = 0;
    for c in &used {
        for call in &c.calls {
            n += 1;
            let m = call.summary();
            let runs: Vec<String> = call.runs.iter().map(|r| r.to_string()).collect();
            let _ = writeln!(
                s,
                "      {{\"call_id\": {}, \"capture\": {}, \"label\": {}, \"dtype\": {}, \
                 \"multiplicity\": {}, \"fresh\": {}, \"jobs\": {}, \"distinct_geometries\": {}, \
                 \"work\": {}, \"work_weighted_median\": {}, \"largest_job_share\": {}, \
                 \"run_coverage\": {}, \"runs\": [{}]}}{}",
                esc(&c.family(call.index)),
                esc(&c.stem),
                esc(&call.label),
                esc(call.dtype.label()),
                call.multiplicity,
                call.beta.is_none(),
                m.jobs,
                m.distinct_geometries,
                m.work,
                m.work_weighted_median,
                num(m.largest_job_share),
                m.run_coverage.map_or("null".into(), num),
                runs.join(", "),
                if n < total { "," } else { "" }
            );
        }
    }
    let _ = writeln!(s, "    ]\n  }},\n  \"rows\": [");
    for (i, r) in rows.iter().enumerate() {
        let samples: Vec<String> = r.samples_ns.iter().map(|&v| num(v)).collect();
        let _ = writeln!(
            s,
            "    {{\"call_id\": {}, \"route\": {}, \"width\": {}, \"host_class\": {}, \"dtype\": {}, \
             \"m\": {}, \"n\": {}, \"k\": {}, \"measured_ns\": {}, \"samples_ns\": [{}], \
             \"route_observed\": {}, \"inner_calls\": {}, \"source\": {}}}{}",
            esc(&r.call_id),
            esc(&r.route),
            r.width,
            esc(host),
            esc(&r.dtype),
            r.mnk[0],
            r.mnk[1],
            r.mnk[2],
            num(replay::median(&r.samples_ns)),
            samples.join(", "),
            esc(&r.route_observed),
            r.inner_calls,
            esc(&r.source),
            if i + 1 < rows.len() { "," } else { "" }
        );
    }
    s.push_str("  ]\n}\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    #[test]
    fn splits_the_quoting_the_harness_writes() {
        assert_eq!(
            super::split(r#"a,"[1,2]","say ""x""",,"#),
            ["a", "[1,2]", r#"say "x""#, "", ""]
        );
    }
}
