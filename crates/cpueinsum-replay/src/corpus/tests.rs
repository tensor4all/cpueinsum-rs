use super::*;

const HEAD: &str = r#"{"kind":"provenance","format":"symtn-gemm-trace/1","symtn_commit":"abc","symtn_base":"def","provider":"cpu-faer","engine_threads":4,"workload":"w","job_fields":["a_offset","b_offset","c_offset","m","n","k","lda","ldb","ldc"],"layout":"column-major"}"#;

fn capture(calls: &[&str]) -> Result<Capture, String> {
    let text = std::iter::once(HEAD)
        .chain(calls.iter().copied())
        .collect::<Vec<_>>()
        .join("\n");
    parse(&text, "t".into(), PathBuf::from("t.jsonl.gz"), "00".into())
}

// Jobs of work 2*3*1 = 6, 4*5*6 = 120 and 4*5*6 = 120; runs [1, 2].
const SKEWED: &str = r#"{"kind":"call","model":"u1","chi_states":64,"scalar":"f64","step":"s","multiplicity":3,"dtype":"F64","ops":["N","N"],"alpha":[1.0,0.0],"beta":[0.0,0.0],"threads":4,"runs":[1, 2],"jobs":[[0,0,0,2,3,1,2,1,2],[2,3,6,4,5,6,4,6,4],[26,33,26,4,5,6,4,6,4]]}"#;

#[test]
fn reads_a_call_and_maps_n_before_k() {
    let c = capture(&[SKEWED]).unwrap();
    assert_eq!((c.engine_threads, c.family(0).as_str()), (4, "t:r000"));
    let call = &c.calls[0];
    assert_eq!(
        (call.dtype, call.multiplicity, call.beta),
        (Dtype::F64, 3, Some([0.0, 0.0]))
    );
    // [a, b, c, m, n, k, ...] -> GroupedGemmJob::new(a, b, c, rows = m, contracted = k, cols = n)
    assert_eq!(
        call.admit().unwrap()[1],
        GroupedGemmJob::new(2, 3, 6, 4, 6, 5)
    );
}

#[test]
fn summarises_skew_geometry_and_runs() {
    let s = capture(&[SKEWED]).unwrap().calls[0].summary();
    assert_eq!((s.jobs, s.distinct_geometries, s.work), (3, 2, 246));
    // Sorted work 6, 120, 120: the cumulative sum first reaches 123 at the second job.
    assert_eq!(s.work_weighted_median, 120);
    assert_eq!(s.largest_job_share, 120.0 / 246.0);
    // Runs [1, 2]: jobs 1 and 2 are in a run, 240 of the 246 work.
    assert_eq!(s.run_coverage, Some(240.0 / 246.0));
    assert_eq!(s.run_job_coverage, Some(2.0 / 3.0));
    assert_eq!(s.dominant, (4, 5, 6));
}

#[test]
fn a_fresh_call_has_no_run_coverage() {
    let fresh = SKEWED
        .replace(r#""beta":[0.0,0.0]"#, r#""beta":null"#)
        .replace("[1, 2]", "[]");
    let call = &capture(&[&fresh]).unwrap().calls[0];
    let s = call.summary();
    assert_eq!(
        (call.beta, s.run_coverage, s.run_job_coverage),
        (None, None, None)
    );
}

#[test]
fn default_families_group_by_dtype() {
    let c64 = SKEWED.replace(r#""dtype":"F64""#, r#""dtype":"C64""#);
    let c = [capture(&[&c64, SKEWED, &c64]).unwrap()];
    assert_eq!(families(&c, Dtype::F64), ["t:r001"]);
    assert_eq!(families(&c, Dtype::C64), ["t:r000", "t:r002"]);
}

#[test]
fn refuses_what_the_twin_cannot_replay() {
    for (from, to, why) in [
        (r#"["N","N"]"#, r#"["N","T"]"#, "ops"),
        (r#""alpha":[1.0,0.0]"#, r#""alpha":[2.0,0.0]"#, "alpha"),
        (r#""beta":[0.0,0.0]"#, r#""beta":[1.0,0.0]"#, "beta"),
        ("[0,0,0,2,3,1,2,1,2]", "[0,0,0,2,3,1,3,1,2]", "non-compact"),
    ] {
        let call = &capture(&[&SKEWED.replace(from, to)]).unwrap().calls[0];
        assert!(call.admit().unwrap_err().contains(why), "{why}");
    }
}

#[test]
fn refuses_another_format() {
    let text = HEAD.replace("symtn-gemm-trace/1", "symtn-gemm-trace/2");
    let err = parse(&text, "t".into(), PathBuf::new(), String::new()).unwrap_err();
    assert!(err.contains("symtn-gemm-trace/2"), "{err}");
}
