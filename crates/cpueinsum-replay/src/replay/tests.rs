use super::*;

/// Three jobs over shared buffers: a skewed pair and an empty-k-free tail, with
/// offsets that are not multiples of the fill periods.
fn jobs() -> Vec<GroupedGemmJob> {
    vec![
        GroupedGemmJob::new(0, 0, 0, 3, 2, 4),
        GroupedGemmJob::new(6, 8, 12, 5, 3, 2),
        GroupedGemmJob::new(1, 3, 22, 2, 1, 3),
    ]
}

#[test]
fn naive_is_the_definition() {
    let (a, b): (Vec<f64>, Vec<f64>) = (
        (0..21).map(f64::lhs).collect(),
        (0..14).map(f64::rhs).collect(),
    );
    let out = naive(&jobs(), &a, &b, 28);
    // Job 1, element (i = 2, j = 1): sum_p A[6 + 2 + 5p] B[8 + p + 3*1].
    let want: f64 = (0..3).map(|p| a[8 + 5 * p] * b[11 + p]).sum();
    assert_eq!(out[12 + 2 + 5], want);
    assert!(out.iter().all(|v| v.is_finite()), "every slot is written");
}

#[test]
fn every_route_matches_the_definition_exactly() {
    let pool = threadteam::Pool::builder().threads(2).build().unwrap();
    let team = pool.into_team(2).unwrap();
    let mut exec = Exec::from_token(team.exec(2), None, BatchPolicy::default());
    for fresh in [false, true] {
        let mut real = Cell::<f64>::new(jobs(), fresh).unwrap();
        let mut complex = Cell::<Complex64>::new(jobs(), fresh).unwrap();
        for name in ROWS {
            let row = Row::of(name).unwrap();
            let got = real.call(row, &mut exec, true).unwrap().1.unwrap();
            assert_eq!(
                mismatch(&got, &real.reference()),
                None,
                "{name} fresh={fresh}"
            );
            let got = complex.call(row, &mut exec, true).unwrap().1.unwrap();
            assert_eq!(
                mismatch(&got, &complex.reference()),
                None,
                "{name} fresh={fresh}"
            );
        }
    }
}

#[test]
fn a_changed_or_unwritten_value_is_a_mismatch() {
    assert_eq!(mismatch(&[1.0, -0.0], &[1.0, 0.0]), None);
    assert!(mismatch(&[1.0, 2.0], &[1.0, 3.0])
        .unwrap()
        .starts_with("element 1"));
    assert!(mismatch(&[f64::NAN], &[f64::NAN]).is_some());
}

#[test]
fn a_sample_lasts_at_least_a_millisecond() {
    let (samples, inner, _) = time(3, 0, || {
        Ok(Observed::Native(cpueinsum::GroupedRoute::Serial))
    })
    .unwrap();
    assert_eq!(samples.len(), 3);
    assert!(inner > 1, "a no-op call is batched");
    let mut n = 0;
    let err = time(2, 0, || {
        n += 1;
        Ok(Observed::Native(if n < 3 {
            cpueinsum::GroupedRoute::Serial
        } else {
            cpueinsum::GroupedRoute::Outer { lanes: 2 }
        }))
    })
    .unwrap_err();
    assert!(err.contains("route changed"), "{err}");
}
