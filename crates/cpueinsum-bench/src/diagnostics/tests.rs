use super::*;

#[test]
fn a_shuffle_keeps_the_moments_and_breaks_the_weighted_sum() {
    let a: Vec<f64> = (0..64).map(|i| (i as f64) * 0.5 - 3.0).collect();
    let mut b = a.clone();
    b.swap(3, 40);
    let (da, db) = (Diagnostics::of(&a), Diagnostics::of(&b));
    assert!((da.sum_re - db.sum_re).abs() < 1e-12);
    assert!((da.sum_sq - db.sum_sq).abs() < 1e-9);
    assert!(da.max_abs == db.max_abs);
    let d = db.disagree(&da, 1e-9).expect("a transpose must not pass");
    assert_eq!(d.0, "weighted_re");
}

#[test]
fn identical_outputs_agree() {
    let a: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
    let d = Diagnostics::of(&a);
    assert!(d.disagree(&d, 1e-9).is_none());
}

#[test]
fn the_weights_are_index_dependent_and_in_range() {
    let w: Vec<f64> = (0..128).map(weight).collect();
    assert!(w.iter().all(|x| (0.0..1.0).contains(x)));
    let mut sorted = w.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    assert_eq!(sorted.len(), 128, "the weights must differ across indices");
}

#[test]
fn a_complex_output_reports_both_parts() {
    use num_complex::Complex64;
    let a: Vec<Complex64> = (0..16)
        .map(|i| Complex64::new(i as f64, -(i as f64)))
        .collect();
    let d = Diagnostics::of(&a);
    assert!((d.sum_re - 120.0).abs() < 1e-9);
    assert!((d.sum_im + 120.0).abs() < 1e-9);
}
