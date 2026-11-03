use super::*;

/// A corrupted result must not pass the cross-row comparison.
#[test]
fn rel_error_detects_a_flipped_element() {
    let want = [1.0f64, 2.0, 3.0];
    let clean = [1.0f64, 2.0, 3.0];
    let corrupted = [1.0f64, 2.0, 4.0];
    assert!(rel_error(&clean, &want) <= f64::TOLERANCE);
    assert!(rel_error(&corrupted, &want) > f64::TOLERANCE);
}

/// The readback accepts only the request the runtime actually granted.
#[test]
fn thread_readback_accepts_a_granted_request_and_refuses_a_clamp() {
    assert!(thread_readback(2, 2, 2).is_ok());
    let err = thread_readback(4, 2, 2).unwrap_err();
    assert!(err.contains("requested 4"), "{err}");
    assert!(err.contains("2 rayon worker"), "{err}");
    assert!(thread_readback(4, 4, 2).is_err());
}
