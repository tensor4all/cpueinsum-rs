use super::*;

#[test]
fn no_trans_layouts() {
    // Column-major 3x4.
    assert_eq!(no_trans(3, 4, 1, 3), Some(3));
    // Padded leading dimension.
    assert_eq!(no_trans(3, 4, 1, 5), Some(5));
    // Leading dimension below the row count.
    assert_eq!(no_trans(3, 4, 1, 2), None);
    // Row-major is not column-major.
    assert_eq!(no_trans(3, 4, 4, 1), None);
    // A single row or column ignores the irrelevant stride.
    assert_eq!(no_trans(1, 4, 7, 1), Some(1));
    assert_eq!(no_trans(3, 1, 1, 0), Some(3));
    // Negative leading dimension.
    assert_eq!(no_trans(3, 4, 1, -3), None);
}

#[test]
fn orientation() {
    assert_eq!(orient(3, 4, 1, 3, false), Some((Trans::No, 3)));
    assert_eq!(orient(3, 4, 4, 1, false), Some((Trans::T, 4)));
    // Conjugation exists only transposed.
    assert_eq!(orient(3, 4, 4, 1, true), Some((Trans::C, 4)));
    assert_eq!(orient(3, 4, 1, 3, true), None);
    // Neither axis unit stride.
    assert_eq!(orient(3, 4, 2, 6, false), None);
}
