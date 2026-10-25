//! Unit tests for the loader and the positional-to-SSA translation. They live
//! here (not inline) so the production module stays production-only.

use super::*;

use std::collections::BTreeMap;

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{EinsumPlan, Exec, Layout, Scratch};

fn translate(format: &str, shapes: &[Vec<usize>], path: &[[usize; 2]]) -> Translation {
    let mut paths = BTreeMap::new();
    paths.insert("opt_size".to_string(), path.to_vec());
    Translation::new(format, shapes, paths).expect("the fixture translates")
}

#[test]
fn positional_path_becomes_ssa() {
    assert_eq!(to_ssa(&[[0, 1]], 2).unwrap(), vec![[0, 1]]);
    // After (0, 1) the list is [2, result], so the next step pairs 2 with it.
    assert_eq!(to_ssa(&[[0, 1], [0, 1]], 3).unwrap(), vec![[0, 1], [2, 3]]);
    // numpy's four-operand example: (0,1), (0,2), (0,1).
    assert_eq!(
        to_ssa(&[[0, 1], [0, 2], [0, 1]], 4).unwrap(),
        vec![[0, 1], [2, 4], [3, 5]]
    );
    assert!(to_ssa(&[[0, 0]], 2).is_err());
    assert!(to_ssa(&[[0, 1]], 3).is_err());
    assert!(to_ssa(&[[0, 2]], 3).is_err());
}

#[test]
fn translation_rejects_an_extent_conflict() {
    let shapes = vec![vec![2, 3], vec![4, 4]];
    let err = Translation::new("ij,jk->ik", &shapes, BTreeMap::new()).unwrap_err();
    assert!(err.contains("extent"), "{err}");
}

#[test]
fn translated_path_matches_a_naive_contraction() {
    // i j, j k, k l -> i l, contracting (A B) first.
    let shapes = vec![vec![2, 3], vec![3, 4], vec![4, 2]];
    let translation = translate("ij,jk,kl->il", &shapes, &[[0, 1], [0, 1]]);
    let spec = translation.spec("opt_size").unwrap();
    let layouts: Vec<Layout<'_>> = translation
        .dims
        .iter()
        .zip(&translation.strides)
        .map(|(dims, strides)| Layout::new(dims, strides).unwrap())
        .collect();
    let out = Layout::new(&translation.out_dims, &translation.out_strides).unwrap();
    let plan = EinsumPlan::<f64>::new(&spec, &layouts, out).unwrap();

    let a: Vec<f64> = (0..6).map(|k| k as f64 + 1.0).collect();
    let b = vec![1.0; 12];
    let c: Vec<f64> = (0..8).map(|k| (k % 3) as f64 - 1.0).collect();
    let views = vec![
        StridedView::new(&a, &translation.dims[0], &translation.strides[0], 0).unwrap(),
        StridedView::new(&b, &translation.dims[1], &translation.strides[1], 0).unwrap(),
        StridedView::new(&c, &translation.dims[2], &translation.strides[2], 0).unwrap(),
    ];
    let mut got = vec![0.0; translation.out_len];
    let mut view =
        StridedViewMut::new(&mut got, &translation.out_dims, &translation.out_strides, 0).unwrap();
    let mut scratch = Scratch::new();
    plan.execute_into(&mut Exec::serial(), &views, &mut view, &mut scratch)
        .unwrap();

    let mut want = vec![0.0; translation.out_len];
    for i in 0..2 {
        for l in 0..2 {
            let mut sum = 0.0;
            for j in 0..3 {
                for k in 0..4 {
                    sum += a[i * 3 + j] * b[j * 4 + k] * c[k * 2 + l];
                }
            }
            want[i * 2 + l] = sum;
        }
    }
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "{g} vs {w}");
    }
}

#[test]
fn json_reads_unicode_labels_and_escapes() {
    let text = r#"{"name":"x","format_string":"çè,cd->çd","shapes":[[2,3],[3,4]],"paths":{"opt_size":{"path":[[0,1]]}}}"#;
    let doc = crate::json::parse(text).unwrap();
    assert_eq!(doc.get("name").unwrap().as_str(), Some("x"));
    assert_eq!(
        doc.get("format_string").unwrap().as_str(),
        Some("çè,cd->çd")
    );
    let shapes = parse_shapes(doc.get("shapes").unwrap()).unwrap();
    assert_eq!(shapes, vec![vec![2, 3], vec![3, 4]]);
    let paths = parse_paths(doc.get("paths").unwrap()).unwrap();
    assert_eq!(paths["opt_size"], vec![[0, 1]]);
}
