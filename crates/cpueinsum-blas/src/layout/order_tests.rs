use super::*;
use cpueinsum::tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};

// The pre-#52 allocating selector is the oracle for priority and stable ties.
fn previous_choose(group: &[RoleAxis], carriers: &[OperandId; 2]) -> Vec<usize> {
    let candidates: Vec<Vec<usize>> = carriers
        .iter()
        .map(|&o| {
            let mut order: Vec<_> = (0..group.len()).collect();
            sort_by_stride(group, o, &mut order);
            order
        })
        .collect();
    candidates
        .iter()
        .find(|order| carriers.iter().all(|&o| fuse(group, order, o).is_some()))
        .or_else(|| {
            candidates
                .iter()
                .find(|order| fuse(group, order, carriers[0]).is_some())
        })
        .unwrap_or(&candidates[0])
        .clone()
}

#[test]
fn shared_order_buffer_preserves_candidate_priority_and_stable_ties() {
    let operand = |dims: &[usize], strides: &[isize]| {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
    };
    for extents in [[2, 2, 2], [1, 2, 2], [1, 1, 1]] {
        for a in [
            [1, 2, 4],
            [4, 2, 1],
            [2, 4, 1],
            [1, 1, 4],
            [0, 0, 0],
            [-1, -2, -4],
        ] {
            for d in [[1, 2, 4], [4, 2, 1], [1, 3, 9], [-1, -2, -4]] {
                let dims = [extents[0], extents[1], extents[2], 2];
                let p = Problem::from_labels(
                    DType::F64,
                    operand(&dims, &[a[0], a[1], a[2], 32]),
                    operand(&[2, 2], &[1, 2]),
                    CSpec::Absent,
                    operand(&dims, &[d[0], d[1], d[2], 32]),
                    &Labels::new(&[0, 1, 2, 4], &[4, 3], &[0, 1, 2, 3]),
                )
                .unwrap();
                for (axes, carriers) in [
                    (p.roles().m(), [OperandId::D, OperandId::A]),
                    (p.roles().n(), [OperandId::D, OperandId::B]),
                    (p.roles().k(), [OperandId::A, OperandId::B]),
                ] {
                    let expected = previous_choose(axes, &carriers);
                    let mut scratch = vec![usize::MAX; axes.len()];
                    let got = Group::new(axes, &carriers, &mut scratch).unwrap();
                    assert_eq!(got.order, expected, "dims={dims:?}, a={a:?}, d={d:?}");
                }
            }
        }
    }
}
