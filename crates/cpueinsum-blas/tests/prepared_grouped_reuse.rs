//! Shared prepared plans retain immutable metadata, not per-execution scratch.

extern crate blas_src as _;

#[path = "../../cpueinsum/tests/common/mod.rs"]
mod common;

use common::{close, naive, Elem, Operand};
use cpueinsum::strided_view::StridedViewMut;
use cpueinsum::tcontract::api::Op;
use cpueinsum::tcontract::PlanConfig;
use cpueinsum::{EinsumPlan, EinsumSpec, Exec, GroupedGemmJob, Scratch};
use cpueinsum_blas::{BinaryPlan, BlasScalar, GroupedPlan, GroupedRoute};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn shared_group_matches_single_einsums<T: Elem + BlasScalar + Send + Sync>(fallback: bool) {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<GroupedPlan<T>>();
    send_sync::<BinaryPlan<T>>();

    let mut rng = ChaCha8Rng::seed_from_u64(52);
    let marker = T::random(&mut rng);
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    let mut jobs = Vec::new();
    let mut out_len = 0;
    // Rectangular GEMM, dot, GEMV, vector-matrix, outer-product geometry,
    // zero reduction and empty outputs. Leave gaps in all three buffers.
    let cases = [
        (2, 3, 4),
        (1, 4, 1),
        (3, 4, 1),
        (1, 3, 4),
        (4, 1, 3),
        (2, 0, 3),
        (0, 3, 2),
        (2, 3, 0),
    ];
    let cases = if fallback { &cases[5..] } else { &cases[..5] };
    for &(m, k, n) in cases {
        jobs.push(GroupedGemmJob::new(lhs.len(), rhs.len(), out_len, m, k, n));
        lhs.extend((0..m * k + 2).map(|_| T::random(&mut rng)));
        rhs.extend((0..k * n + 3).map(|_| T::random(&mut rng)));
        out_len += m * n + 1;
    }
    let spec = EinsumSpec::new(&[&[0, 1], &[1, 2]], &[0, 2], &[[0, 1]]).unwrap();
    for conjugate in [false, true] {
        let ops = [
            if conjugate {
                Op::Conjugate
            } else {
                Op::Identity
            },
            Op::Identity,
            Op::Identity,
        ];
        let plan = GroupedPlan::<T>::new_with_ops(&jobs, ops, &PlanConfig::default()).unwrap();
        if !fallback && conjugate && T::COMPLEX {
            assert!(plan.work_len() > 0, "exercise caller-owned packing storage");
        }
        let mut expected = vec![marker; out_len];
        for job in &jobs {
            let m = job.rows;
            let k = job.contracted;
            let n = job.cols;
            let a = Operand::new(
                lhs[job.lhs_offset..job.lhs_offset + m * k]
                    .iter()
                    .map(|&x| if conjugate { x.conjugate() } else { x })
                    .collect(),
                &[m, k],
                &[1, m.max(1) as isize],
                0,
            )
            .labelled(&[0, 1]);
            let b = Operand::new(
                rhs[job.rhs_offset..job.rhs_offset + k * n].to_vec(),
                &[k, n],
                &[1, k.max(1) as isize],
                0,
            )
            .labelled(&[1, 2]);
            let dims = [m, n];
            let strides = [1, m.max(1) as isize];
            let mut output = vec![T::default(); m * n];
            let mut view = StridedViewMut::new(&mut output, &dims, &strides, 0).unwrap();
            let single = EinsumPlan::<T>::new(
                &spec,
                &[a.layout(), b.layout()],
                cpueinsum::Layout::new(&dims, &strides).unwrap(),
            )
            .unwrap();
            single
                .execute_into(
                    &mut Exec::serial(),
                    &[a.view(), b.view()],
                    &mut view,
                    &mut Scratch::new(),
                )
                .unwrap();
            close(&output, &naive(&[a, b], &[0, 2], &dims), 1e-12);
            expected[job.out_offset..job.out_offset + m * n].copy_from_slice(&output);
        }
        // Every thread shares this same plan and input buffers. Mutable output,
        // packing storage and executor are private to each call's owner.
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let (plan, lhs, rhs, expected, start) = (&plan, &lhs, &rhs, &expected, &start);
                scope.spawn(move || {
                    let mut out = vec![marker; out_len];
                    let mut work = vec![T::default(); plan.work_len()];
                    let mut exec = Exec::serial();
                    start.wait();
                    for _ in 0..3 {
                        let route = plan
                            .execute_into(&mut exec, T::ONE, lhs, rhs, &mut out, &mut work)
                            .unwrap();
                        if fallback {
                            assert!(matches!(route, GroupedRoute::Native(_)));
                        } else {
                            assert_eq!(route, GroupedRoute::Vendor { jobs: cases.len() });
                        }
                        close(&out, expected, 1e-12);
                    }
                });
            }
        });
    }
}

#[test]
fn shared_f64_group_matches_single_einsums() {
    shared_group_matches_single_einsums::<f64>(false);
    shared_group_matches_single_einsums::<f64>(true);
}

#[test]
fn shared_c64_group_matches_single_einsums() {
    shared_group_matches_single_einsums::<num_complex::Complex64>(false);
    shared_group_matches_single_einsums::<num_complex::Complex64>(true);
}
