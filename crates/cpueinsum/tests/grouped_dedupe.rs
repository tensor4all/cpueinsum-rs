mod common;

use common::Elem;
use cpueinsum::tcontract::{api::Op, PlanConfig};
use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bitwise_matches_separate_jobs<T: Elem>(bits: impl Fn(T) -> [u64; 2]) {
    let mut rng = ChaCha8Rng::seed_from_u64(51);
    let lhs: Vec<T> = (0..32).map(|_| T::random(&mut rng)).collect();
    let rhs: Vec<T> = (0..24).map(|_| T::random(&mut rng)).collect();
    let jobs: Vec<_> = (0..200)
        .map(|id| {
            let (m, k, n) = [(2, 0, 2), (2, 1, 2), (3, 2, 1)][id % 3];
            GroupedGemmJob::new(id % 4 * 8, id % 3 * 8, (199 - id) * 8, m, k, n)
        })
        .collect();
    for mask in 0..8 {
        let ops = [0, 1, 2].map(|bit| {
            if mask & (1 << bit) != 0 {
                Op::Conjugate
            } else {
                Op::Identity
            }
        });
        let config = PlanConfig::default();
        let plan = GroupedPlan::<T>::new_with_ops(&jobs, ops, &config).unwrap();
        let separate: Vec<_> = jobs
            .iter()
            .map(|job| GroupedPlan::<T>::new_with_ops(&[*job], ops, &config).unwrap())
            .collect();
        let marker = T::random(&mut rng);
        let alpha = T::random(&mut rng);
        for (alpha, beta) in [
            (alpha, T::default()),
            (alpha, T::random(&mut rng)),
            (T::default(), T::default()),
        ] {
            let mut expected = vec![marker; 1600];
            for single in &separate {
                single
                    .execute_into_accum(&mut Exec::serial(), alpha, &lhs, &rhs, beta, &mut expected)
                    .unwrap();
            }
            let mut actual = vec![marker; 1600];
            for _ in 0..2 {
                actual.fill(marker);
                plan.execute_into_accum(&mut Exec::serial(), alpha, &lhs, &rhs, beta, &mut actual)
                    .unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(&a, &b)| bits(a) == bits(b)),
                    "mask={mask}"
                );
            }
        }
    }
}

#[test]
fn shared_geometry_is_bitwise_equal_for_all_dtypes_and_conjugations() {
    bitwise_matches_separate_jobs::<f32>(|x| [u64::from(x.to_bits()), 0]);
    bitwise_matches_separate_jobs::<f64>(|x| [x.to_bits(), 0]);
    bitwise_matches_separate_jobs::<num_complex::Complex32>(|x| {
        [u64::from(x.re.to_bits()), u64::from(x.im.to_bits())]
    });
    bitwise_matches_separate_jobs::<num_complex::Complex64>(|x| [x.re.to_bits(), x.im.to_bits()]);
}
