//! cpueinsum-rs#80 benchmark driver (not committed): symtn-dense's own
//! grouped GEMM (DefaultDenseExecutor on CpuEngine) at symtn-rs 09a3ebb.
//! Usage: native80 <workloads.txt> <threads> <prime_ms> <reps>
use std::time::{Duration, Instant};

use num_complex::Complex64;
use symtn_dense::{
    strided_batch_runs, CpuEngine, DefaultDenseExecutor, DenseExecutor, DenseGemmBatchJob,
    DenseRead, DenseScalar, DenseView, DenseViewMut, DenseWrite, Pool,
};

struct W { name: String, dtype: String, alpha: [f64; 2], beta: [f64; 2], jobs: Vec<[usize; 6]> }

fn parse(line: &str) -> W {
    let f: Vec<&str> = line.split('|').collect();
    let pair = |s: &str| { let v: Vec<f64> = s.split(' ').map(|x| x.parse().unwrap()).collect(); [v[0], v[1]] };
    let jobs = f[4].split(';').map(|j| { let v: Vec<usize> = j.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2], v[3], v[4], v[5]] }).collect();
    W { name: f[0].into(), dtype: f[1].into(), alpha: pair(f[2]), beta: pair(f[3]), jobs }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a[2].parse().unwrap();
    let prime = Duration::from_millis(a[3].parse().unwrap());
    let reps: usize = a[4].parse().unwrap();
    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build().unwrap());
    let mut executor = DefaultDenseExecutor::new();
    for line in std::fs::read_to_string(&a[1]).unwrap().lines() {
        let w = parse(line);
        let jobs: Vec<DenseGemmBatchJob> = w.jobs.iter().map(|j| DenseGemmBatchJob {
            lhs_offset: j[0], rhs_offset: j[1], dst_offset: j[2], rows: j[3], contracted: j[4], cols: j[5],
        }).collect();
        let runs = strided_batch_runs(&jobs);
        let lhs_len = jobs.iter().map(|j| j.lhs_offset + j.rows * j.contracted).max().unwrap();
        let rhs_len = jobs.iter().map(|j| j.rhs_offset + j.contracted * j.cols).max().unwrap();
        let out_len = jobs.iter().map(|j| j.dst_offset + j.rows * j.cols).max().unwrap();
        let val = |i: usize, s: usize| (((i * 37 + s) % 101) as f64 / 101.0 - 0.5, ((i * 53 + s) % 89) as f64 / 89.0 - 0.5);
        let s = [1usize];
        let mut times = Vec::with_capacity(reps);
        macro_rules! go { ($T:ty, $of:expr, $W:ident, $R:ident, $S:ident) => {{
            let of = $of;
            let lhs: Vec<$T> = (0..lhs_len).map(|i| { let (x, y) = val(i, 1); of(x, y) }).collect();
            let rhs: Vec<$T> = (0..rhs_len).map(|i| { let (x, y) = val(i, 2); of(x, y) }).collect();
            let mut out: Vec<$T> = (0..out_len).map(|i| { let (x, y) = val(i, 3); of(x, y) }).collect();
            let (alpha, beta) = (of(w.alpha[0], w.alpha[1]), of(w.beta[0], w.beta[1]));
            let mut call = |out: &mut [$T]| {
                executor.matmul_batch_axpby_into(
                    &mut engine,
                    DenseWrite::$W(DenseViewMut::new(out, &[out_len], &s, 0).unwrap()),
                    DenseRead::$R(DenseView::new(&lhs, &[lhs_len], &s, 0).unwrap()),
                    DenseRead::$R(DenseView::new(&rhs, &[rhs_len], &s, 0).unwrap()),
                    &jobs, &runs, DenseScalar::$S(alpha), DenseScalar::$S(beta),
                ).unwrap();
            };
            let start = Instant::now();
            call(&mut out);
            while start.elapsed() < prime { call(&mut out); }
            for _ in 0..reps { let t = Instant::now(); call(&mut out); times.push(t.elapsed().as_secs_f64() * 1e6); }
        }}}
        match w.dtype.as_str() {
            "F64" => go!(f64, |x: f64, _y: f64| x, F64, F64, F64),
            _ => go!(Complex64, Complex64::new, C64, C64, C64),
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\tnative", w.name, w.dtype, threads, times[reps / 2], times[0], times[reps - 1]);
    }
}
