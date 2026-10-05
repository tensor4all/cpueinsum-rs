//! tprims-contract against the CBLAS backend, at 1 and 4 threads.
//!
//! ```text
//! cargo run --release -p cpueinsum-blas --example bench [-- --threads 1,4 --case mps --sample-ms 20 --samples 11]
//! ```
//!
//! Cases: the tprims-rs#61 corpus (MPS chains, batched and plain products)
//! and larger ones where threads matter. Arms, per thread count:
//!
//! | arm | backend | timed |
//! | --- | --- | --- |
//! | `tprims_exec` | [`Tprims`] | `execute_into` of a prebuilt plan with kept scratch |
//! | `blas_exec` | [`Blas::default`] | the same |
//! | `tprims_call` | [`Tprims`] | spec, plan, scratch and output built per call |
//! | `blas_call` | [`Blas::default`] | the same |
//!
//! tprims-contract runs on a Rayon pool of the thread count (serial at one
//! thread); so do the steps the BLAS backend declines. BLAS reads its thread
//! count when the library initialises, so each thread count runs in a child
//! process with `VECLIB_MAXIMUM_THREADS` (Accelerate) and
//! `OPENBLAS_NUM_THREADS` set. Every output is checked against the
//! one-thread tprims result. Prints CSV, then a table of `exec` medians.

extern crate blas_src;

use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Layout, Pool, Scratch, StepBackend, Tprims};
use cpueinsum_blas::{Blas, BlasScalar};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

trait Elem: BlasScalar + Default + std::fmt::Debug {
    const NAME: &'static str;
    fn rand(rng: &mut ChaCha8Rng, scale: f64) -> Self;
    fn dist(self, other: Self) -> f64;
    fn magnitude(self) -> f64;
}

impl Elem for f64 {
    const NAME: &'static str = "f64";
    fn rand(rng: &mut ChaCha8Rng, scale: f64) -> Self {
        (rng.gen::<f64>() - 0.5) * scale
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).abs()
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl Elem for Complex64 {
    const NAME: &'static str = "c64";
    fn rand(rng: &mut ChaCha8Rng, scale: f64) -> Self {
        Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5) * scale
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).norm()
    }
    fn magnitude(self) -> f64 {
        Complex64::norm(self)
    }
}

/// An einsum with integer labels, label extents and a contraction order.
struct Case {
    name: String,
    complex: bool,
    inputs: Vec<Vec<i64>>,
    output: Vec<i64>,
    path: Vec<[usize; 2]>,
    extent: Vec<(i64, usize)>,
    /// Scale of the random entries (keeps long chains finite).
    scale: f64,
}

impl Case {
    fn dim(&self, label: i64) -> usize {
        self.extent.iter().find(|e| e.0 == label).unwrap().1
    }
}

/// `<phi|psi>` of two open MPS (bond `chi`, physical 2, `l` sites) from a
/// `chi x chi` left environment; two steps per site:
/// `E[a,b] A[a,s,c] -> T[b,s,c]`, `T[b,s,c] B[b,s,d] -> E'[c,d]`.
fn mps(chi: usize, l: usize) -> Case {
    let l64 = l as i64;
    let (a, b, s) = (|i: i64| i, |i: i64| 1000 + i, |i: i64| 2000 + i);
    let mut inputs = vec![vec![a(0), b(0)]];
    let mut extent = Vec::new();
    for i in 0..=l64 {
        extent.push((a(i), chi));
        extent.push((b(i), chi));
        extent.push((s(i), 2));
    }
    for i in 0..l64 {
        inputs.push(vec![a(i), s(i), a(i + 1)]);
        inputs.push(vec![b(i), s(i), b(i + 1)]);
    }
    let n = inputs.len();
    let mut path = Vec::new();
    let mut env = 0;
    for i in 0..l {
        path.push([env, 1 + 2 * i]);
        path.push([n + 2 * i, 2 + 2 * i]);
        env = n + 2 * i + 1;
    }
    Case {
        name: format!("mps_L{l}_chi{chi}"),
        complex: true,
        inputs,
        output: vec![a(l64), b(l64)],
        path,
        extent,
        scale: 2.0 / (2.0 * chi as f64).sqrt(),
    }
}

fn product(
    name: String,
    complex: bool,
    inputs: &[&[i64]],
    output: &[i64],
    path: &[[usize; 2]],
    extent: &[(i64, usize)],
) -> Case {
    Case {
        name,
        complex,
        inputs: inputs.iter().map(|v| v.to_vec()).collect(),
        output: output.to_vec(),
        path: path.to_vec(),
        extent: extent.to_vec(),
        scale: 1.0,
    }
}

fn corpus() -> Vec<Case> {
    let mut v = Vec::new();
    for chi in [4, 8, 16, 32, 64, 128] {
        v.push(mps(chi, 32));
    }
    // ikb,knb->inb: i=0 k=1 n=2 b=3.
    for b in [16, 64, 256] {
        for n in [2, 4, 8, 16, 32] {
            v.push(product(
                format!("ikb_knb_inb_n{n}_b{b}"),
                false,
                &[&[0, 1, 3], &[1, 2, 3]],
                &[0, 2, 3],
                &[[0, 1]],
                &[(0, n), (1, n), (2, n), (3, b)],
            ));
        }
    }
    v.push(product(
        "ijk_jkl_il_8x16x8".into(),
        false,
        &[&[0, 1, 2], &[1, 2, 3]],
        &[0, 3],
        &[[0, 1]],
        &[(0, 8), (1, 16), (2, 8), (3, 8)],
    ));
    for (n, complex) in [
        (32, true),
        (64, false),
        (256, false),
        (256, true),
        (1024, false),
    ] {
        v.push(product(
            format!("ij_jk_ik_{}_n{n}", if complex { "c64" } else { "f64" }),
            complex,
            &[&[0, 1], &[1, 2]],
            &[0, 2],
            &[[0, 1]],
            &[(0, n), (1, n), (2, n)],
        ));
    }
    for n in [64, 512] {
        v.push(product(
            format!("ij_jk_kl_il_n{n}"),
            false,
            &[&[0, 1], &[1, 2], &[2, 3]],
            &[0, 3],
            &[[0, 1], [3, 2]],
            &[(0, n), (1, n), (2, n), (3, n)],
        ));
    }
    v
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut acc = 1isize;
    dims.iter()
        .map(|&d| {
            let s = acc;
            acc *= d as isize;
            s
        })
        .collect()
}

/// Nanoseconds per call of `f`: repetitions sized to `target`, one warm-up
/// sample, then `samples` samples.
fn time(target: Duration, samples: usize, mut f: impl FnMut()) -> Vec<f64> {
    let mut reps = 1u64;
    loop {
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        let el = t0.elapsed();
        if el >= target / 4 {
            let per = el.as_secs_f64() / reps as f64;
            reps = ((target.as_secs_f64() / per).ceil() as u64).max(1);
            break;
        }
        reps *= 4;
    }
    for _ in 0..reps {
        f();
    }
    (0..samples)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..reps {
                f();
            }
            t0.elapsed().as_secs_f64() * 1e9 / reps as f64
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Opts {
    target: Duration,
    samples: usize,
}

/// The operands of a case, column-major.
struct Data<T> {
    shapes: Vec<Vec<usize>>,
    strides: Vec<Vec<isize>>,
    values: Vec<Vec<T>>,
    out_shape: Vec<usize>,
    out_strides: Vec<isize>,
}

impl<T: Elem> Data<T> {
    fn new(c: &Case) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(61);
        let shapes: Vec<Vec<usize>> = c
            .inputs
            .iter()
            .map(|l| l.iter().map(|&x| c.dim(x)).collect())
            .collect();
        let values = shapes
            .iter()
            .map(|s| {
                (0..s.iter().product::<usize>())
                    .map(|_| T::rand(&mut rng, c.scale))
                    .collect()
            })
            .collect();
        let out_shape: Vec<usize> = c.output.iter().map(|&x| c.dim(x)).collect();
        Data {
            strides: shapes.iter().map(|s| col_major(s)).collect(),
            shapes,
            values,
            out_strides: col_major(&out_shape),
            out_shape,
        }
    }

    fn views(&self) -> Vec<StridedView<'_, T>> {
        (0..self.values.len())
            .map(|i| {
                StridedView::new(&self.values[i], &self.shapes[i], &self.strides[i], 0).unwrap()
            })
            .collect()
    }

    fn out_len(&self) -> usize {
        self.out_shape.iter().product()
    }
}

/// Time one arm; returns the median and the output of the last call.
fn arm<T: Elem, B: StepBackend<T> + Copy>(
    c: &Case,
    d: &Data<T>,
    exec: &Exec<'_>,
    backend: B,
    per_call: bool,
    o: &Opts,
) -> (f64, usize, Vec<T>) {
    let refs: Vec<&[i64]> = c.inputs.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![T::default(); d.out_len()];
    let plan_of = |views: &[StridedView<'_, T>], dv: &StridedViewMut<'_, T>| {
        let spec = EinsumSpec::new(&refs, &c.output, &c.path).unwrap();
        let layouts: Vec<Layout<'_>> = views.iter().map(Layout::of).collect();
        EinsumPlan::<T, B>::with_backend(backend, &spec, &layouts, Layout::of_mut(dv)).unwrap()
    };
    let taken;
    let times = if per_call {
        {
            let views = d.views();
            let dv = StridedViewMut::new(&mut out, &d.out_shape, &d.out_strides, 0).unwrap();
            taken = plan_of(&views, &dv).backend_steps();
        }
        time(o.target, o.samples, || {
            let views = d.views();
            let mut fresh = vec![T::default(); d.out_len()];
            {
                let mut dv =
                    StridedViewMut::new(&mut fresh, &d.out_shape, &d.out_strides, 0).unwrap();
                let plan = plan_of(&views, &dv);
                let mut scratch = Scratch::with_len(plan.scratch_len());
                plan.execute_into(exec, &views, &mut dv, &mut scratch)
                    .unwrap();
            }
            out = black_box(fresh);
        })
    } else {
        let views = d.views();
        let plan = {
            let dv = StridedViewMut::new(&mut out, &d.out_shape, &d.out_strides, 0).unwrap();
            plan_of(&views, &dv)
        };
        taken = plan.backend_steps();
        let mut scratch = Scratch::with_len(plan.scratch_len());
        time(o.target, o.samples, || {
            let views = d.views();
            let mut dv = StridedViewMut::new(&mut out, &d.out_shape, &d.out_strides, 0).unwrap();
            plan.execute_into(exec, &views, &mut dv, &mut scratch)
                .unwrap();
            black_box(&out);
        })
    };
    (median(times), taken, out)
}

fn rel_err<T: Elem>(got: &[T], want: &[T]) -> f64 {
    let scale = want
        .iter()
        .map(|x| x.magnitude())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .map(|(a, b)| a.dist(*b))
        .fold(0.0, f64::max)
        / scale
}

fn run_case<T: Elem>(c: &Case, threads: usize, exec: &Exec<'_>, o: &Opts) {
    let d = Data::<T>::new(c);
    let (_, _, reference) = arm(c, &d, &Exec::serial(), Tprims, false, &o_once());
    let steps = c.path.len();
    let arms: [(&str, bool, bool); 4] = [
        ("tprims_exec", false, false),
        ("blas_exec", true, false),
        ("tprims_call", false, true),
        ("blas_call", true, true),
    ];
    for (name, blas, per_call) in arms {
        let (ns, taken, out) = if blas {
            arm(c, &d, exec, Blas::default(), per_call, o)
        } else {
            arm(c, &d, exec, Tprims, per_call, o)
        };
        let err = rel_err(&out, &reference);
        assert!(err < 1e-10, "{} {name} {threads}T: rel err {err:e}", c.name);
        println!(
            "{},{},{},{},{},{:.1},{:.1},{},{:.1e}",
            c.name,
            T::NAME,
            steps,
            threads,
            name,
            ns,
            ns / steps as f64,
            taken,
            err
        );
    }
}

/// One untimed-quality sample, for the reference result.
fn o_once() -> Opts {
    Opts {
        target: Duration::from_micros(1),
        samples: 1,
    }
}

fn worker(threads: usize, filter: Option<&str>, o: &Opts) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let pool = Pool::borrow(&pool);
    let exec = if threads == 1 {
        Exec::serial()
    } else {
        Exec::rayon(&pool)
    };
    for c in corpus() {
        if filter.is_some_and(|f| !c.name.contains(f)) {
            continue;
        }
        if c.complex {
            run_case::<Complex64>(&c, threads, &exec, o);
        } else {
            run_case::<f64>(&c, threads, &exec, o);
        }
    }
}

/// Medians of the `exec` arms by case, one column per thread count and backend.
fn summary(csv: &str, threads: &[usize]) {
    let rows: Vec<Vec<&str>> = csv.lines().map(|l| l.split(',').collect()).collect();
    let mut cases: Vec<&str> = Vec::new();
    for r in &rows {
        if !cases.contains(&r[0]) {
            cases.push(r[0]);
        }
    }
    let get = |case: &str, t: usize, a: &str| -> Option<(f64, &str)> {
        rows.iter()
            .find(|r| r[0] == case && r[3] == t.to_string() && r[4] == a)
            .map(|r| (r[5].parse().unwrap(), r[7]))
    };
    print!("\n{:<24}", "exec median (us)");
    for t in threads {
        print!(
            " {:>10} {:>10}",
            format!("tprims {t}T"),
            format!("blas {t}T")
        );
    }
    println!(" {:>6}", "BLAS steps");
    for case in cases {
        print!("{case:<24}");
        let mut taken = "";
        for &t in threads {
            for a in ["tprims_exec", "blas_exec"] {
                match get(case, t, a) {
                    Some((ns, k)) => {
                        print!(" {:>10.2}", ns / 1e3);
                        if a == "blas_exec" {
                            taken = k;
                        }
                    }
                    None => print!(" {:>10}", "-"),
                }
            }
        }
        println!(
            " {taken:>6}/{}",
            rows.iter().find(|r| r[0] == case).unwrap()[2]
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let filter = get("--case");
    let opts = Opts {
        target: Duration::from_millis(get("--sample-ms").map_or(20, |s| s.parse().unwrap())),
        samples: get("--samples").map_or(11, |s| s.parse().unwrap()),
    };
    if let Some(t) = get("--worker") {
        worker(t.parse().unwrap(), filter.as_deref(), &opts);
        return;
    }
    let threads: Vec<usize> = get("--threads")
        .unwrap_or_else(|| "1,4".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    println!(
        "# cpueinsum-blas bench: {} {}, sample_ms={} samples={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        opts.target.as_millis(),
        opts.samples
    );
    println!("case,dtype,steps,threads,arm,median_ns,ns_per_step,blas_steps,rel_err");
    let mut csv = String::new();
    for &t in &threads {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--worker", &t.to_string()])
            .args(["--sample-ms", &opts.target.as_millis().to_string()])
            .args(["--samples", &opts.samples.to_string()])
            .env("VECLIB_MAXIMUM_THREADS", t.to_string())
            .env("OPENBLAS_NUM_THREADS", t.to_string());
        if let Some(f) = &filter {
            cmd.args(["--case", f]);
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{t} threads: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        print!("{text}");
        csv.push_str(&text);
    }
    summary(&csv, &threads);
}
