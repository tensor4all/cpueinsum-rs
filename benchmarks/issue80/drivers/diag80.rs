use cpueinsum::tcontract::{OutputContract, PlanConfig};
use cpueinsum::{Exec, GroupedGemmJob, GroupedPlan};
use std::time::Instant;
fn main() {
    let t = 64;
    let tp = threadteam::Pool::builder().threads(t).build().unwrap();
    let pool = cpueinsum::Pool::borrow(&tp);
    let mut exec = Exec::new(&pool, t).unwrap();
    // 1. f64 Faer child grants on wide lanes.
    for n in [512usize, 768] {
        let jobs = [GroupedGemmJob::new(0, 0, 0, n, n, n)];
        let mut g = vec![];
        let _ = GroupedPlan::<f64>::from_prepared_with(GroupedPlan::<f64>::prepare_jobs(&jobs).unwrap(), Default::default(), &PlanConfig::default(), |k| {
            let p = k.build::<f64>()?;
            for w in [8, 16, 32, 64] { g.push((w, p.execution_width(&exec, w, 1.0, OutputContract::Initialized).unwrap().map(|x| x.get()))); }
            Ok(std::sync::Arc::new(p))
        }).unwrap();
        println!("f64 {n}^3 default child execution_width: {g:?}");
    }
    // 2. gm-dom f64: execute on a fresh plan (searches) vs a warm plan (cache hit).
    let mut jobs = vec![GroupedGemmJob::new(0, 0, 0, 512, 512, 512)];
    let (mut a, mut d) = (512 * 512, 512 * 512);
    for _ in 0..63 { jobs.push(GroupedGemmJob::new(a, a, d, 32, 32, 32)); a += 1024; d += 1024; }
    let plan = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
    let lhs = vec![0.5; plan.required_lhs_len()]; let rhs = lhs.clone(); let mut out = vec![0.0; plan.required_out_len()];
    for _ in 0..20 { plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out).unwrap(); }
    let mut warm = vec![]; let mut fresh = vec![];
    for _ in 0..11 {
        let t0 = Instant::now(); plan.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out).unwrap(); warm.push(t0.elapsed().as_secs_f64()*1e6);
        let p2 = GroupedPlan::<f64>::new(&jobs, &PlanConfig::default()).unwrap();
        let t0 = Instant::now(); p2.execute_into(&mut exec, 1.0, &lhs, &rhs, &mut out).unwrap(); fresh.push(t0.elapsed().as_secs_f64()*1e6);
    }
    warm.sort_by(|a,b| a.partial_cmp(b).unwrap()); fresh.sort_by(|a,b| a.partial_cmp(b).unwrap());
    println!("gm-dom f64 64T execute: warm plan median {:.0} us, fresh plan (placement search) median {:.0} us", warm[5], fresh[5]);
}
