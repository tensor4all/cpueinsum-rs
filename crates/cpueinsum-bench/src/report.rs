//! Human-readable environment lines and the measurement toolkit's `key=value`
//! contract.
//!
//! The toolkit reads `info` as one `key=value` per line on top of the human
//! text. A key whose value this repository cannot supply is printed empty
//! rather than omitted, so a consumer can tell "not built" from "the harness
//! changed".

use cpueinsum::VERSION;

/// The machine and the build, for the record.
pub fn print_environment() {
    println!("harness     : cpueinsum-bench");
    println!("host        : {}", hostname());
    println!(
        "target      : {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    #[cfg(target_arch = "x86_64")]
    {
        let feat = |name: &str, present: bool| {
            if present {
                format!("{name} ")
            } else {
                String::new()
            }
        };
        print!("cpu features: ");
        print!("{}", feat("avx2", std::is_x86_feature_detected!("avx2")));
        print!("{}", feat("fma", std::is_x86_feature_detected!("fma")));
        print!(
            "{}",
            feat("avx512f", std::is_x86_feature_detected!("avx512f"))
        );
        print!(
            "{}",
            feat("avx512dq", std::is_x86_feature_detected!("avx512dq"))
        );
        println!();
    }
    println!("provider    : cpueinsum {VERSION}");
}

/// The `key=value` lines the toolkit reads. `threads` is empty when `info` was
/// called without `--threads`.
pub fn print_evidence_keys(threads: Option<usize>) {
    let requested = threads.map(|n| n.to_string()).unwrap_or_default();
    println!("provider=cpueinsum");
    println!("provider_version={VERSION}");
    println!("threads={requested}");
    println!("threads_effective={requested}");
    // cpueinsum links no vendor library; there is no TBLIS version to report.
    println!("tblis=");
}

fn hostname() -> String {
    match std::fs::read_to_string("/proc/sys/kernel/hostname") {
        Ok(name) if !name.trim().is_empty() => name.trim().to_string(),
        _ => std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
    }
}
