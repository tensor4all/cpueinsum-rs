//! The linked vendor BLAS: its identity and its thread count.
//!
//! cpueinsum-blas leaves the vendor's thread count to the vendor, so the
//! harness sets it to the row's width and reads it back where the vendor
//! allows. The campaign recorder strips the vendor variables from the
//! environment; the width is the measurement's, not the environment's.

/// `key=value` lines the recorder's vendor block reads.
pub fn identity() -> Vec<(&'static str, String)> {
    imp::identity()
}

/// Make the vendor run on `threads` threads, returning the count it reports
/// back, or `None` when the vendor has no read-back.
pub fn set_threads(threads: usize) -> Result<Option<usize>, String> {
    imp::set_threads(threads)
}

#[cfg(target_os = "macos")]
mod imp {
    /// Accelerate reads it once, when it initialises.
    pub const VARIABLE: &str = "VECLIB_MAXIMUM_THREADS";

    pub fn identity() -> Vec<(&'static str, String)> {
        vec![
            ("vendor.name", "accelerate".into()),
            ("vendor.linkage", "framework".into()),
            (
                "vendor.threads",
                std::env::var(VARIABLE).unwrap_or_default(),
            ),
        ]
    }

    /// Accelerate takes its thread count from the environment before its first
    /// call and offers no setter, so a process that does not carry the width
    /// re-executes itself with it (same arguments, same process id). It has no
    /// read-back either.
    pub fn set_threads(threads: usize) -> Result<Option<usize>, String> {
        use std::os::unix::process::CommandExt;
        let want = threads.to_string();
        if std::env::var(VARIABLE).ok().as_deref() == Some(want.as_str()) {
            return Ok(None);
        }
        let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
        let error = std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .env(VARIABLE, &want)
            .exec();
        Err(format!("re-executing with {VARIABLE}={want}: {error}"))
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::ffi::{c_char, c_int, CStr};

    extern "C" {
        fn openblas_set_num_threads(n: c_int);
        fn openblas_get_num_threads() -> c_int;
        fn openblas_get_num_procs() -> c_int;
        fn openblas_get_parallel() -> c_int;
        fn openblas_get_config() -> *const c_char;
        fn openblas_get_corename() -> *const c_char;
    }

    fn text(p: *const c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        // SAFETY: OpenBLAS returns a pointer to a static NUL-terminated string
        // (or null, handled above) that lives as long as the library.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    pub fn identity() -> Vec<(&'static str, String)> {
        // SAFETY: the getters take no arguments and only read OpenBLAS's
        // global configuration.
        let (config, corename, parallel, procs, threads) = unsafe {
            (
                text(openblas_get_config()),
                text(openblas_get_corename()),
                openblas_get_parallel(),
                openblas_get_num_procs(),
                openblas_get_num_threads(),
            )
        };
        let version = config
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
        let parallel = match parallel {
            0 => "sequential",
            1 => "pthreads",
            2 => "openmp",
            _ => "unknown",
        };
        vec![
            ("vendor.name", "openblas".into()),
            ("vendor.linkage", "system".into()),
            ("vendor.version", version),
            ("vendor.config", config),
            ("vendor.corename", corename),
            ("vendor.parallel", parallel.into()),
            ("vendor.procs", procs.to_string()),
            ("vendor.threads", threads.to_string()),
        ]
    }

    pub fn set_threads(threads: usize) -> Result<Option<usize>, String> {
        let n = c_int::try_from(threads).map_err(|_| format!("{threads} threads"))?;
        // SAFETY: called from the main thread before any BLAS call of this
        // process, with a positive count.
        let got = unsafe {
            openblas_set_num_threads(n);
            openblas_get_num_threads()
        };
        Ok(Some(got.max(0) as usize))
    }
}
