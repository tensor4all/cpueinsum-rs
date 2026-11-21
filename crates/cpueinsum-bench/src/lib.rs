//! The parts of the `cpueinsum-bench` harness that a sibling harness reuses.
//!
//! Only the JSON reader is shared: `cpueinsum-replay` reads the
//! `symtn-gemm-trace/1` corpus with it, so the workspace keeps one JSON reader
//! and no JSON crate.

pub mod json;
