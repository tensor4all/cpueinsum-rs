//! The C ABI over `cpueinsum`: build a plan once from labels, then execute it
//! into caller-owned buffers.
//!
//! The library allocates nothing for an operand: every input and the output
//! arrive as a pointer plus a layout and an element length, exactly as the Rust
//! entry points take a slice plus a [`Layout`](cpueinsum::Layout). Intermediates
//! live in the plan's own scratch, which grows to the plan's requirement on the
//! first execution.
//!
//! # Operand contract (also stated in `include/cpueinsum/cpueinsum.h`)
//!
//! - extents, strides and offset are in **elements**, not bytes;
//! - `buffer_len` is the element count of the slice the data pointer refers to.
//!   Every element the layout addresses from `offset` must lie inside it, and the
//!   call rejects a layout that does not ([`CPUEINSUM_ERR_SHAPE`]);
//! - an input's buffer is read, the output's is written; neither is resized;
//! - the output must not alias an input, and no other thread may touch either
//!   during the call;
//! - strides may be zero (a broadcast axis) or negative, and the layout's
//!   addressed span may be gapped: the elements between the lowest and the
//!   highest addressed element must still be initialized memory of the operand's
//!   type.
//!
//! # Threading
//!
//! [`cpueinsum_execute`] takes a `TAPP_executor` handle from tcontract-capi: the
//! call runs on that executor's pool and budget, so a host that owns one pool
//! shares it here rather than seeing a second one. The handle is `intptr_t`, and
//! zero means a serial call on the plan's own scratch. This ABI creates no pool
//! of its own and never infers a width from the environment.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

use core::ffi::c_char;

mod plan;

pub use plan::{
    cpueinsum_execute, cpueinsum_plan_create, cpueinsum_plan_destroy, cpueinsum_plan_scratch_len,
};

/// ABI revision of this library, reported by [`cpueinsum_abi_version`].
pub const CPUEINSUM_ABI_VERSION: i32 = 100;

/// Status of a call: `0` is success, anything else is a failure code.
#[allow(non_camel_case_types)]
pub type cpueinsum_status = i32;

/// Success.
pub const CPUEINSUM_OK: cpueinsum_status = 0;
/// A null pointer, a negative count, or another malformed argument.
pub const CPUEINSUM_ERR_INVALID_ARGUMENT: cpueinsum_status = 1;
/// The operands do not fit the description or the plan.
pub const CPUEINSUM_ERR_SHAPE: cpueinsum_status = 2;
/// The dtype is not one of the four supported codes.
pub const CPUEINSUM_ERR_DTYPE: cpueinsum_status = 3;
/// The labels or the contraction order do not describe an einsum.
pub const CPUEINSUM_ERR_LABELS: cpueinsum_status = 4;
/// A contraction step failed in the contraction engine.
pub const CPUEINSUM_ERR_CONTRACT: cpueinsum_status = 5;
/// The step backend declined or failed a step.
pub const CPUEINSUM_ERR_BACKEND: cpueinsum_status = 6;
/// A panic was caught at the boundary; the library state is unchanged.
pub const CPUEINSUM_ERR_PANIC: cpueinsum_status = 7;
/// An invariant of this library was violated.
pub const CPUEINSUM_ERR_INTERNAL: cpueinsum_status = 8;

/// Element types, numbered as the TAPP datatype enumeration is.
pub const CPUEINSUM_F32: i32 = 0;
/// `f64`.
pub const CPUEINSUM_F64: i32 = 1;
/// `Complex<f32>`.
pub const CPUEINSUM_C32: i32 = 2;
/// `Complex<f64>`.
pub const CPUEINSUM_C64: i32 = 3;

/// A borrowed operand layout: extents, strides and origin, all in elements.
///
/// `buffer_len` is the element count of the slice `data` points at, so the
/// library can check that every addressed element is inside the caller's
/// buffer before it touches anything.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct cpueinsum_layout {
    /// Extents, one per axis.
    pub extents: *const i64,
    /// Element strides, one per axis; may be zero or negative.
    pub strides: *const i64,
    /// Number of axes.
    pub rank: i64,
    /// Element offset of the operand's logical origin inside the buffer.
    pub offset: i64,
    /// Element count of the buffer `data` points at.
    pub buffer_len: i64,
}

/// A plan handle built by [`cpueinsum_plan_create`].
#[allow(non_camel_case_types)]
pub type cpueinsum_plan = isize;

thread_local! {
    static LAST_ERROR: core::cell::RefCell<std::string::String> =
        const { core::cell::RefCell::new(std::string::String::new()) };
}

pub(crate) fn set_last_error(message: impl Into<String>) {
    let message = message.into();
    let _ = LAST_ERROR.try_with(|slot| slot.replace(message));
}

/// The message of the most recent failure on the calling thread.
///
/// The pointer is valid until the next failing call on the same thread, and is
/// never null (it is an empty string when there has been no failure).
#[no_mangle]
pub extern "C" fn cpueinsum_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        let mut bytes = slot.borrow_mut();
        if !bytes.is_empty() {
            bytes.push('\0');
        }
        bytes.as_ptr().cast::<c_char>()
    })
}

/// The ABI revision this library was built with.
#[no_mangle]
pub extern "C" fn cpueinsum_abi_version() -> i32 {
    CPUEINSUM_ABI_VERSION
}

/// Run `body`, turning a panic into [`CPUEINSUM_ERR_PANIC`].
///
/// The C boundary must not unwind, and a panic means the library is in an
/// unknown state, so the message is recorded and the call fails.
pub(crate) fn guard(body: impl FnOnce() -> cpueinsum_status) -> cpueinsum_status {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic with a non-string payload".to_owned());
            set_last_error(format!("panicked: {message}"));
            CPUEINSUM_ERR_PANIC
        }
    }
}

/// A borrowed slice from a C pointer, treating a null pointer with a zero
/// length as empty and rejecting any other null.
///
/// # Safety
/// `ptr` must be valid for `len` elements of `T` for the duration of the call.
pub(crate) unsafe fn slice<'a, T>(
    ptr: *const T,
    len: i64,
    what: &str,
) -> Result<&'a [T], cpueinsum_status> {
    if len < 0 {
        set_last_error(format!("{what}: negative length {len}"));
        return Err(CPUEINSUM_ERR_INVALID_ARGUMENT);
    }
    let len = len as usize;
    if ptr.is_null() {
        if len == 0 {
            return Ok(&[]);
        }
        set_last_error(format!("{what}: null pointer for {len} elements"));
        return Err(CPUEINSUM_ERR_INVALID_ARGUMENT);
    }
    // SAFETY: the caller guarantees `ptr` addresses `len` initialized elements.
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) })
}

pub(crate) use slice as slice_from_ptr;

/// A borrowed mutable slice from a C pointer, with the same null handling as
/// [`slice`].
///
/// # Safety
/// `ptr` must be valid for `len` elements of `T`, uniquely borrowed, for the
/// duration of the call.
pub(crate) unsafe fn slice_mut<'a, T>(
    ptr: *mut T,
    len: i64,
    what: &str,
) -> Result<&'a mut [T], cpueinsum_status> {
    if len < 0 {
        set_last_error(format!("{what}: negative length {len}"));
        return Err(CPUEINSUM_ERR_INVALID_ARGUMENT);
    }
    let len = len as usize;
    if ptr.is_null() {
        if len == 0 {
            return Ok(&mut []);
        }
        set_last_error(format!("{what}: null pointer for {len} elements"));
        return Err(CPUEINSUM_ERR_INVALID_ARGUMENT);
    }
    // SAFETY: the caller guarantees `ptr` addresses `len` elements of `T` and
    // that nothing else accesses them during the call.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}

pub(crate) use slice_mut as slice_from_ptr_mut;

/// Reject a null pointer that must be writable.
pub(crate) fn require_out<T>(ptr: *mut T, what: &str) -> Result<(), cpueinsum_status> {
    if ptr.is_null() {
        set_last_error(format!("{what}: null output pointer"));
        return Err(CPUEINSUM_ERR_INVALID_ARGUMENT);
    }
    Ok(())
}
