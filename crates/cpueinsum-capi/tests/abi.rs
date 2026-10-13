//! The C ABI: a re-declaration test that fails to link on any drift, numerical
//! checks against a naive reference, and the error paths.

use std::ffi::{c_char, c_void};

use cpueinsum_capi::{
    cpueinsum_layout, cpueinsum_plan, CPUEINSUM_ABI_VERSION, CPUEINSUM_C64, CPUEINSUM_ERR_DTYPE,
    CPUEINSUM_ERR_INVALID_ARGUMENT, CPUEINSUM_ERR_LABELS, CPUEINSUM_ERR_SHAPE, CPUEINSUM_F64,
    CPUEINSUM_OK,
};

// The shipped header's declarations. A renamed symbol is a link failure, which
// is the point; the bodies below are the ABI this crate actually exports.
extern "C" {
    fn cpueinsum_plan_create(
        out: *mut cpueinsum_plan,
        dtype: i32,
        n_inputs: i64,
        labels: *const *const i64,
        label_counts: *const i64,
        output_labels: *const i64,
        output_rank: i64,
        path: *const i64,
        path_steps: i64,
        input_layouts: *const cpueinsum_layout,
        output_layout: *const cpueinsum_layout,
    ) -> i32;
    fn cpueinsum_plan_destroy(plan: cpueinsum_plan) -> i32;
    fn cpueinsum_plan_scratch_len(plan: cpueinsum_plan, out: *mut usize) -> i32;
    fn cpueinsum_execute(
        plan: cpueinsum_plan,
        exec: isize,
        inputs: *const *const c_void,
        output: *mut c_void,
    ) -> i32;
    fn cpueinsum_abi_version() -> i32;
    fn cpueinsum_last_error() -> *const c_char;
}

fn layout_for(extents: &[i64], strides: &[i64], buffer_len: i64) -> cpueinsum_layout {
    cpueinsum_layout {
        extents: extents.as_ptr(),
        strides: strides.as_ptr(),
        rank: extents.len() as i64,
        offset: 0,
        buffer_len,
    }
}

/// Run `ij,jk->ik` through the ABI on `exec` and return the result, checking it
/// against a naive loop. `exec` is a `TAPP_executor` handle and zero is serial.
fn run_matmul(exec: isize) -> [f64; 4] {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [0.0f64, 1.0, 1.0, 0.0];
    let mut d = [0.0f64; 4];
    let (m, k, n) = (2i64, 2i64, 2i64);

    let la = [0i64, 1];
    let lb = [1i64, 2];
    let ld = [0i64, 2];
    let label_arrays: [*const i64; 2] = [la.as_ptr(), lb.as_ptr()];
    let label_counts = [2i64, 2];
    let path = [0i64, 1];
    // The extents and strides stay alive for the create call: the ABI borrows
    // them for its duration (the plan keeps its own copy afterwards).
    let (a_extents, a_strides) = ([m, k], [1, m]);
    let (b_extents, b_strides) = ([k, n], [1, k]);
    let (d_extents, d_strides) = ([m, n], [1, m]);
    let layouts = [
        layout_for(&a_extents, &a_strides, 4),
        layout_for(&b_extents, &b_strides, 4),
    ];
    let output_layout = layout_for(&d_extents, &d_strides, 4);

    let mut plan: cpueinsum_plan = 0;
    // SAFETY: every pointer addresses an array of the stated length for the call.
    let status = unsafe {
        cpueinsum_plan_create(
            &mut plan,
            CPUEINSUM_F64,
            2,
            label_arrays.as_ptr(),
            label_counts.as_ptr(),
            ld.as_ptr(),
            2,
            path.as_ptr(),
            1,
            layouts.as_ptr(),
            &output_layout,
        )
    };
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());

    let inputs: [*const c_void; 2] = [a.as_ptr().cast(), b.as_ptr().cast()];
    // SAFETY: the pointers address the buffers the layouts describe, and `exec`
    // is zero or a live handle.
    let status = unsafe { cpueinsum_execute(plan, exec, inputs.as_ptr(), d.as_mut_ptr().cast()) };
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());

    let mut scratch = 0usize;
    // SAFETY: `plan` is live, `scratch` is writable.
    assert_eq!(
        unsafe { cpueinsum_plan_scratch_len(plan, &mut scratch) },
        CPUEINSUM_OK
    );
    // SAFETY: `plan` is live and no call uses it afterwards.
    assert_eq!(unsafe { cpueinsum_plan_destroy(plan) }, CPUEINSUM_OK);

    let mut expected = [0.0f64; 4];
    for j in 0..k {
        for i in 0..m {
            for l in 0..n {
                expected[(i + l * m) as usize] += a[(i + j * m) as usize] * b[(j + l * k) as usize];
            }
        }
    }
    assert_eq!(d, expected);
    d
}

#[test]
fn a_two_input_contraction_matches_the_naive_reference() {
    run_matmul(0);
}

/// An executor from tcontract-capi runs the plan on its own pool.
#[test]
fn an_owned_executor_runs_the_plan_on_its_pool() {
    let mut exec: isize = 0;
    // SAFETY: a writable out-parameter and a null options pointer.
    let status = unsafe {
        tcontract_capi::tcontract_tapp_executor_create_rayon(&mut exec, 2, std::ptr::null())
    };
    assert!(
        tcontract_capi::TAPP_check_success(status),
        "{}",
        last_error()
    );

    run_matmul(exec);

    let (mut width, mut budget) = (0usize, 0usize);
    // SAFETY: a live handle and two writable out-parameters.
    let status = unsafe {
        tcontract_capi::tcontract_tapp_executor_get_threads(exec, &mut width, &mut budget)
    };
    assert!(tcontract_capi::TAPP_check_success(status));
    assert_eq!((width, budget), (2, 2));

    // SAFETY: a live handle no call uses afterwards.
    assert!(tcontract_capi::TAPP_check_success(unsafe {
        tcontract_capi::TAPP_destroy_executor(exec)
    }));
}

/// A host-owned pool, lent to an executor and used by the plan: the sharing the
/// stack is built around.
#[test]
fn a_host_owned_pool_can_be_lent_to_the_plan() {
    let mut pool: isize = 0;
    // SAFETY: a writable out-parameter and a null options pointer.
    let status =
        unsafe { tcontract_capi::tcontract_pool_create_rayon(&mut pool, 2, std::ptr::null()) };
    assert!(
        tcontract_capi::TAPP_check_success(status),
        "{}",
        last_error()
    );

    let mut exec: isize = 0;
    // SAFETY: a live pool handle and a writable out-parameter.
    let status = unsafe { tcontract_capi::tcontract_tapp_executor_borrow_pool(&mut exec, pool, 2) };
    assert!(
        tcontract_capi::TAPP_check_success(status),
        "{}",
        last_error()
    );

    run_matmul(exec);

    // SAFETY: a live handle no call uses afterwards.
    assert!(tcontract_capi::TAPP_check_success(unsafe {
        tcontract_capi::TAPP_destroy_executor(exec)
    }));
    // The pool is busy until its last borrowed executor is gone, so it is
    // destroyed only now.
    // SAFETY: a live pool handle, with no borrowed executor left.
    let status = unsafe { tcontract_capi::tcontract_pool_destroy(pool) };
    assert!(
        tcontract_capi::TAPP_check_success(status),
        "{}",
        last_error()
    );
}

/// A three-input chain exercises more than one step, so the plan's own scratch
/// holds an intermediate.
#[test]
fn a_three_input_chain_matches_the_naive_reference() {
    // d[i] = sum_{j,k} a[i,j] b[j,k] c[k], contracting (b c) first.
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3], column-major
    let b = [1.0f64, 0.0, 0.0, 1.0, 0.5, 0.25]; // [3, 2]
    let c = [1.0f64, 2.0]; // [2]
    let mut d = [0.0f64; 2];
    let (i, j, k) = (2usize, 3usize, 2usize);

    let la = [0i64, 1];
    let lb = [1i64, 2];
    let lc = [2i64];
    let ld = [0i64];
    let label_arrays: [*const i64; 3] = [la.as_ptr(), lb.as_ptr(), lc.as_ptr()];
    let label_counts = [2i64, 2, 1];
    // Contract operands 1 and 2 first (the new operand is 3), then 0 and 3.
    let path = [1i64, 2, 0, 3];
    let (a_extents, a_strides) = ([2i64, 3], [1i64, 2]);
    let (b_extents, b_strides) = ([3i64, 2], [1i64, 3]);
    let (c_extents, c_strides) = ([2i64], [1i64]);
    let (d_extents, d_strides) = ([2i64], [1i64]);
    let layouts = [
        layout_for(&a_extents, &a_strides, 6),
        layout_for(&b_extents, &b_strides, 6),
        layout_for(&c_extents, &c_strides, 2),
    ];
    let output_layout = layout_for(&d_extents, &d_strides, 2);

    let mut plan: cpueinsum_plan = 0;
    // SAFETY: as above.
    let status = unsafe {
        cpueinsum_plan_create(
            &mut plan,
            CPUEINSUM_F64,
            3,
            label_arrays.as_ptr(),
            label_counts.as_ptr(),
            ld.as_ptr(),
            1,
            path.as_ptr(),
            2,
            layouts.as_ptr(),
            &output_layout,
        )
    };
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());

    let mut scratch = 0usize;
    // SAFETY: `plan` is live, `scratch` is writable.
    assert_eq!(
        unsafe { cpueinsum_plan_scratch_len(plan, &mut scratch) },
        CPUEINSUM_OK
    );
    assert!(scratch > 0, "a chained plan keeps an intermediate");

    let inputs: [*const c_void; 3] = [a.as_ptr().cast(), b.as_ptr().cast(), c.as_ptr().cast()];
    // SAFETY: the pointers address the layouts' buffers.
    let status = unsafe { cpueinsum_execute(plan, 0, inputs.as_ptr(), d.as_mut_ptr().cast()) };
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());
    // SAFETY: `plan` is live and no call uses it afterwards.
    assert_eq!(unsafe { cpueinsum_plan_destroy(plan) }, CPUEINSUM_OK);

    let mut expected = [0.0f64; 2];
    for row in 0..i {
        for jj in 0..j {
            for kk in 0..k {
                expected[row] += a[row + jj * i] * b[jj + kk * j] * c[kk];
            }
        }
    }
    assert_eq!(d, expected);
}

#[test]
fn the_reported_abi_version_matches_the_header() {
    // SAFETY: no arguments, no preconditions.
    assert_eq!(unsafe { cpueinsum_abi_version() }, CPUEINSUM_ABI_VERSION);
}

#[test]
fn malformed_arguments_are_rejected_without_touching_buffers() {
    let la = [0i64, 1];
    let lb = [1i64, 2];
    let ld = [0i64, 2];
    let label_arrays: [*const i64; 2] = [la.as_ptr(), lb.as_ptr()];
    let label_counts = [2i64, 2];
    let path = [0i64, 1];
    let (l_extents, l_strides) = ([2i64, 2], [1i64, 2]);
    let (o_extents, o_strides) = ([2i64, 2], [1i64, 2]);
    let layouts = [
        layout_for(&l_extents, &l_strides, 4),
        layout_for(&l_extents, &l_strides, 4),
    ];
    let output_layout = layout_for(&o_extents, &o_strides, 4);

    let create = |dtype: i32, path_steps: i64, out: *mut cpueinsum_plan| -> i32 {
        // SAFETY: the pointers address arrays of the stated lengths.
        unsafe {
            cpueinsum_plan_create(
                out,
                dtype,
                2,
                label_arrays.as_ptr(),
                label_counts.as_ptr(),
                ld.as_ptr(),
                2,
                path.as_ptr(),
                path_steps,
                layouts.as_ptr(),
                &output_layout,
            )
        }
    };

    let mut plan: cpueinsum_plan = 0;
    assert_eq!(create(42, 1, &mut plan), CPUEINSUM_ERR_DTYPE);
    assert!(!last_error().is_empty());
    assert_eq!(create(CPUEINSUM_F64, 2, &mut plan), CPUEINSUM_ERR_LABELS);
    // SAFETY: a null output is rejected before anything is written.
    assert_eq!(
        create(CPUEINSUM_F64, 1, std::ptr::null_mut()),
        CPUEINSUM_ERR_INVALID_ARGUMENT
    );

    // A layout that cannot cover the elements it addresses is refused before any
    // write, and the output is untouched.
    let short_layouts = [
        layout_for(&l_extents, &l_strides, 4),
        layout_for(&l_extents, &l_strides, 1),
    ];
    let mut plan: cpueinsum_plan = 0;
    // SAFETY: as above.
    let status = unsafe {
        cpueinsum_plan_create(
            &mut plan,
            CPUEINSUM_F64,
            2,
            label_arrays.as_ptr(),
            label_counts.as_ptr(),
            ld.as_ptr(),
            2,
            path.as_ptr(),
            1,
            short_layouts.as_ptr(),
            &output_layout,
        )
    };
    // The plan itself is fine (the layout is only read back at execution), so the
    // short buffer is caught by the execution call.
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());
    let a = [1.0f64; 4];
    let b = [1.0f64; 4];
    let mut d = [7.0f64; 4];
    let inputs: [*const c_void; 2] = [a.as_ptr().cast(), b.as_ptr().cast()];
    // SAFETY: the pointers address the buffers the layouts describe.
    let status = unsafe { cpueinsum_execute(plan, 0, inputs.as_ptr(), d.as_mut_ptr().cast()) };
    assert_eq!(status, CPUEINSUM_ERR_SHAPE, "{}", last_error());
    assert_eq!(d, [7.0; 4], "a refused call writes nothing");
    // SAFETY: `plan` is live and no call uses it afterwards.
    assert_eq!(unsafe { cpueinsum_plan_destroy(plan) }, CPUEINSUM_OK);

    // Destroying the null handle is a no-op.
    // SAFETY: zero is explicitly allowed.
    assert_eq!(unsafe { cpueinsum_plan_destroy(0) }, CPUEINSUM_OK);
}

#[test]
fn the_complex_element_types_have_their_own_plans() {
    use num_complex::Complex64;

    let a = [Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)];
    let b = [Complex64::new(1.0, 0.0), Complex64::new(1.0, -1.0)];
    let mut d = [Complex64::new(0.0, 0.0); 1];

    // A plain dot product: a[i] * b[i] summed over i, so the output is rank 0.
    let la = [0i64];
    let lb = [0i64];
    let ld: [i64; 0] = [];
    let label_arrays: [*const i64; 2] = [la.as_ptr(), lb.as_ptr()];
    let label_counts = [1i64, 1];
    let path = [0i64, 1];
    let (x_extents, x_strides) = ([2i64], [1i64]);
    let (y_extents, y_strides) = ([2i64], [1i64]);
    let (out_extents, out_strides): ([i64; 0], [i64; 0]) = ([], []);
    let layouts = [
        layout_for(&x_extents, &x_strides, 2),
        layout_for(&y_extents, &y_strides, 2),
    ];
    let output_layout = layout_for(&out_extents, &out_strides, 1);

    let mut plan: cpueinsum_plan = 0;
    // SAFETY: the pointers address arrays of the stated lengths.
    let status = unsafe {
        cpueinsum_plan_create(
            &mut plan,
            CPUEINSUM_C64,
            2,
            label_arrays.as_ptr(),
            label_counts.as_ptr(),
            ld.as_ptr(),
            0,
            path.as_ptr(),
            1,
            layouts.as_ptr(),
            &output_layout,
        )
    };
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());

    let inputs: [*const c_void; 2] = [a.as_ptr().cast(), b.as_ptr().cast()];
    // SAFETY: the pointers address the layouts' buffers.
    let status = unsafe { cpueinsum_execute(plan, 0, inputs.as_ptr(), d.as_mut_ptr().cast()) };
    assert_eq!(status, CPUEINSUM_OK, "{}", last_error());
    // SAFETY: `plan` is live and no call uses it afterwards.
    assert_eq!(unsafe { cpueinsum_plan_destroy(plan) }, CPUEINSUM_OK);
    assert_eq!(d[0], a[0] * b[0] + a[1] * b[1]);
}

fn last_error() -> String {
    // SAFETY: the pointer is never null and valid until the next failure.
    let pointer = unsafe { cpueinsum_last_error() };
    if pointer.is_null() {
        return "no message".to_owned();
    }
    // SAFETY: a NUL-terminated string owned by this thread.
    unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}
