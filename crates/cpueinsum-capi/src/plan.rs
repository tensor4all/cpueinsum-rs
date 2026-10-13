//! Plans built from labels, and their execution into caller-owned buffers.

use core::ffi::c_void;

use cpueinsum::{ArenaProvider, EinsumPlan, EinsumSpec, Error, Exec, Layout, Scalar, Scratch};
use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tcontract_capi::executor_from_handle;

use crate::{
    cpueinsum_layout, cpueinsum_plan, cpueinsum_status, guard, require_out, set_last_error,
    slice_from_ptr, slice_from_ptr_mut, CPUEINSUM_C32, CPUEINSUM_C64, CPUEINSUM_ERR_BACKEND,
    CPUEINSUM_ERR_CONTRACT, CPUEINSUM_ERR_DTYPE, CPUEINSUM_ERR_INTERNAL, CPUEINSUM_ERR_LABELS,
    CPUEINSUM_ERR_SHAPE, CPUEINSUM_OK,
};

/// A layout the plan owns, so a later execution can rebuild its views without
/// the caller repeating the description.
#[derive(Clone, Debug)]
struct OwnedLayout {
    extents: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    buffer_len: usize,
}

impl OwnedLayout {
    fn layout(&self) -> Layout<'_> {
        Layout::new(&self.extents, &self.strides).expect("both vectors have one length")
    }
}

/// One dtype's plan, its scratch, and the layouts it was planned for.
struct Planned<T: Scalar> {
    plan: EinsumPlan<T>,
    scratch: Scratch<T>,
    arena: ArenaProvider,
    inputs: Vec<OwnedLayout>,
    output: OwnedLayout,
    scratch_len: usize,
}

impl<T: Scalar> Planned<T> {
    fn views<'a>(
        inputs: &'a [OwnedLayout],
        pointers: &[*const c_void],
    ) -> Result<Vec<StridedView<'a, T>>, cpueinsum_status> {
        if pointers.len() != inputs.len() {
            set_last_error(format!(
                "execute: {} inputs planned, {} given",
                inputs.len(),
                pointers.len()
            ));
            return Err(crate::CPUEINSUM_ERR_INVALID_ARGUMENT);
        }
        let mut views = Vec::with_capacity(pointers.len());
        for (index, (layout, pointer)) in inputs.iter().zip(pointers).enumerate() {
            // SAFETY: the caller guarantees each pointer addresses
            // `layout.buffer_len` initialized elements that nobody writes
            // during the call.
            let data = unsafe {
                slice_from_ptr(
                    pointer.cast::<T>(),
                    layout.buffer_len as i64,
                    &format!("input {index} data"),
                )?
            };
            let view = StridedView::<T>::new(data, &layout.extents, &layout.strides, layout.offset)
                .map_err(|error| {
                    set_last_error(format!("input {index}: {error}"));
                    CPUEINSUM_ERR_SHAPE
                })?;
            views.push(view);
        }
        Ok(views)
    }

    fn execute(
        &mut self,
        executor: isize,
        pointers: &[*const c_void],
        output: *mut c_void,
    ) -> cpueinsum_status {
        // Destructure first: the views borrow only `inputs`, so the plan, the
        // scratch and the arena stay available mutably.
        let Planned {
            plan,
            scratch,
            arena,
            inputs,
            output: out_layout,
            ..
        } = self;
        let views = match Self::views(inputs, pointers) {
            Ok(views) => views,
            Err(status) => return status,
        };
        // SAFETY: the caller guarantees the output pointer addresses
        // `buffer_len` elements and that nothing else accesses them.
        let out_data = match unsafe {
            slice_from_ptr_mut(
                output.cast::<T>(),
                out_layout.buffer_len as i64,
                "output data",
            )
        } {
            Ok(data) => data,
            Err(status) => return status,
        };
        let mut out_view = match StridedViewMut::new(
            out_data,
            &out_layout.extents,
            &out_layout.strides,
            out_layout.offset,
        ) {
            Ok(view) => view,
            Err(error) => {
                set_last_error(format!("output: {error}"));
                return CPUEINSUM_ERR_SHAPE;
            }
        };
        // One call: on the host's executor when one is given, which keeps its
        // pool (and its budget) the host's, and serially on this plan's own
        // arena when the handle is zero. `Executor::with` holds the call in
        // flight, so the pool cannot be released under it.
        let mut call =
            |exec: &Exec<'_>| match plan.execute_into(exec, &views, &mut out_view, scratch) {
                Ok(()) => CPUEINSUM_OK,
                Err(error) => error_status(&error),
            };
        // SAFETY: the ABI contract says the handle is zero or live.
        match unsafe { executor_from_handle(executor) } {
            Some(executor) => {
                let mut status = CPUEINSUM_ERR_INTERNAL;
                let outcome = executor.with(|exec| {
                    status = call(exec);
                    Ok(())
                });
                if outcome.is_err() {
                    set_last_error("executor: the call's budget was rejected");
                    return CPUEINSUM_ERR_INTERNAL;
                }
                status
            }
            None => call(&Exec::serial_with_workspace(arena)),
        }
    }
}

/// The plan behind a handle, one variant per element type.
enum Handle {
    F32(Planned<f32>),
    F64(Planned<f64>),
    C32(Planned<Complex32>),
    C64(Planned<Complex64>),
}

impl Handle {
    fn scratch_len(&self) -> usize {
        match self {
            Handle::F32(planned) => planned.scratch_len,
            Handle::F64(planned) => planned.scratch_len,
            Handle::C32(planned) => planned.scratch_len,
            Handle::C64(planned) => planned.scratch_len,
        }
    }

    fn execute(
        &mut self,
        executor: isize,
        pointers: &[*const c_void],
        output: *mut c_void,
    ) -> cpueinsum_status {
        match self {
            Handle::F32(planned) => planned.execute(executor, pointers, output),
            Handle::F64(planned) => planned.execute(executor, pointers, output),
            Handle::C32(planned) => planned.execute(executor, pointers, output),
            Handle::C64(planned) => planned.execute(executor, pointers, output),
        }
    }
}

fn error_status(error: &Error) -> cpueinsum_status {
    set_last_error(error.to_string());
    match error {
        Error::Spec(_) => CPUEINSUM_ERR_LABELS,
        Error::Shape(_) | Error::Grouped(_) => CPUEINSUM_ERR_SHAPE,
        Error::Contract { .. } => CPUEINSUM_ERR_CONTRACT,
        Error::Backend { .. } => CPUEINSUM_ERR_BACKEND,
        _ => CPUEINSUM_ERR_INTERNAL,
    }
}

/// Read one C layout into an owned one, checking the counts before touching
/// anything.
fn read_layout(
    description: &cpueinsum_layout,
    what: &str,
) -> Result<OwnedLayout, cpueinsum_status> {
    let rank = description.rank;
    // SAFETY: the caller promises arrays of `rank` elements behind each pointer.
    let raw_extents = unsafe { slice_from_ptr(description.extents, rank, what) }?;
    // SAFETY: as above.
    let raw_strides = unsafe { slice_from_ptr(description.strides, rank, what) }?;
    let mut extents = Vec::with_capacity(raw_extents.len());
    for (index, extent) in raw_extents.iter().enumerate() {
        let extent = usize::try_from(*extent).map_err(|_| {
            set_last_error(format!(
                "{what}: extent {index} is not a non-negative count"
            ));
            crate::CPUEINSUM_ERR_INVALID_ARGUMENT
        })?;
        extents.push(extent);
    }
    let mut strides = Vec::with_capacity(raw_strides.len());
    for (index, stride) in raw_strides.iter().enumerate() {
        let stride = isize::try_from(*stride).map_err(|_| {
            set_last_error(format!(
                "{what}: stride {index} does not fit an element stride"
            ));
            crate::CPUEINSUM_ERR_INVALID_ARGUMENT
        })?;
        strides.push(stride);
    }
    let offset = isize::try_from(description.offset).map_err(|_| {
        set_last_error(format!("{what}: offset does not fit an element offset"));
        crate::CPUEINSUM_ERR_INVALID_ARGUMENT
    })?;
    let buffer_len = usize::try_from(description.buffer_len).map_err(|_| {
        set_last_error(format!("{what}: buffer length is not a non-negative count"));
        crate::CPUEINSUM_ERR_INVALID_ARGUMENT
    })?;
    Ok(OwnedLayout {
        extents,
        strides,
        offset,
        buffer_len,
    })
}

/// A validated plan description, before the element type is dispatched.
struct Description<'a> {
    inputs: Vec<&'a [i64]>,
    output: &'a [i64],
    path: Vec<[usize; 2]>,
    input_layouts: Vec<OwnedLayout>,
    output_layout: OwnedLayout,
}

fn build<T: Scalar>(description: &Description<'_>) -> Result<Planned<T>, cpueinsum_status> {
    let spec = EinsumSpec::new(&description.inputs, description.output, &description.path)
        .map_err(|error| error_status(&error))?;
    let layouts: Vec<Layout<'_>> = description
        .input_layouts
        .iter()
        .map(OwnedLayout::layout)
        .collect();
    let plan = EinsumPlan::<T>::new(&spec, &layouts, description.output_layout.layout())
        .map_err(|error| error_status(&error))?;
    let scratch_len = plan.scratch_len();
    Ok(Planned {
        plan,
        scratch: Scratch::new(),
        arena: ArenaProvider::new(),
        inputs: description.input_layouts.clone(),
        output: description.output_layout.clone(),
        scratch_len,
    })
}

/// Build a plan from labels, a contraction order, and one layout per operand.
///
/// # Safety
///
/// Every pointer argument must address an array of the length its count
/// argument states; the label and layout arrays stay the caller's.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cpueinsum_plan_create(
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
) -> cpueinsum_status {
    guard(|| {
        if let Err(status) = require_out(out, "out") {
            return status;
        }
        // SAFETY: the caller passes arrays of the stated lengths.
        let label_pointers = match unsafe { slice_from_ptr(labels, n_inputs, "labels") } {
            Ok(value) => value,
            Err(status) => return status,
        };
        // SAFETY: as above.
        let counts = match unsafe { slice_from_ptr(label_counts, n_inputs, "label_counts") } {
            Ok(value) => value,
            Err(status) => return status,
        };
        // SAFETY: as above.
        let output = match unsafe { slice_from_ptr(output_labels, output_rank, "output_labels") } {
            Ok(value) => value,
            Err(status) => return status,
        };
        // SAFETY: as above.
        let path_values =
            match unsafe { slice_from_ptr(path, path_steps.saturating_mul(2), "path") } {
                Ok(value) => value,
                Err(status) => return status,
            };
        // SAFETY: as above.
        let layouts = match unsafe { slice_from_ptr(input_layouts, n_inputs, "input_layouts") } {
            Ok(value) => value,
            Err(status) => return status,
        };
        if output_layout.is_null() {
            set_last_error("output_layout: null pointer");
            return crate::CPUEINSUM_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: the caller passes one layout for the output.
        let output_layout = unsafe { &*output_layout };

        let mut inputs: Vec<&[i64]> = Vec::with_capacity(label_pointers.len());
        for (index, (pointer, count)) in label_pointers.iter().zip(counts).enumerate() {
            // SAFETY: the caller passes one label array per input of that length.
            match unsafe { slice_from_ptr(*pointer, *count, &format!("labels[{index}]")) } {
                Ok(value) => inputs.push(value),
                Err(status) => return status,
            }
        }
        let mut path_pairs = Vec::with_capacity(path_values.len() / 2);
        for pair in path_values.as_chunks::<2>().0 {
            let mut step = [0usize; 2];
            for (slot, value) in step.iter_mut().zip(pair) {
                match usize::try_from(*value) {
                    Ok(value) => *slot = value,
                    Err(_) => {
                        set_last_error(format!("path: {value} is not an operand index"));
                        return CPUEINSUM_ERR_LABELS;
                    }
                }
            }
            path_pairs.push(step);
        }
        let mut owned_inputs = Vec::with_capacity(layouts.len());
        for (index, layout) in layouts.iter().enumerate() {
            match read_layout(layout, &format!("input_layouts[{index}]")) {
                Ok(value) => owned_inputs.push(value),
                Err(status) => return status,
            }
        }
        let owned_output = match read_layout(output_layout, "output_layout") {
            Ok(value) => value,
            Err(status) => return status,
        };

        let description = Description {
            inputs,
            output,
            path: path_pairs,
            input_layouts: owned_inputs,
            output_layout: owned_output,
        };
        let handle = match dtype {
            crate::CPUEINSUM_F32 => build::<f32>(&description).map(Handle::F32),
            crate::CPUEINSUM_F64 => build::<f64>(&description).map(Handle::F64),
            CPUEINSUM_C32 => build::<Complex32>(&description).map(Handle::C32),
            CPUEINSUM_C64 => build::<Complex64>(&description).map(Handle::C64),
            other => {
                set_last_error(format!("dtype: {other} is not a supported element type"));
                return CPUEINSUM_ERR_DTYPE;
            }
        };
        let handle = match handle {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        // SAFETY: `out` is writable per the check above.
        unsafe { *out = Box::into_raw(Box::new(handle)) as cpueinsum_plan };
        CPUEINSUM_OK
    })
}

/// Drop a plan handle. Destroying the null handle is a no-op.
///
/// # Safety
///
/// `plan` is zero or a live handle from [`cpueinsum_plan_create`], and no call
/// uses it afterwards.
#[no_mangle]
pub unsafe extern "C" fn cpueinsum_plan_destroy(plan: cpueinsum_plan) -> cpueinsum_status {
    guard(|| {
        if plan == 0 {
            return CPUEINSUM_OK;
        }
        // SAFETY: the caller passes a live handle it no longer uses.
        unsafe { drop(Box::from_raw(plan as *mut Handle)) };
        CPUEINSUM_OK
    })
}

/// Report the scratch elements the plan needs; it grows to that on the first
/// execution.
///
/// # Safety
///
/// `plan` is a live handle and `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn cpueinsum_plan_scratch_len(
    plan: cpueinsum_plan,
    out: *mut usize,
) -> cpueinsum_status {
    guard(|| {
        if let Err(status) = require_out(out, "out") {
            return status;
        }
        // SAFETY: a live handle per the contract.
        let handle = match unsafe { resolve(plan) } {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        // SAFETY: `out` is writable per the check above.
        unsafe { *out = handle.scratch_len() };
        CPUEINSUM_OK
    })
}

/// Execute a plan: `output = sum over the labels` of the given inputs, on
/// `executor` when one is given.
///
/// # Safety
///
/// `plan` is a live handle, `executor` is zero or a live `TAPP_executor` from
/// tcontract-capi, `inputs` addresses as many pointers as the plan was created
/// with, and every pointer addresses the buffer its layout describes for the
/// duration of the call.
#[no_mangle]
pub unsafe extern "C" fn cpueinsum_execute(
    plan: cpueinsum_plan,
    executor: isize,
    inputs: *const *const c_void,
    output: *mut c_void,
) -> cpueinsum_status {
    guard(|| {
        // SAFETY: a live handle per the contract.
        let handle = match unsafe { resolve(plan) } {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        let count = match handle {
            Handle::F32(planned) => planned.inputs.len(),
            Handle::F64(planned) => planned.inputs.len(),
            Handle::C32(planned) => planned.inputs.len(),
            Handle::C64(planned) => planned.inputs.len(),
        };
        // SAFETY: the caller passes that many input pointers.
        let pointers = match unsafe { slice_from_ptr(inputs, count as i64, "inputs") } {
            Ok(value) => value,
            Err(status) => return status,
        };
        handle.execute(executor, pointers, output)
    })
}

/// Resolve a handle to its state.
///
/// # Safety
/// `plan` is a live handle from [`cpueinsum_plan_create`].
unsafe fn resolve<'a>(plan: cpueinsum_plan) -> Result<&'a mut Handle, cpueinsum_status> {
    if plan == 0 {
        set_last_error("plan: null handle");
        return Err(crate::CPUEINSUM_ERR_INVALID_ARGUMENT);
    }
    // SAFETY: the caller guarantees a live handle.
    Ok(unsafe { &mut *(plan as *mut Handle) })
}
