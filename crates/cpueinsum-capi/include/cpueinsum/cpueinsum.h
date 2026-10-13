/* libcpueinsum_capi: the C ABI over cpueinsum.
 *
 * A plan is built once from labels, a contraction order and one layout per
 * operand; every later call writes into caller-owned buffers. Nothing here
 * allocates for an operand: intermediates live in the plan's own scratch.
 *
 * Threading: cpueinsum_execute takes a TAPP_executor from tcontract-capi
 * (<tapp.h>), so a host that owns one pool shares it here instead of seeing a
 * second one; that header also offers a handle for a pool the host owns. Zero is
 * a serial call. This library creates no pool and never infers a width from the
 * environment.
 */
#ifndef CPUEINSUM_H
#define CPUEINSUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CPUEINSUM_ABI_VERSION 100 /* 0.2.0 */

typedef int32_t cpueinsum_status;
#define CPUEINSUM_OK 0
#define CPUEINSUM_ERR_INVALID_ARGUMENT 1 /* null pointer, negative count, malformed argument */
#define CPUEINSUM_ERR_SHAPE 2            /* operands do not fit the description or the plan */
#define CPUEINSUM_ERR_DTYPE 3            /* not one of the four element codes below */
#define CPUEINSUM_ERR_LABELS 4           /* labels or contraction order do not describe an einsum */
#define CPUEINSUM_ERR_CONTRACT 5         /* a contraction step failed in the engine */
#define CPUEINSUM_ERR_BACKEND 6          /* the step backend declined or failed a step */
#define CPUEINSUM_ERR_PANIC 7            /* a panic was caught at the boundary */
#define CPUEINSUM_ERR_INTERNAL 8         /* an invariant of this library was violated */

/* Element types, numbered as the TAPP datatype enumeration is. */
#define CPUEINSUM_F32 0
#define CPUEINSUM_F64 1
#define CPUEINSUM_C32 2 /* complex<float> */
#define CPUEINSUM_C64 3 /* complex<double> */

/* A borrowed operand layout, all counts in elements:
 *
 *   extents     extents[rank]
 *   strides     element strides[rank]; may be zero (a broadcast axis) or
 *               negative, and the addressed span may be gapped
 *   rank        number of axes
 *   offset      element offset of the logical origin inside the buffer
 *   buffer_len  element count of the buffer `data` points at
 *
 * The library checks that every element the layout addresses from `offset` lies
 * inside `buffer_len` and rejects the call otherwise (CPUEINSUM_ERR_SHAPE); the
 * elements between the lowest and the highest addressed element must still be
 * initialized memory of the operand's type. An input is read, the output is
 * written, neither is resized, and an output must not alias an input. No other
 * thread may touch a buffer during the call.
 *
 * Extents, strides and offsets are borrowed for the duration of the call that
 * reads them and are never freed here. */
typedef struct {
  const int64_t *extents;
  const int64_t *strides;
  int64_t rank;
  int64_t offset;
  int64_t buffer_len;
} cpueinsum_layout;

typedef intptr_t cpueinsum_plan;

/* Build a plan. `labels` holds one array per input, of the lengths in
 * `label_counts`; `path` holds `path_steps` pairs `(operand, operand)` naming
 * the order in which the inputs are contracted, exactly as the Rust
 * `EinsumSpec`. A repeated label inside an input selects a diagonal, a label in
 * one input only and not in the output is summed. `out` receives the handle. */
cpueinsum_status cpueinsum_plan_create(cpueinsum_plan *out, int32_t dtype,
                                       int64_t n_inputs,
                                       const int64_t *const *labels,
                                       const int64_t *label_counts,
                                       const int64_t *output_labels,
                                       int64_t output_rank, const int64_t *path,
                                       int64_t path_steps,
                                       const cpueinsum_layout *input_layouts,
                                       const cpueinsum_layout *output_layout);

/* Drop a plan. The null handle is a no-op; no call may use the handle after. */
cpueinsum_status cpueinsum_plan_destroy(cpueinsum_plan plan);

/* Scratch elements the plan needs; it grows to that on the first execution. */
cpueinsum_status cpueinsum_plan_scratch_len(cpueinsum_plan plan, size_t *out);

/* Execute `plan` on `exec` - a TAPP_executor from tcontract-capi (<tapp.h>), or
 * zero for a serial call - with one data pointer per input, in the order the
 * plan was built with, and the output pointer. The call runs on that executor's
 * pool and budget. `exec` is spelled intptr_t so this header needs no TAPP
 * header; the handle must be live for the call and must not be destroyed from
 * one of its own workers. */
cpueinsum_status cpueinsum_execute(cpueinsum_plan plan, intptr_t exec,
                                   const void *const *inputs, void *output);

/* The ABI revision this library was built with. */
int32_t cpueinsum_abi_version(void);

/* The message of the most recent failure on the calling thread; never null, and
 * valid until the next failing call on that thread. */
const char *cpueinsum_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CPUEINSUM_H */
