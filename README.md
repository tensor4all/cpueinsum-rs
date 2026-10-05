# cpueinsum-rs

CPU einsum over integer labels, with the contraction order supplied by the
caller.

Each binary step of an einsum is planned once by
[tprims-contract](https://github.com/tensor4all/tprims-rs) (a packed
TBLIS-style driver for general strides, faer for copy-free GEMM fusions, an
elementwise pass for all-batch steps). cpueinsum adds what an N-ary einsum
needs on top:

- an explicit contraction order in SSA form,
- the labels each intermediate keeps (the output's and those a later operand
  still needs; everything else is summed as early as possible),
- one reusable scratch buffer for all intermediates, with dead storage reused
  first fit,
- a small, stable integer-label API.

Deliberately not included: contraction-order search, string notation
(`"ij,jk->ik"`), BLAS, and GPUs. Those belong to the caller.

```rust
use cpueinsum::strided_view::{StridedView, StridedViewMut};
use cpueinsum::{EinsumPlan, EinsumSpec, Exec, Layout, Scratch};

// d[i, l] = sum_{j,k} a[i, j] b[j, k] c[k, l], contracting (a b) first.
// Inputs are operands 0, 1, 2; step 0 makes operand 3.
let spec = EinsumSpec::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3], &[[0, 1], [3, 2]])?;
let cm = Layout::new(&[2, 2], &[1, 2])?;
let plan = EinsumPlan::<f64>::new(&spec, &[cm, cm, cm], cm)?;
let mut scratch = Scratch::new();

let a = [1.0, 2.0, 3.0, 4.0];
let id = [1.0, 0.0, 0.0, 1.0];
let mut d = [0.0; 4];
let views = [
    StridedView::new(&a, &[2, 2], &[1, 2], 0)?,
    StridedView::new(&id, &[2, 2], &[1, 2], 0)?,
    StridedView::new(&id, &[2, 2], &[1, 2], 0)?,
];
let mut out = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0)?;
plan.execute_into(&Exec::serial(), &views, &mut out, &mut scratch)?;
assert_eq!(d, a);
```

`einsum_into` plans and runs in one call; `contract_into` is a one-shot binary
contraction. Threading is explicit: pass `Exec::rayon(&Pool::borrow(&pool))`
to use a caller-owned Rayon pool.

Element types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Strides may be
arbitrary, including negative; a label repeated within an input selects a
diagonal.

## Status

Early. The plan, layering and gates are in
[tensor4all/tprims-rs#62](https://github.com/tensor4all/tprims-rs/issues/62).
Not published to crates.io; tprims is a git dependency at a pinned rev.
MSRV 1.89.

## License

MIT OR Apache-2.0.
