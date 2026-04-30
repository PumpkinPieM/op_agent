# MindSpore Expander Bprop Reference

Use this reference when working on C++ expander bprop definitions under `mindspore/ccsrc/frontend/expander/grad`.

## Scope

- This skill covers C++ expander bprop registrations.
- This skill does not cover Python-side `bprop` definitions or unrelated autodiff pipelines.

## Entry Points

- `mindspore/ccsrc/frontend/expander/grad/*.cc`
  Register per-op bprop builders with `REG_BPROP_BUILDER("OpName")`.
- `mindspore/ccsrc/frontend/expander/grad/grad_utils.h`
  Declare shared helper functions and the index aliases `i0` through `i35`.
- `mindspore/ccsrc/frontend/expander/grad/grad_utils.cc`
  Implement broadcast, reduction, transpose, scatter, and special-function helpers reused across many ops.
- `mindspore/ccsrc/include/frontend/expander/bprop_interface.h`
  Define `BpropBuilder`, `BpropHandle`, `PynativeCallback`, and the runtime bprop contract.
- `mindspore/ccsrc/frontend/expander/bprop/bprop_irbuilder.h`
  Define the registration factory and macros: `REG_BPROP_BUILDER`, `BODYFUNC`, `FreeUselessValues_*`, `CloneInplaceInput`.
- `mindspore/ccsrc/frontend/expander/bprop/bprop_irbuilder.cc`
  Implement `BpropBuilder` helpers such as `BroadcastGradientArgs`, `SequenceToTensor`, `TensorToSequence`, `TensorGetItem`, and `StridedSlice`.
- `mindspore/ccsrc/frontend/expander/bprop/bprop.cc`
  Look up the registered builder at runtime and expand it into graph-mode IR.

## Bprop Body Contract

- `ib->GetInput(i0..)` always starts with original forward inputs.
- The forward `out` follows the original inputs.
- The final input is `dout`.
- Example: if the forward op has two inputs and one output, the bprop body usually reads `x=i0`, `y=i1`, `out=i2`, `dout=i3`.
- Example: if the forward op returns a tuple, `out` is a tuple node and `dout` is a tuple node. Extract pieces with `ib->TupleGetItem`.
- The return value is `NodePtrList` with one element per original forward input.
- `ReturnZeros(ib)` builds zero gradients for all original forward inputs by ignoring the last two bprop inputs (`out` and `dout`).
- `ib->OutZeros(x)` is the normal zero-gradient helper. By default it calls `ZerosLike(x)`.
- `x->need_compute_grad_out()` is the standard guard for optional gradients.

## Registration Helpers

- `SetBody(BODYFUNC(ib) { ... })`
  Register the gradient body.
- `SetUnusedInputs({...})`
  Deprecated memory hint that marks forward inputs and, if an index is beyond the input count, the output.
- `FreeUselessValues_I({...})`
  Free selected input device addresses in pynative mode.
- `FreeUselessValues_O({...})`
  Free selected output device addresses in pynative mode.
- `FreeUselessValues_IO({...}, {...})`
  Free selected inputs and outputs in pynative mode.
- `CloneInplaceInput()`
  Mark the op as needing cloned inplace input handling.

Do not copy these helpers blindly from a nearby op. They are memory behavior hints, not gradient math.

## Common Builder Surface

Use `BpropBuilder` when you need bprop-specific helpers:

- `GetInput`, `GetInputs`
- `GetAttr`, `GetAttrs`
- `GetShape`, `GetRank`, `GetDtype`, `GetDtypeId`, `GetSize`
- `BroadcastGradientArgs`
- `SequenceToTensor`, `TensorToSequence`, `SequenceSetItem`, `SequenceSlice`, `TensorToScalar`
- `TensorGetItem`, `StridedSlice`
- `OutZeros`

Use inherited `Emitter` helpers for graph construction:

- `Emit`
- `TupleGetItem`, `MakeTuple`, `MakeList`
- `Cast`, `Reshape`, `ExpandDims`, `Squeeze`, `Transpose`, `Concat`
- `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `BatchMatMul`, `Pow`
- `ReduceSum`, `SumExt`, `BroadcastTo`
- `Shape`, `ShapeCalc`, `Fill`, `ZerosLike`, `OnesLike`
- `Select`, `Equal`, `NotEqual`, `Less`, `Greater`, `LogicalAnd`, `LogicalOr`
- `Conditional`, `While`

Prefer the high-level helper when it exists. Fall back to `Emit("PrimitiveName", ...)` only when the builder API does not expose the exact primitive you need.

## Dynamic Shape Patterns

Dynamic bprop definitions usually fall into one of these patterns:

- **Static metadata first, runtime shape when needed.**
  `ib->GetShape(node)` and `ib->GetRank(node)` read inferred metadata. If `IsDynamic(...)`, `IsDynamicRank(...)`, or unknown scalar/sequence inputs make that metadata insufficient, switch to graph nodes such as `ib->Shape(node)`, `ib->DynSize(node)`, `ib->Emit("Rank", {node})`, or `ib->ShapeCalc(...)`.
- **Use `ShapeCalc` for shape arithmetic.**
  Put non-trivial shape math in a `DEF_PURE_SHAPE_CALC(...)` or `ShapeCalcFunctor` rather than rebuilding shape tuples with ad hoc IR. The `SetCalc` body computes concrete shape values when they are known; `SetInfer` describes the rank or output arity when the inputs are dynamic. Pass value-dependent inputs by index in the third `ShapeCalc` argument, for example `{1}` when an `axis`, `perm`, or `repeats` value affects the result.
- **Reshape, slice, tile, fill, and transpose with runtime shape nodes.**
  Prefer `ib->Reshape(x, ib->Shape(input))`, `ib->Slice(dout, offset, ib->Shape(input))`, `ib->Fill(..., ib->Shape(x), ...)`, or a perm produced by `ShapeCalc` when the target shape or permutation can be dynamic.
- **Dynamic rank needs graph control flow.**
  If static C++ branching depends on a rank, size, boolean input, or mutable sequence length that is not known at build time, use `ib->Conditional(...)` or `ib->While(...)` with branch/body lambdas that return `NodePtrList`. Examples include zero-rank reduction handling, dynamic sequence `Concat`/`Meshgrid`, and `Repeat` paths that skip empty reductions.
- **Broadcast gradients should reuse the shared helpers.**
  For binary grads, call `BinopGradCommon`; for `Select`/`Where`-style grads, call `BinopGradSelect`. These helpers inspect static shapes first and fall back to runtime broadcast axes via `BroadcastGradientArgs` or shape calculators, then reduce and reshape gradients back to input shapes.
- **Distinguish dynamic dimension from dynamic rank.**
  A shape like `{2, -1}` can often keep a static-rank path with runtime dimensions. Dynamic rank (`IsDynamicRank`) usually needs `ShapeCalc`, `Rank`, `Conditional`, or rank-agnostic primitives.

Representative examples:

```cpp
if (IsDynamic(ib->GetShape(x))) {
  auto shape = ib->ShapeCalc(g_some_shape_calc, {x, axis}, {1})[0];
  dx = ib->Reshape(dx, shape);
} else {
  dx = ib->Reshape(dx, ib->GetShape(x));
}
```

```cpp
auto axes = ib->BroadcastGradientArgs(dout, x)[1];
auto dx = ib->ReduceSum(dout, axes, false, true);
dx = ib->Reshape(dx, ib->Shape(x));
```

Useful examples in-tree:

- `grad_utils.cc`: `DynBinopGradCommon`, `BinopGradCommon`, `ReduceShapeShapeCalc`, `SumGrad`, `VarGrad`.
- `grad_array_ops.cc`: `GatherDropNegatives`, `BinopGather`, `Concat`, `Transpose`, `Repeat`.
- `grad_math_ops.cc`: `MatMulBackwardDynamic`, FFT/DCT shape calculators, reduce-product and diagonal dynamic paths.
- `grad_linalg_ops.cc`: SVD and QR dynamic helpers that combine `ShapeCalc`, runtime rank checks, and `Conditional`.
- `grad_nn_ops.cc`: dense/top-k/pooling shape calculators and dynamic output-size handling.

Dynamic shape authoring rules:

1. Guard every direct use of `shape.size()` or `shape[i]` with `!IsDynamicRank(shape)` and any required rank checks.
2. Do not call `GetIntValue` or `GetScalarValue(...).value()` on mutable or unknown inputs unless the op contract guarantees a constant value. If the value can be unknown, move the dependent computation into `ShapeCalc` or graph control flow.
3. Use `ib->Shape(input)` as the final reshape target when restoring a gradient to an original input shape.
4. For runtime reduce axes, prefer `ReduceSum(..., skip_mode=true)` when the code must tolerate dynamic rank or empty-axis behavior.
5. Keep zero gradients dynamic-safe with `ib->OutZeros(input)` or `ib->Zeros(ib->Shape(input), dtype_node)` instead of static shape literals.

## Common `grad_utils.*` Helpers

Use these first before reimplementing shared logic:

- Zero or broadcast helpers:
  `ReturnZeros`, `BinopGradCommon`, `NormalizeAxis`, `ReduceShape`, `ReduceShapeTupleDiv`, `GetIntValue`, `GetEps`
- Reduction helpers:
  `SumGrad`, `GetUnsqueezeTensor`, `LogSumExpGrad`, `VarGrad`, `MinOrMaxGrad`
- Index or scatter helpers:
  `GenerateInverseIndex`, `GenerateShapeIndex`, `RegenerateOutputShape`, `InplacePutGrad`, `ArgminOrArgmaxGrad`, `MeidanDimGrad`
- Permutation or transpose helpers:
  `InvertPermutation`, `GetTransposition`, `MatrixTranspose`, `MatrixTransposeExt`, `Adjoint`
- Math or dtype helpers:
  `PromoteBinaryDtype`, `LGamma`, `CheckType`, `VectorNormGrad`

`BinopGradCommon` is especially important. It reduces `dx` and `dy` back to the original input shapes after broadcasting and handles both static and dynamic shape paths.

## File Map

Use this map to search the right file first:

- `grad_math_ops.cc`
  Largest math and reduction coverage. About 313 builders.
- `grad_array_ops.cc`
  Shape, view, slice, gather, scatter, cast, transpose, repeat, and indexing patterns. About 201 builders.
- `grad_nn_ops.cc`
  Neural network, loss, normalization, pooling, convolution, dropout, and activation patterns. About 187 builders.
- `grad_sequence_ops.cc`
  Tuple, list, sequence slice, sequence stack, and scalar-sequence conversions.
- `grad_sparse_ops.cc`
  Sparse tensor and sparse operator gradients.
- `grad_scalar_ops.cc`
  Scalar arithmetic and simple scalar zero-grad cases.
- `grad_clip_ops.cc`
  Clamp and clipping-related gradients.
- `grad_image_ops.cc`
  Image resize and image-color related gradients.
- `grad_inner_ops.cc`
  Internal helper primitives and some dynamic helper ops.
- `grad_other_ops.cc`
  Miscellaneous stateful or framework-specific ops.
- `grad_comm_ops.cc`
  Communication ops.
- `grad_linalg_ops.cc`
  Small set of linear algebra gradients.
- `grad_scipy_ops.cc`
  SciPy-style linalg helpers.
- `grad_quant_ops.cc`
  Quantization-related ops.
- `grad_debug_ops.cc`
  Summary, debug, and side-effect ops.
- `grad_implementations_ops.cc`
  Special implementation or wrapper ops.

## Representative Patterns

Simple scalar pattern from `grad_scalar_ops.cc`:

```cpp
REG_BPROP_BUILDER("ScalarAdd").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i3);
  return {dout, dout};
});
```

Broadcast-aware binary pattern to copy:

```cpp
auto dx = ib->Mul(dout, y);
auto dy = ib->Mul(dout, x);
return BinopGradCommon(ib, x, y, dx, dy);
```

Tuple-output pattern to copy:

```cpp
auto out = ib->GetInput(iN);
auto dout = ib->GetInput(iN + 1);
auto first = ib->TupleGetItem(out, i0);
auto dfirst = ib->TupleGetItem(dout, i0);
```

## Search Recipes

- Find the bprop builder for one op:
  `rg -n 'REG_BPROP_BUILDER\\("OpName"' mindspore/ccsrc/frontend/expander/grad`
- Find existing uses of a helper:
  `rg -n 'BinopGradCommon|SumGrad|VarGrad|VectorNormGrad' mindspore/ccsrc/frontend/expander/grad`
- Find examples that guard optional gradients:
  `rg -n 'need_compute_grad_out\\(\\)' mindspore/ccsrc/frontend/expander/grad`
- Find memory hint usage:
  `rg -n 'FreeUselessValues|CloneInplaceInput|SetUnusedInputs' mindspore/ccsrc/frontend/expander/grad`
- Find tuple-output handling:
  `rg -n 'TupleGetItem\\(out|TupleGetItem\\(dout' mindspore/ccsrc/frontend/expander/grad`
- Find dynamic-shape branches:
  `rg -n 'IsDynamic\\(|IsDynamicRank\\(|IsDynamicShape\\(' mindspore/ccsrc/frontend/expander/grad`
- Find shape calculators:
  `rg -n 'DEF_PURE_SHAPE_CALC|ShapeCalc\\(' mindspore/ccsrc/frontend/expander/grad`
- Find graph control flow in bprop definitions:
  `rg -n 'Conditional\\(|While\\(' mindspore/ccsrc/frontend/expander/grad`

## Authoring Checklist

1. Confirm the exact forward signature and how many original inputs exist.
2. Map `i0..` to original inputs, then `out`, then `dout`.
3. Check whether `out` or `dout` is a tuple.
4. Reuse an existing helper or a nearby bprop pattern before writing new math.
5. Reduce broadcasted grads back to input shape when needed.
6. Handle `keep_dims`, scalar-vs-tensor, and dynamic shape or dynamic rank explicitly.
7. Return `OutZeros` for non-differentiable inputs.
8. Add `need_compute_grad_out()` guards when an expensive gradient can be skipped.
9. Add memory hints only when they are semantically correct for the op.
