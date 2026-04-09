---
name: bprop-helper
description: Inspect, explain, review, or implement MindSpore expander bprop definitions for operators. Use when working in `mindspore/ccsrc/frontend/expander/grad`, tracing a `REG_BPROP_BUILDER(...)` registration, understanding the `BpropBuilder` or `Emitter` interface, mapping `i0/i1/...` to forward inputs, `out`, and `dout`, or reusing helpers from `grad_utils.*` while adding or debugging C++ bprop logic.
---

# Bprop Helper

Use this skill for C++ expander bprop work in MindSpore. Focus on the registrations under `mindspore/ccsrc/frontend/expander/grad` and the shared builder utilities under `frontend/expander/bprop`.

## Quick Start

- Locate the target definition with `rg -n 'REG_BPROP_BUILDER\\("OpName"' mindspore/ccsrc/frontend/expander/grad`.
- Read `mindspore/ccsrc/include/frontend/expander/bprop_interface.h` to confirm the `BpropBuilder` input and return contract.
- Read `mindspore/ccsrc/frontend/expander/bprop/bprop_irbuilder.h` for `REG_BPROP_BUILDER`, `BODYFUNC`, `FreeUselessValues_*`, and `CloneInplaceInput`.
- Read [references/grad-reference.md](references/grad-reference.md) when you need the source map, helper catalog, or authoring checklist.

## Read The Contract

- Treat `ib->GetInput(i0..)` as original forward inputs, followed by forward `out`, followed by `dout`.
- For a one-output op with `N` original inputs, `out` is `iN` and `dout` is `iN + 1`.
- For multi-output ops, `out` and `dout` are often tuples. Use `ib->TupleGetItem(...)` to extract pieces.
- Return one gradient per original forward input, in the same order as the forward inputs.
- Use `ib->OutZeros(input)` for non-differentiable inputs, attrs carried as inputs, indices, masks, booleans, or intentionally ignored gradients.
- Use `x->need_compute_grad_out()` when there's multiple input. Return zero gradients for inputs that don't require grad to save memory and computation.

## Follow The Registration Pattern

```cpp
REG_BPROP_BUILDER("OpName")
  .SetUnusedInputs({iK})
  .FreeUselessValues_IO({iA}, {iB})
  .SetBody(BODYFUNC(ib) {
    const auto &x = ib->GetInput(i0);
    const auto &y = ib->GetInput(i1);
    const auto &out = ib->GetInput(i2);
    const auto &dout = ib->GetInput(i3);

    auto dx = ...;
    auto dy = ...;
    return {dx, dy};
  });
```

- Add `SetUnusedInputs(...)` only as a compatibility or readability hint. The code marks it deprecated.
- Prefer `FreeUselessValues_I`, `FreeUselessValues_O`, or `FreeUselessValues_IO` for pynative memory release hints when they are truly correct for the op.
- Use `CloneInplaceInput()` only for inplace semantics.

## Reuse Existing Helpers

- Use `ReturnZeros` when every gradient is zero.
- Use `BinopGradCommon` after computing elementwise binary grads that need broadcast reduction back to input shapes.
- Use helpers from `grad_utils.*` before reimplementing reduction, transpose, norm, or index/scatter logic.
- Prefer `ib->Emit("ExistingGradOp", ...)` when a dedicated gradient primitive already exists.

## Work In This Order

1. Confirm the forward input layout, output layout, and attrs for the exact registered op.
2. Find a nearby example with the same pattern in the same grad file or in `grad_utils.*`.
3. Handle tuple outputs, broadcasting, `keep_dims`, scalar-vs-tensor differences, and dynamic shape or dynamic rank explicitly.
4. Return zero gradients for non-differentiable arguments unless the forward op really differentiates through them.
5. Add memory hints only after verifying they match the op behavior.

## Use The Reference

- Read [references/grad-reference.md](references/grad-reference.md) for the runtime entry points, builder interface, helper catalog, file-by-file source map, and search recipes.
