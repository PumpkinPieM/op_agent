# Custom Op Builder Pitfalls

## Do not derive ACLNN launch arguments from torch_npu kwargs

### Symptom

The custom op compiles, but execution fails at synchronization or crashes even though the Python wrapper signature looks reasonable.

One confirmed case is `torch_npu.npu_rms_norm_quant`, whose Python interface includes `dst_dtype`, while the ACLNN document for `aclnnRmsNormQuantGetWorkspaceSize` has no `dst_dtype` parameter:

```cpp
aclnnRmsNormQuantGetWorkspaceSize(x, gamma, beta, scale, offset, epsilon, y, workspaceSize, executor)
```

Passing `dst_dtype` into `LAUNCH_ACLNN_FUNC(aclnnRmsNormQuant, ...)` shifted the ABI-level argument list and caused a crash when the output was materialized.

### Rule

Use the ACLNN document as the primary source for `LAUNCH_ACLNN_FUNC` argument order. Use torch_npu wrapper code only to understand wrapper behavior, defaults, dtype enum mapping, and fallback selection.

### Bad

```cpp
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRmsNormQuant, x, gamma, beta, scale, offset, epsilon, dst_dtype, y));
```

### Good

```cpp
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRmsNormQuant, x, gamma, beta, scale, offset, epsilon, y));
```

If a torch_npu kwarg controls output dtype but the ACLNN signature has no such attribute, allocate the output tensor with the desired dtype and pass only the ACLNN-documented output tensor.

## Dtype enum values must be ACLNN-compatible

### Symptom

`GetWorkspaceSize` fails with a dtype mismatch even when the allocated output tensor dtype looks correct. A confirmed error from `aclnnAscendAntiQuant`:

```text
AclNN_Parameter_Error(EZ1001): out(y) data type must be the same as dstType.
```

### Root Cause

The dtype attribute passed to ACLNN must match the ACL/CANN dtype enum expected by that ACLNN interface. A value guessed from a framework enum can select a different dtype than the allocated MindSpore output tensor.

### Rule

For dtype enum attributes, check the ACLNN document and torch_npu op-plugin adapter. In torch_npu adapters, look for calls such as `c10_npu::GetAclDataType(...)` and the default value passed when the kwarg is omitted. Then make the MindSpore output allocation and ACLNN dtype attribute agree.

For `aclnnAscendAntiQuant`, the default used by torch_npu maps to float16 output; the custom adapter must allocate float16 and pass the matching ACL dtype value.

## Output metadata must follow the ACLNN document

### Symptom

The custom op builds but `GetWorkspaceSize` rejects the call, output materialization crashes, or shape assertions fail.

Confirmed examples:

- `npu_gather_sparse_index`: output shape is `index.shape + input.shape[1:]`, not just `index.shape`.
- `npu_group_norm_silu`: `meanOut` and `rstdOut` dtype must match the ACLNN interface expectations for the selected input/weight path; allocating float32 unconditionally caused a dtype error.

### Rule

Before allocating outputs, read the ACLNN parameter table for each output tensor. Record:

- output rank and shape relation to inputs
- output dtype
- dtype relationships between inputs and outputs
- product-specific dtype restrictions

Use torch_npu meta registration or op-plugin allocation as secondary confirmation only when the ACLNN document is ambiguous.

## Allocate every ACLNN output, including auxiliary outputs

### Symptom

`GetWorkspaceSize` fails even though the returned outputs look correctly allocated.

One confirmed case is `aclnnAddRmsNormQuantV2`: the MindSpore wrapper returned only `{y1, y2, x_out}`, but the ACLNN interface also required a real `rms_norm` output tensor in the launch argument list. Passing a null tensor caused runtime failure.

Another confirmed detail from the same op: `y2Out` had to be allocated with the full input shape for the tested ACLNN path. A zero-shape placeholder was rejected by CANN.

### Rule

Allocate every output tensor required by the ACLNN workspace-size signature, even if the public wrapper does not return that tensor. Include those auxiliary outputs in `runner->Run(..., outputs)` so MindSpore tracks the write dependencies.

Return only the tensors required by the target Python interface, but never pass null output tensors to ACLNN unless the ACLNN document explicitly permits that exact output to be null or optional.

## Avoid std::vector<bool> for ACLNN bool arrays

### Symptom

The adapter fails to compile inside `CustomOpBuilder` with conversion errors like:

```text
no matching function for call to ConvertType(const std::vector<bool>&)
```

### Root Cause

`std::vector<bool>` is a C++ specialization and is not handled by MindSpore's ACLNN conversion helpers for bool arrays. MindSpore has conversion paths for byte-like bool arrays such as `std::vector<uint8_t>`.

### Rule

For ACLNN attributes that are boolean arrays, use `std::vector<uint8_t>` with `0` and `1` values instead of `std::vector<bool>`.

```cpp
const std::vector<uint8_t> &output_mask = std::vector<uint8_t>{1, 1};
```

Python tests may still express the logical mask as booleans when the wrapper accepts them reliably, but passing `[1, 1]` is usually the safest validation form for generated custom ops.

## Disabled gradient-mask outputs may be undefined

### Symptom

A gradient adapter matches torch-npu for enabled outputs but fails comparison for outputs disabled by a `grad_input_mask` or similar boolean mask.

One confirmed case is `npu_group_norm_swish_grad`: `grad_input_mask` controls whether weight and bias gradients are requested by ACLNN. Disabled output tensors can contain placeholder or undefined data rather than guaranteed zeros.

### Rule

When a torch-npu interface has a gradient output mask, compare only the outputs whose mask entries are enabled unless the API documentation explicitly guarantees the value of disabled outputs.

The adapter should still allocate all output tensors required by ACLNN, but tests should not assert numeric equality for masked-off outputs.

## Host CANN support can differ from available docs and headers

### Symptom

The code compiles, but runtime fails with a missing symbol, unsupported implementation, or missing kernel binary:

```text
aclnnRotaryPositionEmbeddingV2GetWorkspaceSize not in /lib64/libopapi.so
Op ConfusionTransposeD does not has any binary.
Parse dynamic kernel config fail.
Support for 2201 is not implemented.
```

### Rule

Treat these as host/runtime capability issues, not necessarily adapter bugs. Options:

- Fall back to an older ACLNN interface only when semantics still match the requested wrapper. Example: use `aclnnRotaryPositionEmbedding` when V2 is unavailable and the requested case does not require the V2-only `rotate` input.
- Otherwise, make tests skip with a specific reason and keep the adapter code documented.
- Mention the exact unsupported symbol/kernel in the report.

## MindSpore execution may be deferred until tensor materialization

### Symptom

A `try`/`except RuntimeError` around the custom op call does not catch a CANN error. Pytest still fails later at `actual.asnumpy()`.

### Root Cause

In pynative custom op execution, ACLNN launch or error reporting may occur when the output tensor is synchronized, converted to NumPy, printed, or otherwise materialized.

### Rule

When handling expected host-capability skips, wrap both the custom op call and all synchronization points:

```python
try:
    actual = custom_op(Tensor(x))
    actual_np = actual.asnumpy()
except RuntimeError as exc:
    if "does not has any binary" in str(exc).lower():
        pytest.skip("kernel is not supported by this CANN package")
    raise
```

## Installed torch_npu may not expose the spreadsheet interface

### Symptom

The target interface exists in the task list or source tree, but the validation host fails with:

```text
AttributeError: module 'torch_npu' has no attribute 'npu_xxx'
```

### Rule

Before writing reference-comparison tests, check `hasattr(torch_npu, "npu_xxx")` on the validation host. If the public API is absent:

- Keep the generated MindSpore wrapper matching the intended torch_npu interface.
- Prefer a smoke/shape/dtype test if the ACLNN kernel is available and enough expected metadata is known.
- Otherwise skip inside the test with a clear reason. Avoid module-level skips that collect zero tests and make pytest exit with code 5 in per-file scripts.

## CustomOpBuilder caching can hide source edits

### Symptom

After copying a fixed `.cc` file to the remote host, pytest appears to run the old behavior.

### Rule

If a changed adapter still behaves like the previous build, force rebuild by changing the `CustomOpBuilder` module name or clearing the relevant build/cache artifacts. This is especially important when iterating remotely on C++ adapter source.

## Match scalar wrapper type to the ACLNN C signature

### Symptom

A custom op using `ms::pynative::AclnnOpRunner` builds and the exported C++ function returns, but the process segfaults when the output is synchronized, printed, or converted to NumPy.

One confirmed case is `aclnnAdd`: passing `alpha` as a raw C++ integer can segfault because the ACLNN interface expects `aclScalar *`.

### Root Cause

`LAUNCH_ACLNN_FUNC` eventually calls MindSpore's ACLNN `ConvertTypes`. The conversion is type-driven and must match the ACLNN workspace-size C signature.

For raw scalar C++ types, the converter keeps the value as a scalar C++ value:

```cpp
template <typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
T ConvertType(T value) {
  return value;
}
```

For MindSpore scalar values, the converter creates an `aclScalar *`:

```cpp
inline aclScalar *ConvertType(const ScalarPtr &value) {
  ...
}
```

If the ACLNN parameter is `aclScalar *`, passing a raw `int64_t` or `double` gives the ACLNN workspace/execute function the wrong ABI-level argument type. If the ACLNN parameter is a native `int64_t` or `double`, passing a raw C++ scalar is correct.

### Bad

`aclnnAdd` expects `alpha` as `aclScalar *`, so this is wrong:

```cpp
constexpr int64_t alpha = 1;
auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("Add");
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAdd, x, y, alpha, out));
runner->Run({x, y}, {out});
```

### Good

Wrap only because the ACLNN interface expects `aclScalar *`:

```cpp
auto alpha = mindspore::MakeValue<int64_t>(1);
auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("Add");
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAdd, x, y, alpha, out));
runner->Run({x, y}, {out});
```

Use the `MakeValue<T>` type that matches the ACLNN scalar's intended value type. For example, use `mindspore::MakeValue<double>(scale)` or `mindspore::MakeValue<int64_t>(alpha)` when the ACLNN parameter is `aclScalar *`.

### Rule

Before writing `LAUNCH_ACLNN_FUNC`, check the ACLNN workspace-size signature:

- If the parameter is a plain C/C++ scalar such as `int64_t`, `int`, `bool`, or `double`, pass the raw scalar value, casting as needed.
- If the parameter is `aclScalar *`, pass a MindSpore `ScalarPtr`, usually from `mindspore::MakeValue<T>(value)`.
- Keep scalar attributes out of `runner->Run(...)`; it should contain tensor dependencies only.

## Dynamic TensorList outputs need a custom pybind caller

### Symptom

The target wrapper returns `Tensor[]`, but the number of outputs depends on runtime inputs such as `split_item`, `group_list`, or the length of an input tensor list. A fixed `PYBOOST_CALLER(N, ...)` either returns the wrong arity or cannot represent all valid paths.

One confirmed case is `torch_npu.npu_grouped_matmul`: `split_item` and `group_list` determine whether the ACLNN call returns one output or multiple grouped outputs.

### Rule

For dynamic-output custom ops, do not use a fixed-output `PYBOOST_CALLER`. Add a custom `py::args` caller that:

- casts the Python arguments to the exported C++ signature types
- infers the output count from those arguments before dispatch
- creates a promised vector with `mindspore::tensor::MakeVector<false>(output_num)`
- executes the real adapter inside a `PassthroughFrontendTask`
- resolves the promise with the returned output `TensorPtr`s

This keeps Python return arity aligned with the runtime output contract while still using `ms::pynative::AclnnOpRunner` for the ACLNN launch.

## Materialize tensors inside optional TensorList arguments

### Symptom

The adapter compiles, but V4/V5 or other TensorList-heavy ACLNN paths fail at output synchronization with:

```text
RuntimeError: The pointer[tensor] is null
```

The stack often points into MindSpore's ACLNN converter, for example `ConvertType(const tensor::TensorPtr&)`. A temporary guard may reveal a nested argument such as `bias[0] tensor pointer is null`.

### Root Cause

`ms::inner::ConvertStubNodeToTensor(args...)` does not reliably materialize tensors nested inside containers such as:

```cpp
std::optional<std::vector<ms::Tensor>>
```

The optional list object can exist, and each `ms::Tensor` wrapper can exist, while `tensor.tensor()` is still null when `LAUNCH_ACLNN_FUNC` converts the vector to `std::vector<TensorPtr>`.

### Rule

When a custom pybind caller captures arguments into an async frontend task, explicitly walk nested tensor containers before invoking the adapter:

```cpp
void ConvertStubArg(TensorList *tensors) {
  for (auto &tensor : *tensors) {
    ms::inner::ConvertStubNodeToTensor(tensor);
  }
}

void ConvertStubArg(std::optional<TensorList> *tensors) {
  if (tensors->has_value()) {
    ConvertStubArg(&tensors->value());
  }
}

void ConvertStubArg(std::optional<ms::Tensor> *tensor) {
  if (tensor->has_value()) {
    ms::inner::ConvertStubNodeToTensor(tensor->value());
  }
}
```

Apply this before `std::apply(func, cast_args)` in the dispatched task. This is especially important for torch_npu-style interfaces with many optional `Tensor[]?` arguments, such as `bias`, `scale`, `offset`, `antiquant_scale`, and `per_token_scale`.
