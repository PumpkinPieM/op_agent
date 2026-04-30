# ACLNN Operator Development Reference

This file provides detailed reference material and copyable templates for `aclnn-builder`.
Load it only when needed so `SKILL.md` can stay concise.

<a id="reference-index"></a>
## Table Of Contents

- [1. Directory and File Discovery](#directory-and-file-discovery)
- [2. YAML Design Templates](#yaml-design)
- [3. gen_ops.py Troubleshooting](#gen-ops-py-troubleshooting)
- [4. GeneralInfer C++ Conventions](#general-infer)
- [5. PyBoost Pynative Implementation Notes](#pyboost-reference)
- [6. KBK Graph Kernel Notes](#kbk-reference)
- [7. BPROP Wiring Notes](#bprop-reference)
- [8. Test Strategy](#testing-reference)
- [9. Documentation Notes](#documentation-reference)
- [10. Backward Implementation Notes](#bprop-advanced-notes)
- [11. Resize/Launch Optimization Notes](#resize-launch-optimization)
- [12. Interface Development Notes](#api-development)
- [13. Code Skeletons](#code-skeletons)
- [14. PTA Source Review](#pta-source-review)
- [15. InferValue Constant Folding](#infervalue-constant-folding)
- [16. Dynamic Shape Categories And Strategies](#dynamic-shape-strategy)
- [17. ACLNN Call Chain Analysis](#aclnn-callchain-analysis)
- [18. Composite Implementation Patterns](#composite-implementation)
- [19. Feature Document](#feature-document-reference)
- [20. API Overload Adaptation](#api-overload-adaptation)

---

<a id="directory-and-file-discovery"></a>
## 1. Directory And File Discovery

MindSpore and op-plugin directory layouts may vary by branch. Prefer search over hard-coded paths.

Useful search keys:
- `gen_ops.py`
- `LAUNCH_ACLNN`
- `MS_ACLNN_KERNEL_FACTORY_REG`
- `REG_BPROP_BUILDER`

Common target areas:
- **YAML**: `mindspore/ops/op_def/yaml/`
- **Infer/meta implementation**: directories under `mindspore/` such as `ops`, `infer`, and `ops_func_impl`
- **Ascend kernel / PyBoost / KBK**: `mindspore/ccsrc/` and `op-plugin-*/` directories containing `ascend`, `kernel`, `aclnn`, or `customize`
- **bprop**: `bprop` and `grad_*ops.cc` under `mindspore/ccsrc/`
- **Tests**: `tests/ut/` and `tests/st/`
- **Docs**: English function_doc YAML and Chinese `docs/api/api_python/ops/*.rst`

For similar operator search, see [2.4 Similar Operator Search Strategy](#similar-operator-search).

<a id="yaml-design"></a>
## 2. YAML Design Templates

<a id="yaml-minimal-consistency"></a>
### 2.1 Minimal Consistency Rule

The same argument, such as `actual_seq_len`, must be consistent across:
- YAML (`op_def`, `api_def`, and `function_doc`)
- GeneralInfer C++ inference
- PyBoost Pynative call path
- KBK Graph kernel argument extraction and launch
- English and Chinese documentation
- UT/ST coverage for boundary cases and error paths

<a id="customize-suffix"></a>
### 2.2 Customize Suffix

If the operator uses the default ACLNN kernel mechanism, you usually do not need to add a `Customize` suffix manually in YAML. The framework handles it.

<a id="dispatch-path-selection"></a>
### 2.3 Dispatch Path Selection

The key integration decision is whether the MindSpore API arguments can be passed through to the ACLNN API unchanged. This determines whether to use the auto-generated path or the manual Customize path.

<a id="dispatch-path-1-auto-generated"></a>
#### Path 1: Auto-Generated Direct Passthrough

Use this path when MindSpore API arguments and ACLNN API arguments match exactly: argument count, order, type, and defaults require no pre-call conversion.

YAML key: set `dispatch.enable: True` in `op_def`, and omit the `Ascend` field. Internally, `Ascend` defaults to `default`, which triggers auto generation.

```yaml
# Path 1 example, such as abs, mul, trunc.
dispatch:
  enable: True
  # omit Ascend -> auto-generated path
```

Generated at build time:
- PyBoost call code from `pyboost_ascend_call_template.tpl`, usually using `LAUNCH_ACLNN(aclnnXxx, ...)`
- KBK registration through `MS_ACLNN_COMMON_KERNEL_FACTORY_REG` in `aclnn_kernel_register_auto.cc`
- Python wrappers such as `functional_overload.py`

Files developers usually write:

| File | Step |
| --- | --- |
| `op_def/yaml/xxx_op.yaml` | Step 1 |
| `api_def/xxx.yaml` | Step 1 |
| `op_def/yaml/doc/xxx_doc.yaml` in `_ext` style, or old-style `api_def/function_doc/xxx_doc.yaml` | Step 1 |
| `infer/ops_func_impl/xxx.h` and `.cc` | Step 3 |
| Add mapping to `aclnn_config.yaml` for Path 1 only | Step 2 |
| Export from `math_func.py`, `mint/__init__.py`, or `tensor_method.py` | Step 7 |
| `tests/ut/cpp/ops/test_xxx.cc` | Step 8 |
| `tests/st/ops/share/_op_info/op_database.py` plus corresponding `test_xxx_ops.py` | Step 8 |
| English function_doc and Chinese RST for each interface form | Step 9 |

You do not need PyBoost customize files or KBK customize files. Skip Step 4 and Step 5.

Examples: `abs`, `mul`, `trunc`, `xlogy`, and `div`.

<a id="dispatch-path-2-customize"></a>
#### Path 2: Manual Customize

Use this path when arguments must be converted before calling ACLNN. Common cases:
- `tuple[int]` to `std::vector<int64_t>`, such as `actual_seq_qlen`
- special `None` semantics for `Optional[Tensor]`
- string to enum or integer conversion, such as `layout: "BSND"` to an integer code
- scalar extraction from `Value`
- input reordering or merging
- manual output tensor allocation when output shape differs from input shape

YAML key: set `dispatch.enable: True` and specify `Ascend: XxxAscend`.

```yaml
# Path 2 example.
dispatch:
  enable: True
  Ascend: DenseLightningIndexerGradKlLossAscend
```

At build time, `gen_ops.py` generates wrapper code that calls the handwritten Customize class through `pyboost_ascend_customize_call_template.tpl`, such as `XxxAscendCustomize(...)`.

Extra files on top of Path 1:

| File | Step |
| --- | --- |
| `kernel/.../pyboost_impl/customize/xxx.h` and `.cc` | Step 4 |
| `kernel/.../kernel_mod_impl/customize/xxx_aclnn_kernel.h` and `.cc` | Step 5 |
| `_grad` variants for the files above when backward is required | Step 4/5 |

Examples: `dense_lightning_indexer_grad_kl_loss`, `multi_scale_deformable_attn`, `conv2d_ext`, and `add`.

<a id="dispatch-path-decision-flow"></a>
#### Path Decision Flow

```text
Compare MindSpore API arguments with ACLNN API arguments
                |
      Can arguments pass through unchanged?
       /                              \
      Yes                              No
      |                                |
  Path 1 auto-generated            Path 2 Customize
      |                                |
  Omit Ascend in YAML              Add Ascend: XxxAscend
      |                                |
  Skip Step 4/5                    Implement Step 4/5
      |                                |
  Build generates PyBoost/KBK      Build calls your Customize class
```

<a id="integration-type-mapping"></a>
#### Integration Types

| Integration type | Description | Path |
| --- | --- | --- |
| **Type 1** | API definition exactly matches ACLNN | **Path 1** |
| **Type 2** | Different name but same function | Usually **Path 1**, using the YAML `class` field for name mapping |
| **Type 3** | Prototype or semantics differ | **Path 2** |

Type 2 only avoids Customize when the difference is limited to operator name mapping. If argument order or types differ, use Path 2.

<a id="similar-operator-search"></a>
### 2.4 Similar Operator Search Strategy

Use similar integrated operators to confirm code style, directory layout, macro usage, registration style, and test patterns. Do not assume a fixed list of reference operators. Classify the target operator first, then search for matching operators.

#### A. Functional Or Algorithm Family

| Family | Typical operators | Common traits |
| --- | --- | --- |
| **Attention** | `flash_attention`, `nsa_compress_attention`, `paged_attention`, `incre_flash_attention` | TND/BSND layout, multiple outputs, mask or `actual_seq_len`, independent Grad operator |
| **Loss** | `cross_entropy`, `cosine_embedding_loss`, `ctc_loss`, `nll_loss` | loss plus cache outputs, `reduction`, backward needs intermediate values |
| **Norm** | `layer_norm`, `group_norm`, `rms_norm`, `batch_norm` | input/weight/bias, running stats, `rstd`, backward returns dx/dw/db |
| **Optimizer** | `adam`, `sgd`, `lamb`, `adamw` | in-place updates, scalar hyperparameters, many Tensor inputs, usually no backward |
| **Activation** | `relu`, `gelu`, `silu`, `swish`, `leaky_relu` | elementwise, one input and one output, simple backward, usually Type 1 |
| **Elementwise arithmetic** | `add`, `mul`, `div`, `eq`, `ne`, `gt` | broadcasting, Tensor-Scalar overloads, symbol overloads |
| **Reduce** | `sum`, `mean`, `prod`, `amax`, `argmax` | axis reduction, `keepdim`, output rank changes, mixed backward support |
| **Matrix** | `matmul`, `bmm`, `linear`, `baddbmm` | 2D/3D matrix multiplication, transpose flags, alpha/beta factors |
| **Index/gather** | `index_select`, `gather`, `scatter`, `embedding` | index Tensor input, irregular shape inference, scatter or zero-fill backward |
| **View/permute** | `reshape`, `transpose`, `permute`, `contiguous` | often pure shape transform, usually no ACLNN compute |
| **Convolution/pooling** | `conv2d`, `avg_pool2d`, `max_pool2d` | kernel/stride/padding/dilation tuples, NCHW/NHWC, separate Grad operators |
| **Communication** | `all_reduce`, `all_gather`, `reduce_scatter` | collective communication, group argument, side effects, usually HCCL instead of standard ACLNN |

#### B. Technical Traits

| Dimension | Typical categories | Search method |
| --- | --- | --- |
| **Input layout** | TND, BSND, BNSD, standard elementwise | Search the same shape comments in `op_def/yaml/` |
| **ACLNN integration** | single ACLNN passthrough, multi-ACLNN composite, no ACLNN | Search `LAUNCH_ACLNN`; inspect `customize` directories |
| **Backward** | independent Grad op, automatic differentiation, no backward | Search `REG_BPROP_BUILDER` and `_grad` YAML |
| **Interface form** | functional only, functional + nn, functional + tensor, symbol overload | Inspect `interface` in `api_def` YAML |
| **Special arguments** | `Optional[Tensor]`, `tuple[int]`, enum strings, scalar values | Search `default: None`, `type_cast`, `arg_handler` |
| **Integration type** | Type 1, Type 2, Type 3 | Decide using [2.3 Dispatch Path Selection](#dispatch-path-selection) |

#### Search Flow

1. Determine the functional family.
   - `nsa_compress_attention` -> Attention
   - `cosine_embedding_loss` -> Loss
   - `eq` and `==` overload -> Elementwise arithmetic

2. Choose two or three technical traits for filtering.
   - `nsa_compress_attention` -> Attention + TND layout + single ACLNN + independent Grad + tuple argument
   - `cosine_embedding_loss` -> Loss + multi-ACLNN composite + functional + nn + `reduction`
   - `adamw` -> Optimizer + in-place update + many Tensor inputs + no backward

3. Search the repository.

   ```bash
   # Search by family.
   rg -l "attention" mindspore/ops/op_def/yaml/

   # Search by layout.
   rg -l "TND" mindspore/ops/op_def/yaml/

   # Search composite ACLNN code.
   rg -l "LAUNCH_ACLNN" mindspore/ops/kernel

   # Search Grad YAML.
   rg --files mindspore/ops/op_def/yaml | rg "_grad_op\.yaml$"

   # Search tensor + function interfaces.
   rg -l "interface:.*tensor.*function" mindspore/ops/api_def/

   # Search reduction-style operators.
   rg -l "reduction" mindspore/ops/op_def/yaml/
   ```

4. Pick two or three closest matches and compare their YAML, Infer, PyBoost, KBK, bprop, tests, and docs. Prefer same family plus close technical traits. If no close family exists, use operators with the same integration type and argument pattern.

Principle: similar operators are references for structure and style, not sources for functional logic. Functional logic comes from PTA source plus ACLNN documentation.

<a id="dispatch-bootstrap-pattern"></a>
### 2.5 Practical Bootstrap: Auto Generate First, Then Customize

When PyBoost or KBK must be customized:

1. Enable `dispatch.enable: True` in YAML.
2. Temporarily comment out `dispatch.Ascend: XxxAscend` so `gen_ops.py` generates a compilable skeleton.
3. Copy generated `.h/.cc` files into the `customize` directory or the corresponding custom directory.
4. Adjust arguments according to the ACLNN signature, such as removing an unused dtype or converting tuple to vector.
5. Rename the entrypoint following project conventions, such as `OpNameAscendCustomize` or `OpNameGradAscendCustomize`, then restore the YAML `Ascend` field.
6. Delete temporary auto-generated files and keep only the custom implementation.

<a id="gen-ops-py-troubleshooting"></a>
## 3. gen_ops.py Troubleshooting

Typical errors and fixes:
- **Mismatched keys structure**: compare against existing basic operator YAML, such as `add`, and align field nesting.
- **Missing `py_method`**: add the Python exposure field.
- **Missing function_doc entry**: add the matching doc node and keep arguments consistent.

Keep English YAML docs ASCII-only when possible, especially on Windows, to avoid encoding issues.

<a id="general-infer"></a>
## 4. GeneralInfer C++ Conventions

<a id="general-infer-responsibilities"></a>
### 4.1 Responsibility Boundary

- Only infer shape and dtype. Do not validate runtime input legality there; leave runtime validation to ACLNN or the kernel.
- Use framework exception macros for errors. Error messages should include the argument name, expected condition, and actual value.

<a id="general-infer-dynamic-shape-rank"></a>
### 4.2 Dynamic Shape And Dynamic Rank

For the full three-category dynamic shape strategy, see [16. Dynamic Shape Categories And Strategies](#dynamic-shape-strategy). This section focuses on quick fallback rules during Infer.

Recommended rules:
- Dynamic rank: return dynamic rank, such as `kShapeRankAny`.
- If a key parameter required for inference is unknown, such as block, stride, or sequence length:
  - set affected output dimensions to dynamic dimension, such as `kShapeDimAny`
  - infer the remaining dimensions from inputs when possible
- If all key parameters are known, return the exact shape whenever possible.

<a id="general-infer-api"></a>
### 4.3 Common InferInfo APIs

`InferInfo` APIs are defined in `mindspore/core/include/ops/infer_info/infer_info.h`. Always follow the actual header in the target branch.

- `IsDynamic`, `IsDynamicRank`: dynamic shape and dynamic rank checks
- `GetScalarValueWithCheck<T>()`: read scalar with validation
- `GetArrayValue<T>()` plus `HasUnknownValue()`: read tuple/list values
- `IsNone()`: check `None`

Do not invent APIs that do not exist in the target project.

<a id="pyboost-reference"></a>
## 5. PyBoost Pynative Implementation Notes

<a id="pyboost-argument-normalization"></a>
### 5.1 Argument Normalization

- Convert tuple/list arguments to `std::vector<int64_t>` before passing them to ACLNN.
- For optional inputs, define the `None` semantics clearly and handle them consistently in PyBoost, Infer, and KBK.

### 5.2 Call Conventions

Follow the existing ACLNN wrapper style in the project, such as `LAUNCH_ACLNN` or `RunOp`.

<a id="kbk-reference"></a>
## 6. KBK Graph Kernel Notes

For Init/Resize/Launch responsibility separation, useless outputs, and compute-dependent outputs, see [11. Resize/Launch Optimization Notes](#resize-launch-optimization).

Recommended structure:
- `GetWorkSpaceInfo()`: extract arguments and call `GetWorkspaceForResize`
- `Launch()`: call `RunOp` or the equivalent execution path
- Registration: `MS_ACLNN_KERNEL_FACTORY_REG` or the project equivalent

Constraints:
- Keep forward and backward in separate files and registrations.
- Keep namespaces consistent between header and implementation, otherwise undeclared or undefined symbol errors are common.

<a id="kbk-auto-generated-skeleton"></a>
### 6.1 KBK Auto-Generated Skeleton Location

Generated KBK code often appears under a directory similar to:

```text
.../ops/kernel/ascend/opapi/aclnn_auto_gen/
```

Confirm the actual path in the target branch. You can first let `gen_ops.py` generate the skeleton, then copy it into a custom directory and adapt it. See [2.5 Practical Bootstrap](#dispatch-bootstrap-pattern).

<a id="bprop-reference"></a>
## 7. BPROP Wiring Notes

For advanced backward implementation notes such as `OutZeros`, `ZerosLikeExt`, in-place behavior, and `Depend`, see [10. Backward Implementation Notes](#bprop-advanced-notes).

In a bprop builder:
- Build backward subgraphs only for inputs that require gradients.
- Return zero-gradient placeholders for non-Tensor or non-differentiable inputs.
- Use `need_compute_grad_out()` or an equivalent API to avoid unnecessary gradient computation.

<a id="bprop-io-rules"></a>
### 7.1 Backward Input And Output Count Rules

- **Backward input count**: forward input count plus two, namely `out` and `dout`.
- **Backward output count**: forward input count, one gradient slot per input.
- For forward operators with multiple outputs, `out` is usually a tuple on the backward side. Use `TupleGetItem` to read the needed output.

<a id="bprop-set-unused-inputs"></a>
### 7.2 SetUnusedInputs

When backward does not depend on the value of some forward Tensor inputs, only their shape/type or nothing at all, mark them unused. This lets Pynative asynchronous execution release forward kernel memory earlier and reduce peak memory.

<a id="bprop-dynamic-inputs"></a>
### 7.3 Dynamic Inputs In Graph-Mode bprop

Priority: Pynative correctness comes before KBK dynamic compatibility. If a correct KBK dynamic adaptation is not ready, keep the Pynative-correct version and let test failures drive later fixes.

In graph mode, forward input values or shapes may be unknown at compile time (`ValueAny`, dynamic dimension, or dynamic rank). Direct C++ `if/else` plus `GetValue<T>()` only works when the value is known at compile time. If an input value may be unknown, use the framework runtime-deferred mechanisms.

#### Three Common Cases

| Case | Compile-time state | Tool | Notes |
| --- | --- | --- | --- |
| **Unknown scalar value** | `GetScalarValue` returns `has_value == false` | `ib->Conditional(cond, true_br, false_br)` | Convert C++ branches into graph branches and decide at runtime |
| **Unknown shape dimensions** | `IsDynamicShape(shape)` is true | `DEF_PURE_SHAPE_CALC` plus `ib->ShapeCalc(calc, inputs, indices)` | Put shape-dependent computation into a ShapeCalc node |
| **Unknown rank** | `IsDynamicRank(shape)` is true | separate dynamic path function | Split the full backward logic at the entry |

#### Pattern A: Unknown Scalar Value With `Conditional`

```cpp
auto keep_dims_value = keep_dims->BuildValue();
auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
if (keep_dims_opt.has_value()) {
  if (!keep_dims_opt.value()) {
    std_d = ib->Reshape(std_d, res[0]);
  }
} else {
  auto true_branch = [&](Emitter *e) -> NodePtrList { return {e->Reshape(std_d, res[0])}; };
  auto false_branch = [&](const Emitter *e) -> NodePtrList { return {std_d}; };
  auto cond = ib->Equal(keep_dims, ib->Value<bool>(false));
  std_d = ib->Conditional(cond, true_branch, false_branch);
}
```

Reference: `ReduceStd` bprop in `grad_math_ops.cc`.

#### Pattern B: Dynamic Shape With A Separate Path

```cpp
bool is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(w_shape);
bool is_dynamic_shape = IsDynamicShape(x_shape) || IsDynamicShape(w_shape);
if (is_dynamic_rank || is_dynamic_shape) {
  return MatMulBackwardDynamic(ib, is_complex);
}
dx = MatMulInputBackward(ib, is_complex);
```

Reference: `MatMulExt` bprop in `grad_math_ops.cc`.

#### Pattern C: Shape-Dependent Computation With `ShapeCalc`

```cpp
DEF_PURE_SHAPE_CALC(g_reduce_std)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    return ReduceStdShapeFunc(inputs.at(0), inputs.at(1));
  })
  ...

auto res = ib->ShapeCalc(g_reduce_std, {x, axis}, {1});
```

Reference: `ReduceStd` bprop in `grad_math_ops.cc`.

#### bprop Checklist

- [ ] Every scalar input used by backward checks `BuildValue()->ContainsValueAny()` or the equivalent.
- [ ] Unknown scalar values use `Conditional` instead of C++ `if`.
- [ ] Shape-dependent backward computations handle dynamic shape.
- [ ] Inputs that may have dynamic rank have an `IsDynamicRank` check and a fallback.

<a id="testing-reference"></a>
## 8. Test Strategy

<a id="testing-cpp-ut"></a>
### 8.1 C++ UT For GeneralInfer

Typical construction patterns:
- scalar: `ShapeVector{}` plus `CreateScalar<T>(value)`
- tuple: `ShapeArray{{}}` plus `ValuePtrList{CreateScalar<...>(...)}`
- `None`: `kMetaTypeNone` plus `kNone`
- unknown: `kValueAny` or the project equivalent

<a id="testing-st-opinfo"></a>
### 8.2 ST OpInfo Test Framework

Core idea: add the operator to test scenarios instead of writing a completely separate test file for every operator. The main code lives under `tests/st/ops/share/` and is built around **OpInfo** for what to test and **OpsFactory** for how to test it.

#### 8.2.1 Directory Layout

```text
tests/st/ops/share/
├── _op_info/
│   ├── op_info.py        # OpInfo and subclasses
│   ├── op_database.py    # OpInfo registration and xxx_op_db lists
│   └── op_common.py      # make_tensor / OpSampleInput / OpDynamicInput helpers
├── _internal/
│   ├── meta.py           # OpsFactory base class
│   ├── binary_ops.py     # BinaryStdOpsFactory and related factories
│   └── ...
├── ../op_info_tests/
│   ├── test_binary_ops.py
│   ├── test_unary_ops.py
│   └── ...
```

#### 8.2.2 OpInfo Configuration

OpInfo defines the target operator, input generation, and reference implementation.

| Field | Meaning |
| --- | --- |
| `name` | Operator name used in parametrized test IDs |
| `op` | Callable operator, such as `mint.add` |
| `ref` | Reference implementation, such as `numpy.add` |
| `dtypes_xxx` | Supported dtype sets for different scenarios |
| `op_basic_reference_inputs_func` | Basic input generator returning `OpSampleInput` entries |
| `op_extra_reference_inputs_func` | Optional extra input scenarios |
| `compare` | Numeric comparison options such as `atol` and `rtol` |

Common subclasses:

| Subclass | Scenario | Default inputs |
| --- | --- | --- |
| `BinaryOpInfo` | Binary operators such as add and mul | Two random tensors with the same shape |
| `UnaryOpInfo` | Unary operators such as abs and sin | One random tensor |
| `ReductionOpInfo` | Reduction operators such as sum and mean | Tensor plus axis argument |

Example:

```python
BinaryOpInfo(
    name="mint.add",
    op=mint.add,
    ref=numpy.add,
    dtypes_support=FLOAT_TYPES | INT_TYPES,
)

OpInfo(
    name="mint.clamp",
    op=mint.clamp,
    ref=numpy.clip,
    op_basic_reference_inputs_func=clamp_inputs_func,
    dtypes_support=FLOAT_TYPES,
)
```

After registration, add the operator name to the corresponding list, such as `binary_op_db` or `unary_op_db`.

#### 8.2.3 OpsFactory Test Suite

| Method | Meaning |
| --- | --- |
| `test_op_reference()` | Run samples and compare with `ref` |
| `test_op_dynamic()` | Test dynamic shape and dynamic rank |
| `compare_with_torch()` | Compare accuracy with PyTorch |

Factory hierarchy: `OpsFactory` -> `UnaryOpsFactory` / `BinaryOpsFactory` / `ReductionOpsFactory` -> specialized `XxxStdOpsFactory`.

Execution mode is selected by `set_context_mode()`: `pynative`, `kbk`, or `ge`.

#### 8.2.4 Frontend Test File Pattern

Frontend test files use pytest parametrization and `arg_mark` to expand registered operator cases.

```python
from tests.st.ops.share._internal.binary_ops import BinaryOpsFactory
from tests.st.ops.share._op_info.op_database import get_op_info, binary_op_db, binary_op_kbk_db

@pytest.mark.parametrize("op_info", binary_op_db)
def test_binary_op_reference_forward(op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode="pynative")
    fact.test_op_reference()

@pytest.mark.parametrize("op_info", binary_op_db)
def test_binary_op_reference_backward(op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode="pynative")
    fact.test_op_reference(grad_cmp=True)

@pytest.mark.parametrize("op_info", binary_op_kbk_db)
def test_binary_op_dynamic_forward(op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode="kbk")
    fact.test_op_dynamic(only_dynamic_shape=True)
    fact.test_op_dynamic(only_dynamic_rank=True)
```

#### 8.2.5 Helper Overview

| Helper | Purpose |
| --- | --- |
| `make_tensor(shape, dtype)` | Generate a random tensor |
| `OpSampleInput(input, args, kwargs)` | Wrap one sample input |
| `OpDynamicInput(...)` | Wrap dynamic shape test input |
| `OpErrorInput(...)` | Wrap expected-error input |
| `wrap_sample_inputs(func)` | Convert a simple input generator to `OpSampleInput` entries |

<a id="documentation-reference"></a>
## 9. Documentation Notes

### 9.1 Basic Export Requirements

- English function_doc YAML and Chinese RST must match on argument names, defaults, required/optional status, constraints, and examples.
- Operator APIs must be explicitly exported from the ops package.
- Non-Ascend devices should provide placeholder behavior with clear errors when appropriate.

<a id="documentation-general-principles"></a>
### 9.2 General Principles

- Keep Chinese and English docs strictly consistent.
- Add interface entries in alphabetical order to reduce conflicts and duplicates.
- File name, in-file title, and in-file API definition must match; mismatches can break page generation.
- Examples need complete imports and should be runnable. Print output or shape when it helps users understand the result.

<a id="documentation-output-mapping"></a>
### 9.3 Common Scenarios

- New functional API: English doc in the implementation `.py`; Chinese doc under `docs/api/api_python/ops/func_*.rst`; update API lists.
- New mint API: update both Chinese and English mint lists and Chinese RST. If it imports an existing API, reuse the existing documentation when valid.
- New Tensor method: English doc in `tensor.py`; Chinese doc under `docs/api/api_python/mindspore/Tensor/`; update API lists.

<a id="bprop-advanced-notes"></a>
## 10. Backward Implementation Notes

### 10.1 Non-Differentiable Inputs

For PTA non-differentiable inputs such as index or mode, the MindSpore backward output count must still match the input count.

- Return `ib->OutZeros(x)` for non-differentiable inputs.
- If all inputs are non-differentiable, `ReturnZeros` may be used depending on current framework support.

### 10.2 When The Gradient Is Mathematically Zero

When an input gradient is theoretically zero, prefer `ib->ZerosLikeExt()` so execution goes through the ACLNN/backend path expected by the framework.

### 10.3 In-Place Operator Backward

- If backward needs the original value of `self` before an in-place update, register `CloneInplaceInput(...)` so the framework preserves the old value.
- If in-place usage in KBK dynamic shape backward may break ordering, use `ib->Depend(target, inplace_call)`.

<a id="resize-launch-optimization"></a>
## 11. Resize/Launch Optimization Notes

For basic KBK structure and registration, see [6. KBK Graph Kernel Notes](#kbk-reference).

<a id="resize-launch-no-attr-mutation"></a>
### 11.1 Do Not Mutate Attributes In InferShape

Do not set or modify operator attributes in `InferShape` or `InferType`; this can introduce Pynative issues.

### 11.2 Resize/Launch Responsibility Separation

- Put what can be determined in `Init` into `Init`.
- Put shape-dependent work into `Resize`.
- Keep `Launch` focused on emitting or calling the execution path.
- Avoid runtime device memory allocation such as `cudaMalloc` or `cudaFree`; use framework-managed workspace.

### 11.3 Ignore Useless Outputs

For reserved or meaningless outputs, override `GetUseLessOutputIdx()` or the equivalent API to avoid dump, overflow-check, or deterministic side effects.

### 11.4 Compute-Dependent Outputs

Follow framework requirements: allocate the maximum possible output size and synchronize/update the output shape after execution, such as the `NonZero` pattern.

<a id="api-development"></a>
## 12. Interface Development Notes

### 12.1 Functional Interface

- Functional APIs must use `_get_cache_prim` internally to get Primitive instances and avoid repeated `__init__` overhead.
- Complex APIs may map one frontend function to multiple Primitives or a composite implementation based on argument branches.

### 12.2 nn Interface

- nn APIs are `Cell` subclasses. Initialize operators and attributes in `__init__`, and implement execution in `construct`.
- `construct` is similar to a compiler entrypoint. Avoid direct `raise` there; use `@constexpr` helpers when compile-time validation is required.

### 12.3 Tensor Methods And GE Mapping

- Tensor methods must cover required modes: PyNative, KBK, and GE when the project requires GE.
- GE mode often needs:
  - mapping registration in `resource.cc`
  - implementation in `standard_method.py`; validation functions there cannot take Tensor arguments directly and need the corresponding wrapper.

<a id="api-integration-strategy"></a>
### 12.4 Primitive And Interface Integration Strategy

Complete interface analysis and decide the Primitive/interface strategy before YAML definition.

<a id="api-analysis-five-factors"></a>
#### 12.4.1 Interface Analysis

Compare MindSpore with PTA/torch documentation and implementation:
1. Whether the functions are equivalent.
2. Whether argument definitions are equivalent.

If the interface and function are consistent, reuse the existing Primitive. Otherwise, add a new `XXXExt` Primitive. See [12.4.3 `ops.extend` Namespace](#ops-extend-namespace).

PTA/torch may have overloads with the same function name and different signatures. Analyze each signature separately.

<a id="yaml-three-scenarios"></a>
#### 12.4.2 Three YAML Scenarios

| Scenario | YAML action | Example |
| --- | --- | --- |
| **Existing YAML + reuse existing Primitive** | Add `dispatch` to existing YAML | `eye`: existing Primitive plus `dispatch.Ascend: EyeAscend` |
| **Existing YAML + new Primitive** | Create new YAML with `_ext` suffix | `zeros_like_ext`: existing `zeros_like` has incompatible arguments |
| **No YAML** | Create a new YAML, usually without `_ext` | Brand-new operator |

Reuse existing Primitive:

```yaml
dispatch:
  enable: True
  Ascend: EyeAscend
```

New Primitive with `_ext`:

```yaml
zeros_like_ext:
  args:
    input:
      dtype: tensor
    dtype:
      dtype: TypeId
      arg_handler: dtype_to_type_id
      default: None
  returns:
    y:
      dtype: tensor
  function:
    disable: True
  dispatch:
    enable: True
    Ascend: ZerosLikeExtAscend
```

<a id="ops-extend-namespace"></a>
#### 12.4.3 `ops.extend` Namespace

If ACLNN behavior differs from an existing `ops.xxx` method and compatibility changes to the existing method are not acceptable, add an extend interface.

MindSpore interface namespaces:
- `ops.xxx`, `ops.xxx_ext()`, `ops.auto_generate.xxx_ext()`, `ops.extend.xxx_ext()`
- `nn.xxx`, `nn.xxxExt()`, `nn.extend.xxx()`

<a id="existing-primitive-signature-change"></a>
#### 12.4.4 Existing Primitive Signature Changes And Interface Overloads

Existing Primitives sometimes need argument extensions, such as new PTA arguments or ACLNN-specific arguments. PTA/torch may also have same-name overloads.

Practical strategy:
1. Search the MindSpore repository for similar operators.
2. Analyze compatibility: whether the new argument can have a default, whether existing callers are affected, and whether other backends are affected.
3. Decide the route:
   - compatible optional extension -> modify existing YAML, Infer, and interface
   - incompatible change -> add an `_ext` Primitive or use `ops.extend`
4. Ensure existing functionality is not broken by running the relevant UT/ST.
5. Follow the review rules in [12.4.5 Review Rules](#api-review-rules).

For Tensor/functional overload adaptation, use `api_def` YAML with multiple entries. See [20. API Overload Adaptation](#api-overload-adaptation).

<a id="api-review-rules"></a>
#### 12.4.5 Review Rules

| Change type | Review requirement |
| --- | --- |
| No new interface and behavior unchanged | No review required |
| No new interface but behavior extended | Review required |
| New interface | Focused review required |
| Non-compatible existing interface change | Generally not allowed; special-case review required |
| New operator | Review required |
| Non-compatible existing operator behavior change | Review required |

Cases requiring review must be confirmed by the user.

<a id="code-skeletons"></a>
## 13. Code Skeletons

The skeletons below are starting points based on the auto-generate-then-customize workflow. Before using them, compare with similar code in the target directory and adjust macros, namespaces, and argument lists.

<a id="yaml-skeleton"></a>
### 13.1 Minimal YAML Template

```yaml
# ---- op_def ----
op_name: "OpName"
args:
  input:
    dtype: tensor
returns:
  output:
    dtype: tensor
dispatch:
  enable: True
  Ascend: "OpNameAscendCustomize"

# ---- api_def ----
api:
  py_method: "op_name"
  module: "mindspore.ops"

# ---- function_doc ----
function_doc:
  desc: "Brief English description of the operator."
```

<a id="pyboost-skeleton"></a>
### 13.2 PyBoost Customize Skeleton

```cpp
#include "plugin/device/ascend/kernel/pyboost/customize/op_name_ascend_customize.h"

namespace mindspore::kernel::pyboost {

tensor::TensorPtr OpNameAscendCustomize::Call(
    const tensor::TensorPtr &input_x,
    const std::optional<float> &scale) {
  auto output = std::make_shared<tensor::Tensor>(input_x->data_type(), out_shape);

  // Convert arguments here, such as tuple -> vector or None handling.
  // auto scale_val = scale.value_or(1.0f);

  // Launch through the project ACLNN wrapper.
  // LAUNCH_ACLNN(aclnnOpName, stream, input_x, scale_val, output);

  return output;
}

}  // namespace mindspore::kernel::pyboost
```

<a id="kbk-skeleton"></a>
### 13.3 KBK Kernel Skeleton

```cpp
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel/op_name_aclnn_kernel.h"

namespace mindspore::kernel {

void OpNameAclnnKernel::GetWorkSpaceInfo(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &outputs) {
  // Extract scalar/tuple/etc. arguments.
  // auto scale = inputs[1]->GetValueWithCheck<float>();

  // Get workspace.
  // GetWorkspaceForResize(aclnnOpNameGetWorkspaceSize, ...);
}

bool OpNameAclnnKernel::Launch(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &workspace,
    const std::vector<KernelTensor *> &outputs,
    void *stream_ptr) {
  // RunOp(aclnnOpName, stream, ...);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(OpName, OpNameAclnnKernel);

}  // namespace mindspore::kernel
```

<a id="bprop-builder-skeleton"></a>
### 13.4 BPROP Builder Skeleton

```cpp
REG_BPROP_BUILDER("OpName").SetBody([](const BpropBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto scale = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);

  NodePtr dx;
  if (ib->need_compute_grad_out(kIndex0)) {
    dx = ib->Emit("OpNameGrad", {input_x, out, dout, scale});
  } else {
    dx = ib->OutZeros(input_x);
  }

  auto d_scale = ib->OutZeros(scale);
  return {dx, d_scale};
});
```

<a id="pta-source-review"></a>
## 14. PTA Source Review

PTA documentation may lag behind or omit details. Before development, review the actual op-plugin source and use both docs and source as references. If docs and source conflict, do not guess; ask the user to confirm with the ACLNN/PTA owner. See [14.5 Source And Documentation Mismatch](#pta-source-doc-mismatch).

### 14.1 Key Files To Review

| File | Path pattern | What to extract |
| --- | --- | --- |
| **Function signature YAML** | `op_plugin/config/op_plugin_functions.yaml` | exact argument names, types, defaults, return structure, `op_api` / `acl_op` / `gen_opapi` route |
| **Backward registration YAML** | `op_plugin/config/derivatives.yaml` | differentiable inputs, grad function name, argument order, `output_differentiability` |
| **C++ implementation** | `op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp` and Grad variants | actual `aclnnXxx` call, preprocessing, output Tensor construction, hard-coded defaults |

### 14.2 Differences To Watch For

1. **Forward and backward argument names or order differ**
   - Example: forward uses `actual_seq_lengths_query`, backward uses `actual_seq_qlen`.
   - Example: forward has separate `layout_query` and `layout_kv`, while backward uses one `layout`.
   - Impact: MS bprop must pass arguments according to the actual backward signature.

2. **Extra or hidden backward ACLNN arguments**
   - Example: backward hard-codes `deterministic_const = true`.
   - Example: backward omits `block_table` even though forward has it.
   - Impact: MS PyBoost/KBK backward must match these hidden behaviors.

3. **Optional argument `None` handling**
   - Example: PTA passes an empty `at::Tensor()` to ACLNN when `query_rope` is `None`.
   - Example: when `query_rope` is `None`, its gradient is `at::empty({0}, ...)`.
   - Impact: MS must match `None` semantics, or ACLNN may fail or produce different behavior.

4. **Output count and output construction**
   - Example: forward returns `(output, softmax_max, softmax_sum)`.
   - Example: backward returns five gradients.
   - Impact: MS Infer and bprop must match output shape and tuple indexing.

5. **`derivatives.yaml` gradient routing**
   - Example: `result0`, `result1`, and `result2` map to forward outputs 0, 1, and 2.
   - Impact: MS `GetInput` indices and `OutZeros` placeholders must align.

### 14.3 Review Steps

1. Search the operator name in `op_plugin_functions.yaml`; extract exact forward and backward signatures.
2. Search the operator name in `derivatives.yaml`; confirm differentiable inputs and grad argument passing.
3. Locate the C++ implementation under `ops/opapi/` and inspect:
   - output Tensor shape construction
   - optional argument handling and `value_or` defaults
   - actual argument order passed to `EXEC_NPU_NO_FORMAT_CHECK_CMD` or `aclnnXxx`
   - hard-coded arguments such as `deterministic`
4. Record differences as evidence for the MS adaptation.
5. If source and docs conflict, stop and ask for confirmation. See [14.5](#pta-source-doc-mismatch).

### 14.4 Difference Record Template

```text
Operator: npu_sparse_flash_attention

Forward vs backward argument differences:
- actual_seq_lengths_query (forward) -> actual_seq_qlen (backward)
- layout_query + layout_kv (forward) -> layout (backward)
- block_table exists in forward but is not passed in backward
- return_softmax_lse exists in forward but is not passed in backward

Hidden backward behavior:
- deterministic_const = true
- when query_rope is None, d_query_rope = at::empty({0}, ...)

Output structure:
- forward: (output, softmax_max, softmax_sum), 3 outputs
- backward: (d_query, d_key, d_value, d_query_rope, d_key_rope), 5 outputs

derivatives.yaml differentiable inputs:
- query, key, value, query_rope, key_rope
- sparse_indices, block_table, and similar inputs are non-differentiable
```

<a id="pta-source-doc-mismatch"></a>
### 14.5 Source And Documentation Mismatch

Principle: when docs and source agree, use both and proceed. When they conflict, do not guess; ask the user to confirm.

Mismatch process:
1. List each difference: "docs say X, code does Y", with file path and line number.
2. Ask the user to confirm with the ACLNN/PTA owner which behavior is authoritative.
3. After confirmation, record the decision in the implementation plan or Feature document and continue.

Output template:

```text
PTA source and documentation conflict. Confirmation required.

| # | Topic | Documentation | Source behavior | File/line |
| - | ----- | ------------- | --------------- | --------- |
| 1 | ... | ... | ... | ... |

Please confirm with the ACLNN/PTA owner which behavior MS should follow.
```

<a id="infervalue-constant-folding"></a>
## 15. InferValue Constant Folding

When all inputs are known at compile time, InferValue can infer the result value directly and skip runtime computation to improve graph execution performance.

### 15.1 Implementation Options

- **Python callback**, such as concat: register an InferValue callback in `mindspore/python/mindspore/ops/operations/manually_defined/ops_def.py`.
- **C++ implementation**, such as add: implement under `mindspore/ops/infer/ops_frontend_func_impl/`.
- Prefer C++ when practical because it has better performance.

### 15.2 Verification

- Add InferValue UT cases with all-constant inputs.
- Inspect IR after running tests and confirm the output node becomes a ValueNode.

### 15.3 Applicable Scenarios

- Inputs can be determined at compile time, such as shape calculation or type-conversion helper operators.
- Most ACLNN compute operators have runtime inputs and do not need InferValue.

<a id="dynamic-shape-strategy"></a>
## 16. Dynamic Shape Categories And Strategies

For quick Infer fallback rules, see [4.2 Dynamic Shape And Dynamic Rank](#general-infer-dynamic-shape-rank).

### 16.1 Three Dynamic Shape Categories

| Type | Meaning | Typical operators | Infer strategy |
| --- | --- | --- | --- |
| **InputDynamic** | input shape unknown at compile time | most operators | set corresponding output dimensions to `kShapeDimAny` |
| **OutputDynamic (Input Value Depend)** | output shape depends on input values | `Std`, `Ones` | read with `GetScalarValue` / `GetArrayValue`; fall back when unknown |
| **OutputDynamic (Compute Depend)** | output shape must be computed at runtime | `NonZero`, `UniqueConsecutive` | allocate maximum possible size and call `SyncOutputShape` after execution |

### 16.2 InputDynamic

- If an input dimension is `-1`, set the corresponding output dimension to `-1`.
- If input rank is dynamic, such as `-2`, return dynamic rank.
- If a key scalar parameter is unknown, set dimensions depending on that parameter to `-1`.

### 16.3 Input Value Depend

Output shape may depend on an input value, scalar or array, which may be known or unknown at compile time.

- **Scalar value dependency**: use `GetScalarValue<T>()`; if `!has_value()`, fall back to dynamic rank or dynamic dimensions.
- **Array value dependency**: use `GetArrayValue<T>()`; if the whole value is unknown, fall back to dynamic rank; if only one element is unknown, set the corresponding dimension to `kShapeDimAny`.
- Example A, `Std`: output shape depends on `dim` and `keepdim`; if either is unknown, fall back to `kShapeRankAny`.
- Example B, `Ones`: output shape is directly determined by the `shape` array; unknown elements become `kShapeDimAny`.

### 16.4 Compute Depend

- Allocate the maximum possible output size estimated at compile time.
- After execution, use `Sync` plus `SyncOutputShape` to update the actual output shape.
- Override `GetUseLessOutputIdx()` when needed to avoid dump or overflow-check issues.

<a id="aclnn-callchain-analysis"></a>
## 17. ACLNN Call Chain Analysis

A PTA `torch_npu.npu_xxx()` API may call more than one `aclnnXxx` operator internally. Forward or backward may be implemented as a chain of smaller ACLNN operators. In that case, MS must inventory all sub-operators, implement missing ones, and then compose them in the same way.

### 17.1 When Call Chain Analysis Is Needed

- PTA C++ implementation contains multiple `EXEC_NPU_CMD` or `EXEC_NPU_NO_FORMAT_CHECK_CMD` calls.
- PTA C++ implementation calls other `at_npu::native::` functions.
- ACLNN docs or headers do not provide a single operator matching the PTA API.
- Backward is not a single `aclnnXxxGrad` call and is instead built from smaller operators.

<a id="aclnn-callchain-extraction"></a>
### 17.2 Extracting The Call Chain

1. Find the PTA forward C++ implementation, such as `ops/opapi/XxxKernelNpuOpApi.cpp`, and mark:
   - each `EXEC_NPU_CMD(aclnnYyy, ...)` or `OpApiFunc(aclnnYyy, ...)`
   - intermediate Tensor construction, such as `at::empty(...)` or `npu_preparation::apply_tensor(...)`
   - argument preprocessing, default filling, and `None` handling
2. Analyze the backward implementation similarly, such as `XxxGradKernelNpuOpApi.cpp` or the function referenced by `derivatives.yaml`.
3. Produce a text call chain.

```text
torch_npu.npu_foo(q, k, v, scale) forward:
  1. aclnnBarPrepare(q, k) -> intermediate_qk
  2. aclnnAttentionScore(intermediate_qk, v, scale) -> output
  3. aclnnSoftmaxLse(output) -> softmax_lse

torch_npu.npu_foo backward:
  1. aclnnAttentionScoreGrad(dout, q, k, v, softmax_lse) -> (dq, dk, dv)
```

<a id="ms-coverage-inventory"></a>
### 17.3 MS Coverage Inventory

Search each sub-operator in the MS repository.

| Target | Search key | Purpose |
| --- | --- | --- |
| YAML definition | `aclnnYyy` or corresponding op name | confirm `op_def` exists |
| C++ small-operator API | function name in `functions/auto_generate/functions.h` | confirm PyBoost composition can call it |
| Meta DSL Prim | `Prim(OpName)` or `gen_ops_primitive_*.h` | confirm KBK composition can call it |
| PyBoost implementation | `LAUNCH_ACLNN(aclnnYyy` or customize file | confirm Pynative path |
| KBK kernel | `MS_ACLNN_KERNEL_FACTORY_REG` plus class name | confirm Graph path |
| Infer | corresponding `FuncImpl` class | confirm inference exists |
| `aclnn_config.yaml` | operator name mapping | confirm dispatch mapping for Path 1 |

### 17.4 Inventory Template

```text
Target API: torch_npu.npu_foo -> mindspore.ops.foo

ACLNN call chain inventory:
| # | aclnnXxx | Purpose | MS status | Notes |
| - | -------- | ------- | --------- | ----- |
| 1 | aclnnBarPrepare | forward preprocessing | integrated | YAML/Infer/PyBoost/KBK complete |
| 2 | aclnnAttentionScore | forward main compute | partial | YAML+Infer only; missing PyBoost/KBK |
| 3 | aclnnSoftmaxLse | forward auxiliary output | missing | full flow required |
| 4 | aclnnAttentionScoreGrad | backward | integrated | no extra work |

Plan:
1. Implement #3 first through YAML -> Infer -> PyBoost -> KBK -> UT.
2. Complete PyBoost/KBK for #2.
3. Implement foo customize by composing #1, #2, and #3.
```

<a id="callchain-rollout-order"></a>
### 17.5 Rollout Order

- Implement leaf sub-operators before composite operators.
- Implement forward before backward, because backward may reuse forward sub-operators.
- Each missing sub-operator should follow the full flow from YAML to Infer to PyBoost to KBK to UT.
- Implement the composite layer after all sub-operators are available.

<a id="composite-implementation"></a>
## 18. Composite Implementation Patterns

When an operator is composed from smaller operators, MindSpore provides two mechanisms:
- **PyBoost Pynative**: compose generated C++ small-operator APIs.
- **KBK Graph**: compose operators using the Meta DSL.

<a id="composite-pyboost-pattern"></a>
### 18.1 PyBoost Composition With C++ Small-Operator APIs

In PyBoost customize code, directly call existing generated C++ API functions such as `add()`, `mul()`, or `transpose()`. Each API wraps implicit type conversion, the PyBoost call, and automatic differentiation, so you do not need to call `LAUNCH_ACLNN` manually for each sub-operator.

Key header:

```cpp
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
```

YAML:

```yaml
bprop_expander: False
dispatch:
  enable: True
  Ascend: FooAscend
```

Set `bprop_expander: False` to make the large operator rely on sub-operator autodiff rather than its own bprop expander.

If the large operator already has an independent bprop registration and `bprop_expander: False` is not set, use `RequireGradGuard(false)` around sub-operator calls to avoid duplicate autodiff.

Example:

```cpp
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

tensor::TensorPtr FooAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                     const TensorPtr &input1, const TensorPtr &input2,
                                     const FP32ImmPtr &margin, const Int64ImmPtr &reduction) {
  auto prod = mul(input1, input2);
  auto result = sum_ext(prod, dim_tuple, std::make_shared<BoolImm>(False), std::nullopt);
  auto output = div(result, sqrt(denom));

  auto reduction_val = static_cast<Reduction>(GetValue<int64_t>(reduction));
  if (reduction_val == Reduction::MEAN) {
    output = mean_ext(output, std::nullopt, std::make_shared<BoolImm>(False), std::nullopt);
  }

  op->set_outputs({output});
  return op->output(0);
}
```

Available APIs can be inferred from `mindspore/ops/op_def/yaml/`: operators with YAML and dispatch generate corresponding C++ APIs.

<a id="composite-kbk-pattern"></a>
### 18.2 KBK Composition With Meta DSL

Meta DSL uses `REGISTER_FUNCTION_OP` plus `BeginFunction`/`EndFunction` to build a graph in C++. The framework handles type inference, autodiff, and multi-platform adaptation.

Code location: `mindspore/ccsrc/frontend/operator/meta_dsl/func_op/`.

| API | Meaning | Example |
| --- | --- | --- |
| `REGISTER_FUNCTION_OP(OpName)` | Register a composed operator | `REGISTER_FUNCTION_OP(Foo, CheckFunc)` |
| `BeginFunction(Op, args...) { }` | Start composition | `BeginFunction(Foo, x, y, z) { ... }` |
| `EndFunction(Op)` | End composition | `EndFunction(Foo)` |
| `Prim(OpName)` | Reference a Primitive | `Prim(Add)` |
| `Call(prim, args...)` | Call one operator | `Call(Prim(Mul), x, y)` |
| `Value(v)` | Create a constant | `Value(0)` or `Value(kNone)` |
| `Return(out)` | Return output | `Return(output)` |
| `If(cond, true_br, false_br)` | Graph if-else | branches are lambdas |
| `Tuple(...)` / `List(...)` | Create tuple/list | `Tuple(x, y, z)` |
| `Rank(x)` / `Shape(x)` | Get rank/shape | `Rank(x)` |
| `PRIMITIVE_BPROP_REG(Op, Grad)` | Register optional bprop | `PRIMITIVE_BPROP_REG(Foo, FooGrad)` |

Example:

```cpp
REGISTER_FUNCTION_OP(CosineEmbeddingLoss, CheckCosineEmbeddingLossInputs)

BeginFunction(CosineEmbeddingLoss, input1_tensor, input2_tensor, target_tensor, margin, reduction) {
  constexpr float EPSILON = 1e-12;
  auto dim_tuple_ptr = Tuple(Rank(target_tensor));
  auto prod_sum = Call(Prim(SumExt), Call(Prim(Mul), input1_tensor, input2_tensor),
                       dim_tuple_ptr, Value(false), Value(kNone));
  auto denom = Call(Prim(Sqrt), Call(Prim(Mul), mag_square1, mag_square2));
  auto cos = Call(Prim(Div), prod_sum, denom);

  auto zeros = ZerosLike(cos);
  auto pos = Call(Prim(SubExt), OnesLike(cos), cos, Value(1));
  auto neg = Call(Prim(ClampMin), Call(Prim(SubScalar), cos, margin, Value(1)), Value(0));
  auto output_pos = Call(Prim(Select), Equal(target_tensor, Value(1)), pos, zeros);
  auto output_neg = Call(Prim(Select), Equal(target_tensor, Value(-1)), neg, zeros);
  auto output = Call(Prim(AddExt), output_pos, output_neg, Value(1));

  auto condition_none = Equal(reduction, Value(static_cast<int64_t>(Reduction::NONE)));
  auto none_true_branch = [&]() { Return(output); };
  auto none_false_branch = [&]() {
    auto condition_mean = Equal(reduction, Value(static_cast<int64_t>(Reduction::MEAN)));
    auto mean_true_branch = [&]() { Return(Call(Prim(MeanExt), output, Value(kNone), Value(false), Value(kNone))); };
    auto mean_false_branch = [&]() { Return(Call(Prim(SumExt), output, Value(kNone), Value(false), Value(kNone))); };
    Return(If(condition_mean, mean_true_branch, mean_false_branch));
  };
  Return(If(condition_none, none_true_branch, none_false_branch));
}
EndFunction(CosineEmbeddingLoss)
```

### 18.3 YAML Notes

| Field | Value | Meaning |
| --- | --- | --- |
| `bprop_expander` | `False` | large operator does not use bprop expander; sub-operators handle autodiff |
| `dispatch.enable` | `True` | enable dispatch |
| `dispatch.Ascend` | `FooAscend` | point to PyBoost customize implementation if any |

After `bprop_expander: False`, a handwritten `REG_BPROP_BUILDER` is usually not needed. Backward is composed from sub-operator bprop implementations. If custom backward is required, Meta DSL can use `PRIMITIVE_BPROP_REG`.

### 18.4 Infer Notes

- Composite operator Infer only needs to infer final output shape and type.
- It does not need to infer intermediate tensors.
- If final output shape depends on intermediate shapes, infer it directly from known inputs.

### 18.5 Layered Verification

| Stage | Verify | Method |
| --- | --- | --- |
| **Sub-operator** | each sub-operator is correct | each sub-operator's UT/ST |
| **Composite intermediate** | intermediate tensors match PTA | temporary dumps and staged PTA comparison |
| **Composite final output** | final output matches PTA | standard ST alignment |
| **Backward** | gradients are correct | backward ST and numerical gradient checks when applicable |

<a id="feature-document-reference"></a>
## 19. Feature Document

Feature documents are required for operator review and test handoff. They consolidate design, interface definition, implementation details, test plan, and acceptance results in a standard format.

<a id="feature-document-overview"></a>
### 19.1 What A Feature Document Is

The review committee uses the Feature document to decide whether the operator can be merged.

### 19.2 Standard Sections

| No. | Section | Fill time | Notes |
| --- | ------- | --------- | ----- |
| 1 | Background | Pre-B | operator source, motivation, why MindSpore needs it |
| 2 | Benchmark and API | Pre-B | PTA/Torch benchmark API and MindSpore API |
| 3 | Task list | Pre-B init, then update during development | standard task categories |
| 4 | Function and interface spec | Pre-B | formula, signature, argument descriptions |
| 5 | YAML definition | after Step 1 | `op_def` YAML |
| 6 | Constraints and types | Pre-B | device, dtype, shape, empty Tensor strategy |
| 7 | Execution modes and adaptation | after Step 4/5 | PyBoost and KBK implementation |
| 8 | PTA differences and alignment | Pre-B init, then update | functional, precision, and API semantic differences |
| 9 | Dynamic Shape/Rank support | after Step 3 | dynamic dimension/rank strategy |
| 10 | Errors and validation | after Step 3/4 | Infer/runtime checks |
| 11 | Backward BPROP | after Step 6 | bprop registration, backward API, gradients |
| 12 | Test plan | after Step 8 | UT/ST/TEST_OP coverage |
| 13 | Code change summary | after development | full paths for all added/modified files |
| 14 | Acceptance report | before handoff | documentation, function, performance, and secure-coding self-checks |

<a id="feature-document-task-categories"></a>
### 19.3 Task List Categories

Each operator should fill the task list with `new`, `modified`, `unchanged`, or `not applicable`, plus a short note.

| No. | Task item | Subitems |
| --- | --------- | -------- |
| 1 | Basic interface function | Primitive / functional / nn / tensor |
| 2 | Backend and dtype support | Ascend / GPU / CPU |
| 3 | vmap support | - |
| 4 | Dynamic Shape support | dynamic shape / dynamic rank |
| 5 | Backward support | bprop function / complex support |
| 6 | Documentation | API mapping / Chinese and English API docs |
| 7 | Functionality | empty Tensor / inf-nan / 0D-8D / other points |
| 8 | Gate tests | UT / ST / TEST_OP |
| 9 | Security and errors | error cases and error message standards |

<a id="feature-document-acceptance-tables"></a>
### 19.4 Acceptance Tables

**Documentation validation, 17 items**: interface lists, UT/ST cases, Chinese and English docs, interface description, formula, arguments, inputs, outputs, output shape relation, Raises, platform, format checks, samples, sample output, sample executability, API sandbox.

**Functional validation, 26 items**: default arguments, empty Tensor, inf/nan, dtype alignment, value range, 0D-8D coverage, dtype coverage, implicit casting, broadcasting, input constraints, forward accuracy, backward support, single backward operator, error messages, error whitelist, dynamic shape/rank, fallback disable validation, test repository regression, bf16, bprop as required, compute-dependent output shape, non-contiguous input, PTA zero-difference, existing API impact, AMP, inconsistent multi-Tensor dtype.

**Performance validation, 4 items**: broadcast performance, backward memory optimization with `SetUnusedInputs`, at least three specs, memory comparable to PTA.

**Secure coding review, 12 items**: null pointer, use-before-check, out of bounds, divide by zero, memory leak, exception-path release, `nothrow`, secure function library, type conversion overflow, redundant code, sensitive information, weak random numbers.

### 19.5 Feature Document Workflow

```text
Pre-B:
  1. Copy templates/feature-document.md.
  2. Fill background, benchmark/API, task list, function spec, constraints, and PTA alignment.
  3. Submit for design review.

During development:
  4. Backfill the corresponding sections after each workflow step.

Before test handoff:
  5. Fill code change summary.
  6. Fill all acceptance tables.
  7. Update final task list status.
  8. Submit the Feature document with the code PR.
```

### 19.6 Differences By Operator Type

| Scenario | Difference |
| --- | --- |
| **Single ACLNN operator** | standard flow; PyBoost/KBK each call one ACLNN API |
| **Composite operator** | describe the ACLNN call chain, multi-ACLNN execution, and layered verification |
| **Symbol overload, such as `==`** | describe MultitypeFuncGraph adaptation; functional/tensor task list entries are `modified` |
| **Pure Python composite without Primitive** | Primitive is `not applicable`; execution section only describes the functional layer |

### 19.7 Template Location

- Template: `templates/feature-document.md`
- References: existing Feature documents for similar operators, when available

<a id="api-overload-adaptation"></a>
## 20. API Overload Adaptation

When PTA/torch has multiple call forms with the same API name, such as `div(x, y)` and `div(x, y, *, rounding_mode=None)`, MindSpore uses multiple entries in `api_def` YAML. The framework dispatches by argument type and count.

### 20.1 Core Mechanism: api_def YAML Forwarding

In `mindspore/ops/api_def/{op_name}.yaml`, one API name can have multiple `op_yaml` entries. Pynative checks entries in order. Static graph prefers the `deprecated` branch.

| Field | Meaning |
| --- | --- |
| `op_yaml` | maps to an operator YAML under `op_def/yaml/` |
| `py_method` | Python callback name defined in `tensor_method.py` |
| `kwonlyargs` | keyword-only arguments such as `rounding_mode`; must match `op_yaml` |
| `Ascend` / `CPU` / `GPU` | backend route: `pyboost` or `py_method` |
| `interface` | `tensor`, `function`, or `tensor, function` |
| `disable_scalar_tensor` | disables scalar-to-Tensor conversion for selected arguments |

Example, type overload:

```yaml
less:
  - op_yaml: less_scalar_op.yaml
    py_method: tensor_less
    Ascend: pyboost
    CPU: py_method
    GPU: py_method
    interface: tensor, function

  - op_yaml: less_op.yaml
    py_method: tensor_less
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function
```

Example, multiple signatures with keyword-only arguments:

```yaml
div:
  - op_yaml: divs_op.yaml
    py_method: tensor_div
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    disable_scalar_tensor: other
    interface: tensor, function

  - op_yaml: div_op.yaml
    py_method: tensor_div
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function

  - op_yaml: divmods_op.yaml
    py_method: tensor_div
    kwonlyargs: rounding_mode
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    disable_scalar_tensor: other
    interface: tensor, function

  - op_yaml: divmod_op.yaml
    py_method: tensor_div
    kwonlyargs: rounding_mode
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function
```

<a id="api-overload-scenarios"></a>
### 20.2 Common Overload Scenarios

| Scenario | Description | Typical operators | api_def traits |
| --- | --- | --- | --- |
| **Different argument type** | same name, Tensor/Scalar arguments map to different `op_yaml` | `less`, `mul`, `eq` | multiple `op_yaml`, same `py_method` |
| **Argument type plus keyword-only args** | with or without keyword-only args map to different `op_yaml` | `div`, `sub`, `add` | some entries have `kwonlyargs` |
| **New/old interface compatibility** | old MS Tensor API differs from mint/ext API | `flatten`, `pow`, `sub` | includes `deprecated/*.yaml` entries |
| **Symbol alias** | two API names share one implementation | `__mul__` -> `mul`, `__truediv__` -> `div` | one-line `alias: xxx` |

### 20.3 deprecated YAML Mechanism

Use deprecated YAML when an old MS Tensor interface is incompatible with the new mint/ext interface in argument count, names, or keyword passing, and a Python callback is needed to preserve old behavior.

| File | Purpose |
| --- | --- |
| `ops/op_def/deprecated/{op_name}_method.yaml` | defines the old signature; arguments must match `tensor_method.py` |
| `ops/tensor_method.py` | implements `tensor_{op_name}` or `deprecated_tensor_{op_name}` callbacks |
| `_extends/parse/deprecated/deprecated_tensor_method.py` | registers deprecated static graph Tensor method mapping |

Priority:
- Pynative: checks `api_def` entries in order.
- KBK static graph: prefers the `deprecated` branch.

Example:

```yaml
flatten:
  - op_yaml: flatten_ext_op.yaml
    py_method: tensor_flatten
    Ascend: pyboost
    CPU: py_method
    GPU: py_method
    interface: tensor

  - op_yaml: deprecated/flatten_method.yaml
    py_method: deprecated_tensor_flatten
    Ascend: py_method
    CPU: py_method
    GPU: py_method
    interface: tensor
```

Deprecated YAML:

```yaml
flatten:
  args:
    input:
      dtype: tensor
    order:
      dtype: str
      default: "'C'"
    start_dim:
      dtype: int
      default: 0
    end_dim:
      dtype: int
      default: -1
  returns:
    output:
      dtype: tensor
```

### 20.4 Functional Overload Notes

Functional overloads for mint differ from Tensor overloads:

- Static graph has no Python callback mechanism for functional overloads.
- `py_method` and deprecated YAML are ineffective for functional overloads.
- Functional overloads only forward once based on argument type and count.

Steps:
1. Add `function` to the `interface` field in `api_def/{op_name}.yaml`.
2. Update `mint/__init__.py` so the public API imports the auto-generated entry from `functional_overload`.
3. Add matching docs under `api_def/function_doc/`, otherwise generation can fail.

### 20.5 alias Declarations

When two API names share the same PyBoost implementation, use a one-line alias.

```yaml
__mul__:
  alias: mul
```

```yaml
__truediv__:
  alias: div
```

The framework forwards alias API calls to the target API overload logic.

### 20.6 Development Notes

1. After overload migration, remove the original public method from `common/tensor.py`, move it to `tensor_method.py`, and add the `tensor_` prefix.
2. Use two-space YAML indentation.
3. Add overload tests under `tests/st/tensor/overload/test_{op_name}.py`, use `level0`, and set `jit_level` to `O0`.
4. For Ascend-only operators, set CPU/GPU to `py_method` and raise a clear exception in the callback, or use a placeholder `py_method`.
