# Operator ST Test Case Generation

<a id="op-info-generation-goal"></a>
## Goal

Add Python ST coverage for a new operator, cover different cases including functionality, accuracy, and dynamic shape.

**Important:** Similar operator test cases may already exist in the repository, but their scenario coverage may be incomplete. When using similar operators as references, do not treat their coverage as the target coverage. You must strictly follow the scenario coverage requirements in this document.

<a id="op-info-generation-inputs"></a>
## Inputs

- **API name**: Use the API name to collect relevant information, such as the operator API definition YAML and operator API documentation.
- **Torch counterpart API**: the testing benchmark.
- **Dtype discovery result**: Prefer the runtime probe summary from [dtype_support_discovery.md](dtype_support_discovery.md). Use doc-derived dtype lists only when the probe is explicitly skipped or blocked.

**Scope note:** The requested API name is the default scope. If the task asks for `mint.xxx`, only add the `mint.xxx` OpInfo unless the user explicitly asks for additional API forms such as `Tensor.xxx`, `ops.xxx`, or `nn.xxx`.

<a id="op-info-generation-outputs"></a>
## Outputs

> **⚠️ The following output is required**

| Type | File Location |
| --- | --- | --- | --- |
| **Python ST** | `tests/st/ops/share/_op_info/op_database.py` (OpInfo registration) |
| **Python ST companion helpers (when the repo uses split layout)** | `tests/st/ops/share/_op_info/op_sample_inputs.py` and `tests/st/ops/share/_op_info/op_wrappers.py` |

---

## Basic Testing Principles

The behavior of the tested API should fully align with the benchmark API.
- If the benchmark API supports certain inputs and behaviors, the tested API should support them as well.
- If the benchmark API does not support certain inputs and behaviors, the tested API does not need to support them.

<a id="op-info-generation-steps"></a>
## Execution Steps

> The current ST uses the **op info testing framework**. The core operation is registering OpInfo in `op_database.py`. In newer repo layouts, the registration stays there while custom wrappers and sample/error builders live in `_op_info/op_wrappers.py` and `_op_info/op_sample_inputs.py`. Writing separate standalone test files is not allowed. For framework details, see [`../_shared/reference.md` 8.2 ST op-info testing framework](../../_shared/reference.md#testing-st-opinfo).

Before writing `dtypes_*`, first follow [dtype_support_discovery.md](dtype_support_discovery.md):

- For new aclnn operators, default to the runtime probe.
- Keep documentation lookup as a supplemental source, not the default source of truth.
- Apply the applicability gate first. If the API is not suitable for the current op_info workflow, do not create a forced OpInfo registration.

For most tasks, that is enough.

For validation, the default path is local-only:

- Before running any validation, first confirm the current machine is an Ascend environment with a usable local MindSpore Ascend setup.
- If the current machine is not an Ascend validation machine, stop and ask the user to provide a validation machine.
- Do not silently switch to remote validation unless the user explicitly asks for remote validation.

When the task needs faster iteration on one newly added operator, you may temporarily isolate that operator by following [patch_out_old_tests.md](patch_out_old_tests.md). Treat that patch as a short-lived local or remote debugging aid only. Remove it before final full validation and before leaving the repo in a reviewable state.

**There are three scenarios:**

| Applicable Scenario | Action |
| --- | --- |
| Regular operators such as Unary/Binary/Reduction | Add OpInfo in `op_database.py` -> add it to the corresponding `xxx_op_db` -> it is automatically included in frontend parameterized test cases |
| Operators requiring custom test logic | Inherit from `OpsFactory` to build a custom test suite + create a new frontend test file |
| APIs not suitable for current op_info workflow | Do not register OpInfo just to satisfy the workflow. Record the blocker and move to an alternative validation plan. |

<a id="op-info-generation-common-ops"></a>
### Regular Operators Such as Unary/Binary/Reduction

For Unary/Binary/Reduction operators, `op_info.py` already provides rich common input-generation helpers (various shape combinations, broadcasting, discontiguous tensors, special values, extreme values, and so on). Once OpInfo is registered, these scenarios are covered automatically.

1. **Determine the operator's OpInfo category**: Unary -> `UnaryOpInfo`, Binary -> `BinaryOpInfo`, Reduction -> `ReductionOpInfo`, others -> `OpInfo`.
2. **Add an OpInfo instance in `op_database.py`**: configure `name`, `op`, `ref`, and write the candidate forward dtype set into `dtypes_ascend910b` by default. Leave `dtypes_ascend`, `dtypes_cpu`, and `dtypes_gpu` empty unless the task explicitly requires those platforms. If the probe shows a stable narrower backward dtype set and the task requires encoding it, set `dtypes_backward_ascend910b`; otherwise leave backward dtype fields unset.
3. **Add the operator name to the corresponding `xxx_op_db` list** (for example, `binary_op_db`, `unary_op_db`).
4. **If custom input scenarios are needed**: implement `op_basic_reference_inputs_func` / `op_extra_reference_inputs_func` and return a list of `OpSampleInput`.
5. **Decide whether it should be added to `xxx_op_kbk_db`** (see the constraints below).
6. **Verify coverage**: confirm that the frontend test file (for example, `test_binary_ops.py`) includes the new operator in its parameterized cases.

> **Constraints for adding operators to KBK lists (`xxx_op_kbk_db`):**
>
> KBK scenarios are relatively time-consuming, so not every operator needs to be included. Add an operator to the corresponding `xxx_op_kbk_db` (such as `binary_op_kbk_db`, `unary_op_kbk_db`, `reduction_op_kbk_db`, and so on) only in the following cases, so that the frontend test files run KBK forward/backward/dynamic-shape cases:
>
> - The operator has **relatively complex dynamic shape inference logic** (for example, output shape depends on input values or uses multi-branch inference), which can be verified by checking the operator infer code.
> - The operator uses a **composite implementation** (multiple operator calls chained together in PyBoost/aclnn kernelmod).
> - The operator includes **frontend API overloading** (there is an API YAML definition in `mindspore/ops/api_def`).
>
>
> **Cases where it does not need to be added:**
> - Simple passthrough operators (single ACLNN call, no parameter preprocessing)
> - The KBK list already contains **an operator with the same type or implementation pattern**. For example, if `unary_op_kbk_db` already includes `mint.tanh`, then similar trigonometric operators such as `mint.cosh` do not need to be added again.

### Operators Requiring Custom Test Logic

For **other-type operators** (added to `other_op_db`), you must **write input-generation functions manually** and pass them to OpInfo through `op_basic_reference_inputs_func` and `op_extra_reference_inputs_func`.

In the current repo layout, do not assume those helpers belong inline in `op_database.py`. Prefer the repo's active pattern:

- keep the `OpInfo` registration and db-list wiring in `op_database.py`
- place wrappers in `_op_info/op_wrappers.py`
- place basic/extra/dynamic/error sample builders in `_op_info/op_sample_inputs.py`

If the target branch still uses the older all-in-one layout, follow that branch's existing style instead of forcing the split layout.

This path is still intended for **operator-level** APIs. It can handle special operators that need wrappers, custom inputs, sequence inputs, or representative dynamic scenarios. It is **not** the default path for strongly stateful APIs such as optimizers, distributed/process-group controls, or other interfaces whose real behavior is dominated by side effects across steps.

#### Quality Guardrails For Other-Type Operators

When generating custom sample inputs for `other_op_db`, use the following rules so the generated cases stay representative instead of becoming ad hoc:

1. **Start from the closest existing family pattern, not from a blank page**: reuse the nearest operator with the same API shape and helper layout. In newer branches this usually means searching `op_database.py`, `op_sample_inputs.py`, and `op_wrappers.py` together before writing anything new.
2. **Keep wrappers thin and deterministic**: wrappers should only express the public API call. For module-style operators, set weights/bias/other parameters explicitly instead of relying on random initialization or hidden state.
3. **Separate stable samples from edge samples**:
   - `op_basic_reference_inputs_func`: default parameters, one or two stable valid shape/parameter variants, and the main differentiable path if applicable.
   - `op_extra_reference_inputs_func`: discontiguous/layout variants, empty tensors, special values, and other edge behaviors that are meaningful but less stable.
4. **One behavior axis per yielded sample**: each `OpSampleInput` should have a descriptive `sample_name` and should primarily test one thing at a time (shape variation, parameter boundary, empty input, special value, layout, and so on).
5. **Build multi-input coverage in layers**: start with same-dtype and same-shape happy paths. Add broadcasting, mixed dtype, rank mismatch, or parameter-coupled cases only when the operator contract says they matter.
6. **Do not silently omit dynamic or error hooks**: if `op_dynamic_inputs_func` or `op_error_inputs_func` is missing, treat that as an explicit coverage gap in the task summary instead of counting the operator as fully validated.
7. **Avoid overfitting to the probe result**: the probe tells you which dtypes are candidates. It does not tell you which custom sample matrix is representative. Keep the custom matrix compact and behavior-driven.
8. **For parameter-rich `other` ops, prefer family-pattern reuse over generic synthesis**: if the target operator belongs to a complex family such as `conv*`, `linear`, `interpolate`, pooling, normalization, or loss/module wrappers, load [other_family_guardrails.md](other_family_guardrails.md) and follow the nearest family pattern instead of inventing a one-off minimal sample builder.
9. **Wire the coverage groups deliberately**: for `other` ops, check not only `other_op_db`, but also whether the operator should appear in `other_op_kbk_db`, `other_op_error_db`, and `other_op_error_kbk_db`.

<a id="op-info-generation-not-applicable"></a>
### APIs Not Suitable For The Current OpInfo Workflow

Do not try to force every API into `op_database.py`.

Treat the API as **not suitable for the current op_info workflow** when one or more of the following are true:

- the primary behavior is parameter or state mutation across training steps
- the most important correctness signal comes from multi-step evolution, not one forward/backward call
- the API depends on distributed environment setup or communication state
- the reference benchmark cannot be meaningfully compared as one operator-level execution

Typical examples:

- optimizer APIs such as `Adam`, `AdamW`, `SGD`
- process-group or distributed coordination APIs
- other environment- or lifecycle-driven interfaces

When the API is not suitable:

1. Do not write a placeholder or awkward OpInfo registration.
2. Record the blocker explicitly in the task summary.
3. Recommend or switch to a more appropriate validation path, such as:
   - stateful parity tests for optimizers
   - integration tests for distributed APIs
   - lower-level update-op tests when the real target is an underlying kernel

#### Test Case Coverage Scenarios
- [ ] `[MUST]` **Default-parameter scenario validation**: call forward and backward with all default parameter values to confirm the basic path works.
- [ ] `[MUST]` **Dynamic shape self-validation**: the frontend test file calls the `test_op_dynamic` method from `OpsFactory`. Representative dtype coverage is enough in the default workflow; do not build a full dynamic dtype matrix unless the task explicitly requires it.
- [ ] `[MUST]` **Empty tensor input**: verify whether forward/backward with empty tensors is supported or raises the correct error.
- [ ] `[MUST]` **Full static forward dtype coverage**: every dtype declared as supported by the operator should be justified by the dtype discovery result and be covered by the static forward path.
- [ ] `[MUST]` **Backward dtype discovery review**: check the runtime probe result for backward dtype support. Do not add extra custom backward-only dtype matrices unless the task explicitly needs them; use `dtypes_backward_*` only when the narrower backward set must be encoded.
- [ ] `[MUST]` **Input dimension coverage**: include both valid dimensions (covering 0D, 8D, and one intermediate-size dimension if supported) and invalid dimensions.
- [ ] `[MUST]` **Input value range validation**: fully cover boundary values, extreme values (very large/very small), and enumerated parameters such as `margin`/`reduction`.
- [ ] `[MUST]` **Cross-input constraint validation**: shape match/mismatch, dtype same/different, and rank same/different.
- [ ] `[MUST]` **Assert exact error messages for exception cases**: exception scenarios must assert the specific message of `TypeError`/`ValueError`/`RuntimeError`.
- [ ] `[MUST]` **Multi-layout coverage**: if the operator supports multiple layouts (such as BSND/TND/PA_BSND), cover forward and backward for every layout combination.
- [ ] `[MUST]` **Discontiguous tensors**: construct discontiguous inputs using `transpose`/`permute` and verify correctness.
- [ ] `[MUST]` **Special-value robustness**: validate `inf`/`-inf`/`nan` scenarios, at minimum ensuring no crash and correct shape/flow behavior.
- [ ] `[SHOULD]` **Variable-length sequences with multiple batches**: if parameters such as `actual_seq_len` are involved, cover multiple batches plus variable-length scenarios.
- [ ] `[MUST]` **bf16 scenarios**: confirm bf16 support. If supported, test accuracy; otherwise include exception cases. Promote to `float32` before comparison.
- [ ] `[MUST]` **Implicit type conversion**: confirm whether automatic promotion is supported when input dtypes differ; if not, include representative exception cases.
- [ ] `[MUST]` **Broadcasting**: confirm whether shape broadcasting between inputs is supported; if not, include exception cases.
- [ ] `[MUST]` **Inconsistent dtype across multiple Tensor inputs**: confirm whether the operator supports different dtypes across multiple Tensor inputs; if not, include representative exception cases. This is not required for operators that do not take multiple Tensor inputs.

| Required Scenario | How to Write It | Example |
| --- | --- | --- |
| **Multiple shapes** (including 0D scalar, 1D, intermediate 2D-3D, and high-dimensional) | Multiple `yield` statements with different shapes | `make_arg(())`, `make_arg((S,))`, `make_arg((S,M,S))` |
| **Empty tensor** (one dimension is 0) | Include 0 in the shape | `make_arg((0, S))`, `make_arg((S, 0, M))` |
| **Discontiguous tensor** | Use the `discontiguous=True` parameter | `make_tensor(shape, discontiguous=True)` |
| **Boundary parameter values** | Cover extreme/boundary parameter values | `dim=0`, `dim=-1`, `dim=last dimension`; `p=1`, `p=2`, `p=inf` |
| **Large tensor** | At least one relatively large shape | `make_arg((LARGE_DIM_SIZE, M))` |

Implementation reference: follow the patterns of `basic_reference_inputs_binary_op_common_func` and `_generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func` in `op_info.py`.

If the operator supports `op_extra_reference_inputs_func` (extra accuracy scenarios) or `op_dynamic_inputs_func` (dynamic shape/rank), implement them by following similar patterns in `op_info.py`.

#### Test Matrix and Stability
- [ ] `[MUST]` **Test matrix coverage**: backend x mode (Pynative/KBK) x shape type (static/dynamic) for the requested API form.
- [ ] `[MUST]` **API-form expansion only when requested**: do not automatically expand `mint.xxx` to sibling forms such as `Tensor.xxx` / `ops.xxx` / `nn.xxx` unless the task explicitly asks for multi-form coverage or all public API forms.
- [ ] `[MUST]` **Backoff-disabled validation**: all cases must pass under `export MS_DISABLE_KERNEL_BACKOFF=1` to prevent fallback to non-ACLNN paths.
- [ ] `[MUST]` **Local Ascend precheck before validation**: validate locally by default, and run validation only after confirming the current machine is an Ascend environment. If not, stop and ask the user to provide a validation machine instead of silently switching to remote.
- [ ] `[MUST]` **Coverage-gap accounting**: report `op_error_inputs_func is not set` and `op_dynamic_inputs_func is not set` as separate coverage gaps. A run with either gap cannot be reported as `fully_validated`.
- [ ] `[SHOULD]` **Optional stability rerun for flaky-looking cases**: after the main matrix passes, rerun the same command with `--count=50` when the user explicitly asks for stability evidence or when the new case looks prone to intermittent failures.

#### Failure Handling Order

If the candidate OpInfo writeback fails validation, do the following in order:

1. **Check whether the testcase itself is correct**: inspect wrappers, sample builders, error inputs, and any temporary driver logic before changing declared dtypes.
2. **Cross-check the dtype surface again**: compare aclnn documentation, MindSpore API documentation, and PTA benchmark documentation to confirm whether the expected dtype support is actually aligned.
3. **Classify the failure cause**: decide whether it belongs to testcase authoring, generic tool/framework logic, current op_info-path limitations, operator implementation, or environment setup.
4. **Mark opinfo-path limitations explicitly**: if the current `OpInfo` / `OpsFactory` path cannot express the needed case cleanly, record that limitation instead of forcing an awkward registration or an overfit negative matrix.
5. **Rerun only after the right fix**: testcase/tool issues may be fixed and rerun; operator-side or environment-side blockers must be documented with reasons and handling advice rather than hidden by changing the candidate dtype set.

#### Example Failure Drill: `mint.add`

Use `mint.add`-style failures as the default rehearsal for this triage order:

1. **Check the failing sample first**: if the failure only appears on an existing special-value sample such as complex `inf`/`nan`, inspect the sample intent, reference comparator behavior, and whether the case is still representative before touching `dtypes_*`.
2. **Check documentation next**: confirm whether aclnn, MindSpore API docs, and PTA docs actually disagree on the dtype surface, or whether the failure is unrelated to dtype support.
3. **Classify the real owner of the failure**:
   - special-value comparison mismatch in an existing reference sample -> usually `blocked_by_testcase_tool_framework`
   - KBK dynamic backward adapter/transform error -> usually `blocked_by_testcase_tool_framework`
   - genuine unsupported dtype or operator runtime error after sample/doc review -> `blocked_by_operator_environment`
4. **Do not shrink the candidate dtype set just to make the failing sample disappear**: fix the sample/framework issue first, rerun, and only then decide whether the dtype declaration itself needs to change.

#### Dtype-Scope Notes

- Think of the dtype probe as the **discovery tool** and OpInfo registration as the **steady-state test configuration**.
- For current aclnn OpInfo tasks, the default writeback is `dtypes_ascend910b` and, when needed, `dtypes_backward_ascend910b`. Leave CPU/GPU/910A fields empty unless the task explicitly requires them.
- The default order is: remote/runtime dtype probe first, then backfill `dtypes_*`, then run the MindSpore opinfo case tests. If the later opinfo tests diverge from the probe result, record and clarify the discrepancy.
- The candidate backfilled dtype set becomes the final writeback only when the required OpInfo validation matrix passes.
- Dynamic shape/rank does **not** need full dtype coverage in the default workflow.
- Negative cases do **not** need a full unsupported-dtype matrix; representative unsupported-dtype cases are enough.
- The runtime dtype probe is broader than the default OpInfo test matrix. Use it to determine declaration boundaries, not to force duplicate custom test cases for every discovered dtype.
- Binary operators already have generic mixed-dtype/type-promotion coverage through the shared binary test path. Other multi-input operators do **not** have a generic mixed-dtype matrix by default, so add representative custom mixed-dtype samples only when the operator claims such support or needs representative negative cases.
- `not_support_dtypes` is not the automatic complement of `dtypes_ascend910b`. Use it only for dtypes that are expected to fail through representative MindSpore-side negative cases.
<!-- - [ ] `[SHOULD]` **Regression of existing tests in the test repository**: if ST cases already exist in the test repository, confirm that they all PASS. -->

<!-- #### Functional Compliance Confirmation
- [ ] `[MUST]` **Does not affect existing APIs**: adding a new operator/primitive must not cause operators or existing ops APIs to call the new primitive unless that is the intended design.
- [ ] `[SHOULD]` **AMP mixed precision**: confirm whether it is already supported or not applicable (new Primitives need attention to `amp_white/black_list`).
- [ ] `[SHOULD]` **Inconsistent dtype across multiple Tensor inputs**: for multi-input operators, confirm whether different input dtypes are supported.
- [ ] `[SHOULD]` **Whether output shape depends on computation results**: if the output is compute-dependent, a `SyncOutputShape` mechanism is required. -->


<!-- <a id="op-info-generation-bitwise-validation"></a>
### Zero-Bit Accuracy Validation ([`../_shared/reference.md` 14.1 Zero-Bit Accuracy](../../_shared/reference.md#bitwise-accuracy-validation), when needed)

- Fix the random seed and save outputs as `.npy`.
- Use `md5sum` to compare the output hashes of MS/PTA.

<a id="op-info-generation-memory-validation"></a>
### Memory Alignment Validation ([`../_shared/reference.md` 14.2 Memory Usage Alignment](../../_shared/reference.md#memory-alignment-validation), when needed)

- MS: `mindspore.runtime.max_memory_allocated()`
- PTA: `torch_npu.npu.max_memory_allocated()`
- Measure at the same stage. -->

### Extra Setting

- For function that doesn't have gradient, set `is_differentiable=False` in the OpInfo.

---

<a id="op-info-generation-gate"></a>
## Mandatory Check Before Completing Step 8 (Cannot Be Skipped)

**Before marking Step 8 as complete, every item in the following checklist must be confirmed one by one:**

```text
Test Output Checklist:

Python ST (OpInfo Registration):
  - Registration file: tests/st/ops/share/_op_info/op_database.py
  - Has OpInfo been registered? ✅ Yes (operator name: ___) / ❌ No (reason: ___)
  - Has it been added to the corresponding `xxx_op_db` list? ✅ Yes / ❌ No
  - Is it covered by frontend parameterized test cases? ✅ Yes (test file: ___) / ❌ No
  - If custom inputs are required: has `inputs_func` been implemented? ✅ Yes / ⏭ Not needed
  - 🚫 Was a standalone test script created? This must be No. If one was created by mistake, delete it and migrate it into OpInfo.
```

> If the Python ST status is ❌, **you must explain the reason and pause until the user confirms before continuing**.
> Silent skipping is not allowed.

<a id="op-info-generation-success-criteria"></a>
## Success Criteria

- [ ] **Python ST OpInfo is registered and included in frontend parameterized test cases** (automatically covering multiple modes, forward accuracy, and dynamic shape)
- [ ] Covered scenarios: dynamic shape / static shape / discontiguous tensor / empty tensor / special values
- [ ] Dtype declaration source is clear: `probe_verified`, `doc_derived`, or `probe_vs_doc_conflict`
- [ ] If the API was not suitable for the current op_info workflow, that decision and the alternative validation path are explicitly documented instead of forcing a registration
- [ ] Coverage gaps from missing `op_error_inputs_func` / `op_dynamic_inputs_func` are explicitly counted and reported
- [ ] The final result reported to the user is explicit: `fully_validated`, `validated_with_coverage_gaps`, `blocked_by_testcase_tool_framework`, or `blocked_by_operator_environment`
