---
name: op-info-test
description: Generate and locally validate MindSpore Python ST op_info tests. Use when asked to add, repair, or verify op_info-based ST coverage for one or more operators, or run an op_info smoketest on a local Ascend environment.
---

Execute the op_info ST workflow end to end. Prefer direct execution. Ask questions only when permissions block execution or critical information is genuinely missing.

Read only the workflow file needed for the current step. Use shared reference material only when the workflow explicitly needs it.

## Default Path

For most new aclnn operators, the default path is only:

1. Use [workflow/dtype_support_discovery.md](workflow/dtype_support_discovery.md) to run the quick applicability checklist, confirm the operator is suitable for the current op_info path, and then determine dtype support. Default to the PTA runtime probe.
2. Use [workflow/op_info_generation.md](workflow/op_info_generation.md) to backfill or update OpInfo registration with the probed dtype set. Treat the probe result as the candidate writeback set, not as the final answer.
3. Run the MindSpore opinfo case tests only after the dtype backfill step. Before any validation, confirm the current machine is an Ascend environment. If it is not, stop and ask the user to provide a validation machine. Do not silently switch to remote validation.
4. If the required validation matrix fully passes after backfill, the task is complete. If not, enter the fixed failure-triage order described below.

Treat helper templates as optional tools, not as mandatory steps in the main path.

<a id="op-info-test-end-to-end-flow"></a>
## Full End-to-End Workflow

1. For a new or not-yet-registered operator, first use [workflow/dtype_support_discovery.md](workflow/dtype_support_discovery.md) to run the quick applicability checklist, decide whether the API fits the current op_info workflow, and then determine dtype support. Default to the PTA runtime probe. Use the documentation lookup path only when the user explicitly narrows the task or the runtime probe is blocked.
2. Generate or update the target OpInfo cases by following [workflow/op_info_generation.md](workflow/op_info_generation.md), and backfill the dtype declaration from the probe result before test execution. Treat this as a candidate writeback step.
3. Before validation, confirm the current machine is an Ascend environment with a usable local MindSpore Ascend setup. If not, stop and ask the user to provide a validation machine. Do not silently switch to remote validation.
4. Run the MindSpore opinfo case tests for the required frontend matrix: requested API form x mode (PyNative/KBK) x shape type (static/dynamic).
5. If the operator has a PTA benchmark interface or the task explicitly requires parity evidence, run [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md) after the opinfo case tests. Unless the task explicitly narrows the scope, default to one validation driver that covers both PyNative mode and Graph mode by switching `set_context` between runs. In the Graph-mode branch, prefer a named `nn.Cell` wrapper for the target operator. Use `jit` only when a standalone Python function is genuinely a better fit.
6. If the task explicitly requires stability evidence, or if the newly added case looks prone to flakiness, rerun the passing test command with `--count=50` as an optional post-pass stability round. Treat this as additional evidence, not as part of the default minimum path.
7. If the required matrix fully passes after the candidate backfill, mark the task complete and record that the OpInfo registration was added or updated successfully.
8. If any round fails, follow the fixed failure-triage order: first verify testcase correctness, then cross-check aclnn/MS/PTA documentation, then locate whether the issue belongs to testcase authoring, tool/framework logic, OpInfo-path limitations, operator implementation, or environment setup.
9. If the failure is caused by testcase authoring or a safe tool-side bug, fix it and rerun. If MindSpore opinfo validation still diverges from the PTA probe result after testcase/tool fixes, record and clarify that discrepancy instead of silently shrinking the dtype declaration.
10. If the API or failure mode is limited by the current op_info workflow itself, record that limitation explicitly instead of forcing a registration or forcing a negative-case matrix that the framework cannot express cleanly.
11. Write `op_info_test_{op_name}_summary.md` in the working directory and do not add it to git. Include covered scenarios, uncovered scenarios, dtype discovery evidence, dtype writeback evidence, opinfo validation evidence, explicit coverage-gap counts, blocking reasons, and the final user-facing conclusion category.

## Remote Validation Boundary

- Remote validation is not part of the default execution contract of this skill.
- Use remote validation only when the user explicitly asks for remote validation or explicitly asks for remote evidence.
- Remote execution is only an adapter for running the same validation matrix on a user-provided machine. It must not decide which OpInfo cases to add, which dtype set to declare, or how failures are classified.
- When the user provides a remote validation environment, follow [workflow/remote_validation_adapter.md](workflow/remote_validation_adapter.md): copy the prepared local changes to that environment, source the provided environment setup, run the same validation commands there, and bring the logs/results back for the normal op_info analysis loop.

<a id="op-info-test-constraints"></a>
## Execution Constraints

- Reuse the bundled workflows, scripts, and templates. Do not rewrite the process from scratch.
- Do not change tests unrelated to the target API.
- Read the current repository layout before editing. In newer MindSpore op_info code, `OpInfo` registration may stay in `tests/st/ops/share/_op_info/op_database.py` while wrappers and custom sample/error builders live in `tests/st/ops/share/_op_info/op_wrappers.py` and `tests/st/ops/share/_op_info/op_sample_inputs.py`.
- If the API fails the applicability gate for the current op_info workflow, do not force an OpInfo registration. Record the blocker and switch to the appropriate non-op_info validation plan.
- For new aclnn operators, treat runtime dtype probing as the default source of truth for `dtypes_*`; doc-only lookup is supplemental.
- For dtype probing, default to the PTA interface because new MindSpore interfaces are brought in against the PTA benchmark surface. Use the MindSpore interface only when the user explicitly asks for MindSpore-side probing or the PTA path is unavailable.
- For current aclnn OpInfo tasks, default to filling only `dtypes_ascend910b` / `dtypes_backward_ascend910b`. Leave `dtypes_ascend`, `dtypes_cpu`, and `dtypes_gpu` empty unless the task explicitly requires those platforms.
- Treat the probe-supported forward dtype set as the default candidate writeback set. It becomes the final writeback only after the required OpInfo validation matrix passes.
- Do not assume `not_support_dtypes` is the complement of `dtypes_ascend910b`. Only place a dtype in `not_support_dtypes` when a representative MindSpore-side negative case is actually expected to fail there.
- Treat the user-requested API surface as the default scope. If the task asks for `mint.xxx`, do not automatically add sibling API forms such as `Tensor.xxx`, `ops.xxx`, or `nn.xxx` unless the user explicitly asks for multi-form coverage or the task requirement clearly says to cover all public API forms.
- Validation defaults to local execution only.
- Before every local validation run, confirm the current machine is an Ascend environment. If not, stop and ask the user for a validation machine.
- Do not silently switch to remote validation because the local machine is missing Ascend support.
- If the user explicitly provides a remote validation machine, use it only as the execution location for the same op_info validation step.
- When a PTA benchmark exists, follow [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md) and keep its generated driver, `.npy` artifacts, `case_spec` files, and comparator summaries as local-only validation assets.
- For MS/PTA consistency Graph-mode validation, treat `context.set_context(mode=ms.GRAPH_MODE)` as a mode switch only. Do not assume it graph-compiles arbitrary Python helpers on its own. Graph paths should be authored as named `nn.Cell` or named `@jit` callables, and Graph-mode probes should be run from a real `.py` file rather than temporary stdin source.
- If you need fast iteration on one newly added operator, you may temporarily isolate that operator by following [workflow/patch_out_old_tests.md](workflow/patch_out_old_tests.md). Treat this as a debugging tactic only: remove the temporary patch before final full validation or before leaving the repo in a reviewable state.
- For `other`-type operators, validate hook completeness explicitly: `op_basic_reference_inputs_func`, `op_extra_reference_inputs_func`, `op_dynamic_inputs_func`, `op_error_inputs_func`, and any needed loss-override fields or `not_support_dtypes`. Also check whether the operator belongs in `other_op_db`, `other_op_kbk_db`, `other_op_error_db`, and `other_op_error_kbk_db`.
- For parameter-rich `other` operators, do not accept a generic one-shot sample generator. Load [workflow/other_family_guardrails.md](workflow/other_family_guardrails.md) and reuse the nearest family pattern as a structure guide, not as a copy-paste template.
- Mark the workflow complete only when required coverage is present, local validation succeeds on the required matrix, and the required MS/PTA consistency strategy passes when applicable.

## Summary Requirement
- Write the summary with explicit `covered`, `not covered`, and `blocking reason` status. Include error log for failed cases.
- Report coverage-gap counts separately for:
  - `op_error_inputs_func is not set`
  - `op_dynamic_inputs_func is not set`
- Do not classify a run as `fully_validated` when either of the above coverage-gap counters is non-zero.
- Include whether dtype support came from `probe_verified`, `doc_derived`, or `probe_vs_doc_conflict`.
- If the dtype probe reported `sample_or_function_issue`, say whether it was fixed as a sample problem or kept as an operator functional blocker.
- If the API is not suitable for the current op_info workflow, say so explicitly and record the recommended alternative validation path.
- When MS/PTA consistency validation is required, include the evidence required by [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md#ms-pta-consistency-reporting).
- The final user-facing conclusion must land in one of these categories:
  - `fully_validated`
  - `validated_with_coverage_gaps`
  - `blocked_by_testcase_tool_framework`
  - `blocked_by_operator_environment`

