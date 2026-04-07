---
name: op-info-test
description: Generate, isolate, and validate MindSpore Python ST op_info tests end-to-end. Use when asked to add, repair, or verify op_info-based ST coverage for one or more operators, run an op_info smoketest, execute the remote deploy-and-test loop for op_info cases, or produce remote test evidence and coverage summaries for op_info work.
---

Execute the op_info ST workflow end to end. Prefer direct execution. Ask questions only when permissions block execution or critical information is genuinely missing.

Read only the workflow file needed for the current step. Use shared reference material only when the workflow explicitly needs it.

<a id="op-info-test-end-to-end-flow"></a>
## Full End-to-End Workflow

1. Generate or update the target OpInfo cases by following [workflow/op_info_generation.md](workflow/op_info_generation.md).
2. Commit the case changes with message `op_info_test: add xxx`.
3. If remote validation is required, isolate the new cases by following [workflow/patch_out_old_tests.md](workflow/patch_out_old_tests.md).
4. Push the branch.
5. Run the remote validation loop by following [workflow/remote_deploy_and_test.md](workflow/remote_deploy_and_test.md).
6. If the functional round succeeds, run the required stability round with `--count=50`.
7. If the operator has a PTA benchmark interface or the task explicitly requires parity evidence, run [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md) for generic MS/PTA consistency validation. When the operator has a PTA benchmark interface, treat consistency validation as the default behavior even when the user did not separately request parity evidence. If no PTA benchmark interface exists, skip consistency validation even if parity evidence was requested, and record that blocker in the final summary. Unless the task explicitly narrows the scope, default to one validation driver that covers both PyNative mode and Graph mode by switching `set_context` between runs. In the Graph-mode branch, prefer a named `nn.Cell` wrapper for the target operator. Use `jit` only when a standalone Python function is genuinely a better fit.
8. If any round fails with `error_type=testcase`, fix the cases and repeat the remote validation step. If it fails with `error_type=infra`, stop changing cases and record the environment blocker.
9. After remote validation passes, remove the temporary isolation patch so the retained history contains only the clean case commit.
10. Write `op_info_test_{op_name}_summary.md` in the working directory and do not add it to git. Include covered scenarios, uncovered scenarios, blocking reasons, remote job evidence, and MS/PTA consistency evidence when applicable.

<a id="op-info-test-constraints"></a>
## Execution Constraints

- Reuse the bundled workflows, scripts, and templates. Do not rewrite the process from scratch.
- Do not change tests unrelated to the target API.
- Before every rerun, confirm the temporary isolation patch and commit history are in the expected state.
- Treat [remote_deploy_and_test_dev.md](remote_deploy_and_test_dev.md) as implementation reference only, not as execution input.
- When a PTA benchmark exists, follow [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md) and keep its generated driver, `.npy` artifacts, `case_spec` files, and comparator summaries as local-only validation assets.
- For MS/PTA consistency Graph-mode validation, treat `context.set_context(mode=ms.GRAPH_MODE)` as a mode switch only. Do not assume it graph-compiles arbitrary Python helpers on its own. Graph paths should be authored as named `nn.Cell` or named `@jit` callables, and Graph-mode probes should be run from a real `.py` file rather than temporary stdin source.
- Mark the workflow complete only when required coverage is present, the functional round succeeds, the `--count=50` stability round succeeds, the final successful `summary.json` has no failed cases, and the required MS/PTA consistency strategy passes when applicable.

## Summary Requirement
- Write the summary with explicit `covered`, `not covered`, and `blocking reason` status. Include error log for failed cases.
- When MS/PTA consistency validation is required, include the evidence required by [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md#ms-pta-consistency-reporting).

