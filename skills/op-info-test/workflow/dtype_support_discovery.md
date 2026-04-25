# Operator Dtype Support Discovery

<a id="dtype-support-goal"></a>
## Goal

Determine which dtypes are actually supported by a target operator before writing or updating OpInfo registration.

The main use case is simple: determine the forward/backward dtype set for a **new or not-yet-registered operator** before writing `dtypes_*` in `op_database.py`.

Before running dtype discovery, first confirm that the API actually fits the current op_info workflow.

<a id="dtype-support-default"></a>
## Default Strategy

Use **runtime probing** as the default strategy.

- Start from the dtype candidates declared by the aclnn documentation and MindSpore API documentation.
- Then run the **full dtype forward/backward probe** to determine which dtypes actually execute successfully.
- Default to the **PTA interface** for runtime probing, because new MindSpore interfaces are normally brought in against the PTA benchmark surface. Switch to the **MS interface** only when the user explicitly asks for MindSpore-side probing or when the PTA path is unavailable.
- Treat the documentation as the candidate source, not as the final truth.

Use **documentation-only** discovery only when:

- the user explicitly narrows the task to doc lookup only, or
- the runtime probe is blocked by environment issues and you need a temporary fallback.

If runtime probing is skipped, record that the dtype set is **doc-derived and not runtime-verified**.

<a id="dtype-support-applicability-gate"></a>
## Applicability Gate

Before collecting dtypes, check whether the target API matches the current op_info execution model.

### Quick Checklist

Answer these four questions first:

1. Can the API be represented as **one operator-level call** with representative tensor inputs and tensor outputs?
2. Can correctness be judged from **one execution** plus optional backward, instead of from multi-step state evolution?
3. Can you build a **single-execution reference path** for comparison?
4. Is the main behavior **tensor computation**, not optimizer state updates, distributed coordination, global environment control, or object lifecycle management?

If **all four answers are Yes**, continue with the current op_info workflow.

If **any answer is No**, stop the dtype discovery path for this skill, record `not_applicable_to_opinfo`, and switch to the appropriate alternative validation path.

The current op_info path is a good fit when the API can be expressed as:

- one operator-level callable
- representative tensor inputs and outputs
- optional backward comparison on tensor inputs
- a reference path that can be compared in a single execution

Treat the API as **not applicable to the current op_info workflow** when its core semantics are instead dominated by:

- state mutation or parameter updates across steps, such as `optimizer.step()`
- multi-step sequence behavior that cannot be judged from one forward/backward execution
- distributed or process-group side effects
- global environment control rather than operator outputs
- object lifecycle management rather than tensor computation

If the API is not applicable:

1. Stop the dtype discovery path for this skill.
2. Record the status as `not_applicable_to_opinfo`.
3. Record the blocking reason and the recommended alternative validation path.

Examples of alternative validation paths:

- Optimizers: use stateful parity checks that compare parameters and optimizer state after one step and multiple steps.
- Distributed/process-group APIs: use integration-style tests that verify communication behavior and environment setup.
- Low-level update kernels hidden behind stateful APIs: validate the underlying update op directly when that is the real test target.

<a id="dtype-support-minimal-path"></a>
## Minimal Path

For most tasks, only do these four things:

1. Pass the applicability gate for the current op_info workflow.
2. Collect the candidate dtypes from aclnn and MindSpore docs.
3. Run the runtime probe with representative samples.
4. Use the probe result to build the candidate `dtypes_*` writeback set in `op_database.py`, then validate that candidate through the MindSpore OpInfo path before treating it as final.

For current aclnn OpInfo work, the default writeback target is only:

- `dtypes_ascend910b`
- `dtypes_backward_ascend910b` when backward needs to be encoded

Leave `dtypes_ascend`, `dtypes_cpu`, and `dtypes_gpu` empty unless the task explicitly requires those platforms.

You do **not** need to think about extra helper tooling unless the task explicitly needs it.

<a id="dtype-support-assets"></a>
## Shared Assets

Required for the default path:

- Probe framework: `scripts/dtype_probe_execution_framework.py`

Optional helper assets:

- Operator scaffold: `template/dtype_probe_operator_scaffold.template.py`

<a id="dtype-support-inputs"></a>
## Inputs

- Operator API name
- Operator category and representative invocation pattern
- Candidate dtypes from aclnn documentation and MindSpore API docs
- Whether the operator has a backward path

<a id="dtype-support-outputs"></a>
## Outputs

- A probe summary JSON and Markdown file containing, for each operator:
  - `forward_supported_dtypes`
  - `backward_supported_dtypes`
  - `backward_not_applicable_dtypes`
  - `forward_unsupported_dtypes`
  - `backward_unsupported_dtypes`
  - `forward_issue_dtypes`
  - `backward_issue_dtypes`
- Final dtype writeback recommendation for `op_database.py`
- Or a `not_applicable_to_opinfo` decision with blocker notes and a recommended alternative validation path
- If runtime probing is skipped, produce the same summary schema through the framework's doc-derived helper and mark `dtype_declaration_source=doc_derived`.

<a id="dtype-support-probe-steps"></a>
## Runtime Probe Workflow

1. Collect candidate dtypes from the aclnn and MindSpore docs.
2. Write one operator probe driver. The fastest path is to copy `template/dtype_probe_operator_scaffold.template.py` and fill:
   - representative forward path
   - representative backward path
   - representative probe samples
   - doc-declared forward/backward dtype tuples
   - probe backend selection (`pta` by default, `ms` only when explicitly requested)
   - For parameter-rich `other` ops such as `conv*`, `linear`, `interpolate`, pooling, normalization, and loss/module wrappers, do not rely on a generic placeholder sample. Build the probe samples from the nearest family pattern already present in the repo so the dtype result is based on representative arguments.
3. Run the probe:

```bash
python <skill_root>/scripts/dtype_probe_execution_framework.py \
  --driver ./mint_acos_dtype_probe.py \
  --summary-out ./op_dtype_probe_summary.json \
  --markdown-out ./op_dtype_probe_summary.md
```

4. Review the probe summary, backfill the OpInfo dtype declaration from the runtime-supported forward dtypes as the candidate writeback set, and then run the MindSpore opinfo case tests. If the later opinfo tests diverge from the PTA probe result, record and clarify that discrepancy instead of silently shrinking the dtype set.

If you need to probe multiple operators in one run, pass multiple `--driver` arguments to the same command. Keep that as an execution detail, not as a separate workflow branch.

<a id="dtype-support-classification"></a>
## Runtime Result Classification

The probe summary must classify each dtype and direction as follows:

- **Forward success**: record as `forward_supported_dtypes`
- **Backward success**: record as `backward_supported_dtypes`
- **Backward not applicable for this dtype/sample path**: record as `backward_not_applicable_dtypes`
- **Error clearly indicates unsupported dtype**: record as `unsupported_dtype`
- **Any other error**: record as `sample_or_function_issue`

Do **not** convert a generic failure into "dtype unsupported" unless the error really says so.
Do **not** fold "no gradient path" into `backward_supported_dtypes`; keep it separate as `backward_not_applicable_dtypes`.

<a id="dtype-support-issues"></a>
## Handling Probe Failures

When a probe case fails:

1. If the failure is caused by an obvious sample problem, fix the sample and rerun.
2. If the failure is caused by the generic driver logic and can be corrected safely, fix the probe driver and rerun.
3. Cross-check the aclnn documentation, the MindSpore API documentation, and the PTA benchmark documentation before deciding that the probe result is authoritative.
4. If the failure looks like an operator implementation problem, keep the dtype in `*_issue_dtypes` and record it as a functional blocker.

Do not hide operator bugs by silently removing the dtype from the candidate set.

<a id="dtype-support-docs-only"></a>
## Documentation Lookup Workflow

When the task or environment calls for the doc-based path:

1. Check the aclnn interface declaration for the operator's supported dtype list.
2. Check the corresponding MindSpore API documentation and existing similar OpInfo registrations.
3. Produce a candidate dtype set and explicitly mark it as **doc-derived**. When you still want a structured summary artifact, use `build_doc_derived_operator_summary(...)` from `scripts/dtype_probe_execution_framework.py` so the doc-only path emits the same schema as the runtime probe path.
4. If the task later allows runtime execution, rerun the probe and replace the doc-derived result with the runtime result.

Documentation lookup is a valid supplemental path, but it is **not** the default source of truth.

Do not use the doc-based path to justify forcing a clearly non-applicable API into the op_info workflow.

<a id="dtype-support-scope-rules"></a>
## Scope Rules For OpInfo Test Authoring

These rules apply to what you write into the final OpInfo tests:

- **Static forward dtype coverage** should follow the supported forward dtype set.
- **Backward dtype discovery** may use the full probe, but the final OpInfo workflow does not need extra custom backward-only dtype matrices beyond what the framework already provides.
- **Dynamic shape/rank** does not require a full dtype matrix in the default workflow. Representative dtype coverage is enough unless the task explicitly asks for more.
- **Negative cases** do not require a full unsupported-dtype matrix. Representative unsupported-dtype cases are enough.

<a id="dtype-support-writeback"></a>
## Writeback Rules

After the runtime probe:

1. Use the supported forward dtype set from the default runtime probe as the default candidate source for `dtypes_*`.
2. For current aclnn OpInfo tasks, write back only `dtypes_ascend910b` by default. Leave `dtypes_ascend`, `dtypes_cpu`, and `dtypes_gpu` empty unless the task explicitly requires those platforms.
3. If the backward probe shows a stable narrower backward dtype set and the task requires encoding it, write `dtypes_backward_ascend910b` by default. Leave other backward platform fields empty unless explicitly required.
4. Keep runtime issue dtypes in the summary so the gap is visible even when not written into `op_database.py`.
5. Do not treat `not_support_dtypes` as the automatic complement of the writeback set. Negative dtypes must be justified by representative MindSpore-side failure evidence.
6. Treat the probe result as final writeback evidence only after the required OpInfo validation matrix passes. If it does not, move the operator into a clarification or blocker path rather than silently editing the dtype set to make the failure disappear.
7. Record whether the final dtype declaration came from:
   - `probe_verified`
   - `doc_derived`
   - `probe_vs_doc_conflict`
