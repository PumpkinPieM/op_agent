# Workflow 0: Pre-check

## Goal

Before coding, retrieve the current MindSpore API / op status from `operator-facts` (especially ACLNN integration coverage), retrieve the corresponding PTA API facts, and directly conclude how the MindSpore operator should integrate ACLNN and which MindSpore files must be modified or added.

## Inputs

- MindSpore target API: `mindspore.xxx`, `mindspore.Tensor.xxx`, or `mindspore.mint.xxx`
- Or MindSpore target op / primitive: `{op}` or `{primitive}`
- Local indexes under `operator-facts/`
  - `bundles/`: MS-centered single-operator bundles
  - `data/pta_facts.jsonl`: PTA facts index

## Output

- Selected PTA reference
- ACLNN integration decision for the MindSpore operator
- MindSpore file list to modify/add/verify
- PTA reference anchors to open while coding

## Step 1. Retrieval target MindSpore API/OP and corresponding PTA API/OP

```text
operator-facts/
├── bundles/                        # Mindspore single-operator bundles
│   └── mindspore.Tensor.abs/       # one public_api directory
│       └── abs_op.yaml.json        # one MS op branch bundle, **primary key: public_api::op_branch**
│   └── mindspore.Tensor.add/       # one public_api directory, multiple branches possible
│       ├── add_ext_op.yaml.json    # one MS op branch bundle
│       └── add_scalar_op.yaml.json # another MS op branch bundle
├── data/
│   ├── api_identity.jsonl          # MS API -> op_branch / primitive index, **primary key: public_api::op_branch**
│   ├── ms_coverage.jsonl           # MS ACLNN / infer / kbk / bprop coverage index, **primary key: op_branch::primitive**
│   └── pta_facts.jsonl             # PTA API facts + source refs
└── schemas/
    ├── op_bundle.schema.json       # bundle format, for bundles
    ├── pta_facts.schema.json       # PTA facts format, for data/pta_facts.jsonl
    ├── api_identity.schema.json    # API identity format, for data/ms_coverage.jsonl
    └── ms_coverage.schema.json     # MS coverage format, for data/ms_coverage.jsonl
```

1. If the user starts from a MindSpore op / primitive instead of a public API, first resolve it through `operator-facts/data/ms_coverage.jsonl`, Then normalize the target into one or more MindSpore op branches and continue with coverage-based gap checking.

2. If the user starts from a MindSpore public API, **for each op branch of the target MindSpore API**, lookup the MS bundle by `public_api` and read the operator bundle. `identity` tells us which MS API / op branch / primitive we are handling; `coverage` tells us the current MS implementation status and current gaps, especially ACLNN-related coverage.

For example: the following bundle tell us the abs op have implement aclnn in auto_generate ptah, both kbk and pyboost support aclnn backend. Infer code / bprop code / ut / st / docs exits (just mark exits, not necessarily related to ACLNN).

```json
{
  "bundle_key": "mindspore.Tensor.abs::abs_op.yaml",
  "identity": {
    "public_api": "mindspore.Tensor.abs",
    "public_surface": "mindspore.Tensor",
    "api_name": "abs",
    "op_branch": "abs_op.yaml",
    "op": "abs",
    "primitive": "Abs",
    "interface": "tensor",
    "py_method": "tensor_abs",
    ...
  },
  "coverage": {
    "coverage_key": "abs::Abs",
    "op_yaml_path": "mindspore/ops/op_def/yaml/abs_op.yaml",
    "class_name": "Abs",
    "dispatch_enable": true,
    "dispatch_kind": "auto_generate",
    "dispatch_ascend": "default",
    "aclnn": ["aclnnAbs"],
    "aclnn_source": ["auto_generate_source"],
    "infer": true,
    "pyboost": true,
    "kbk": true,
    "bprop": true,
    "ut": true,
    "st": true,
    "docs_cn": false,
    "docs_en": true
  }
}
```


## Step 2. Decision and Implementation Plan

For each op branch, first decide whether there is any implementation gap. If there is no ACLNN-related gap and no backward / test gap for the current target, stop and report that no implementation is needed for this branch.

The current Pre conclusion is limited to what existing `operator-facts` can already tell us. PTA is only a reference for implementation; it does not define the MindSpore requirement itself. Do not reopen broad source analysis here.

1. Select PTA reference
   - Retrieve PTA candidate facts from `operator-facts/data/pta_facts.jsonl`
   - Use bundle `identity.api_name`, `identity.op`, and `identity.primitive` as the main retrieval hints
   - Use bundle `coverage.aclnn` only as an auxiliary hint when it already exists
   - Select the PTA candidate that is most semantically aligned with the current MS branch; prefer non-composite candidates when multiple candidates are close

2. Decide primitive / interface strategy
   - Reuse the current branch only when the current MS branch already matches the target semantics without changing existing behavior
   - Otherwise, default to a new `_ext`-style branch or a new primitive
   - Do not force reuse just because names are similar

3. Decide dispatch / integration path
   - If the selected PTA reference is composite, treat it as handwritten logic, not a simple single-ACLNN passthrough
   - If `preprocess_needed=true` or `custom_output_needed=true`, conclude that handwritten customize integration is required
   - Even if PTA itself looks simple, if the current MS interface cannot pass inputs through to ACLNN unchanged, conclude that handwritten customize integration is still required
   - Only conclude `auto` when the PTA path is non-composite, has no extra preprocessing / custom output logic, and the current MS interface can pass arguments through unchanged to ACLNN

4. Decide backward strategy
   - If the selected PTA reference has no backward, conclude that no backward implementation is needed here
   - If PTA has backward and bundle `coverage.bprop=true`, conclude that the current bprop can be reused or verified
   - If PTA has backward and bundle `coverage.bprop=false`, conclude that bprop must be added or updated

5. Build the MindSpore file list
   - Always list `yaml`
   - Always list `infer`
   - List `kbk`
   - List `pyboost`
   - List `bprop`
   - List `ut/st`
   - For each category, state whether the action is `no change`, `modify_or_verify`, or `add_or_modify`
   - `auto` normally means no handwritten `kbk/pyboost`
   - `customize` means handwritten `kbk/pyboost` must be included

## Output Format

```text
Branch: {op_branch}
Selected PTA: {pta_key or none}

Conclusion:
- Gap check: {no implementation needed / implementation required}
- Primitive / interface strategy: {reuse current branch / new `_ext`-style branch / new primitive}, because {brief reason}
- Integration path: {auto / customize}, because {brief reason}
- Backward: {no backward / reuse or verify current bprop / add or update bprop}, because {brief reason}

MindSpore modify list:
- yaml: {path or target} | {no change / modify_or_verify / add_or_modify}
- infer: {path or target} | {no change / modify_or_verify / add_or_modify}
- kbk: {path or target} | {no change / modify_or_verify / add_or_modify}
- pyboost: {path or target} | {no change / modify_or_verify / add_or_modify}
- bprop: {path or target} | {no change / modify_or_verify / add_or_modify}
- ut/st: {path or target} | {no change / modify_or_verify / add_or_modify}

PTA refs:
- {role} | {path} | {pattern}
- ...
```

## Constraints

- Do not search MindSpore or op-plugin source trees broadly in Pre
- Pre must consume `operator-facts/bundles/` and `operator-facts/data/pta_facts.jsonl`
- Source files may be opened only from selected PTA `refs` or explicit MindSpore target files
