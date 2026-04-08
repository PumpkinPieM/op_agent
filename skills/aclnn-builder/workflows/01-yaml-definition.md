# Workflow 1: YAML Definition

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Define the required YAML for the operator (`op_def` + `api_def` + `function_doc`).

## Inputs

- **Feature document**: integration type, parameter list, input/output definitions
- **PTA source review results**: parameter names, types, defaults, and return structure

## Outputs

- **YAML file**: `mindspore/ops/op_def/yaml/{op_name}_op.yaml`
- **Documentation YAML**: `mindspore/ops/op_def/yaml/doc`

---

## Steps

Create yaml for new primitive or modify yaml of existing primitive for aclnn dispatch. Refer to skill `yaml-helper` for yaml-related knowledge.

**This is where the path decision lands in YAML**

**Path 1 (auto-generate)** - direct argument passthrough, no Customize needed:
```yaml
dispatch:
  enable: True
  # omit the Ascend field -> build will auto-generate PyBoost/KBK code
```

**Path 2 (Customize)**:
```yaml
dispatch:
  enable: True
  Ascend: OpNameAscend    # specify the customize kernel function name
```

### API Yaml

For tensor method and overloaded interface tasks, add/modify corresponding api yaml in `mindspore/ops/api_def`.

For tensor method task, after registering the tensor interface in api yaml, delete the old tensor method registration in the Tensor class in `mindspore/python/mindspore/common/tensor.py`.

---

## Success Criteria

- [ ] OP YAML files have been created/modified for related ops
- [ ] API YAML files have been created/modified for related tensor method/overloaded interface
---
