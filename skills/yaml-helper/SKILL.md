---
name: yaml-helper
description: "Reference for MindSpore operator-definition YAML files and field meanings, contains yaml and field explanation, use cases and examples. Use when op/api/doc yaml files needs to be modified or created. Use when reliable context are needed for MindSpore op/api/doc yaml, the meaning of YAML keys, or what code is auto-generated from these YAML definitions. Keywords: yaml, op definition, auto generation, primitive class, function, dispatch, aclnn."
---

# YAML Helper

## Overview

Use this skill as a compact source of truth for MindSpore op-definition YAML tasks.

## What This Skill Provides

- The primary documentation file for op YAML structure.
- A short list of representative YAML examples for common patterns.
- The main generated-code output locations tied to these YAML definitions.

## Source Priority

Use these sources in order of authority:

1. `mindspore/ops/op_def/yaml/README.md`
   Use for the documented YAML schema and generated output paths.
2. The specific YAML file under `mindspore/ops/op_def/yaml/` that the task is about.

## Op Yaml (`mindspore/ops/op_def/`)

op yaml defines the operator(primitive) metadata and controls the auto generation of operator definition code in python and c++.

> keywords: primitive, op definition, auto generation, code generation, dispatch, aclnn.

### Arg Definition

Each arg in `args` field defines an input of the operator, including the input name, dtype(tensor, int, list, etc.), default value.

`type_cast` field specifies the allowed implicit conversion for the input, including scalar(int, float, bool) <-> tensor, list/tuple <-> scalar/tensor, list <-> tuple. `arg_handler` specifies a customized conversion function for the input. Both types of conversion happens before inputs are fed to the primitive. Other than this, there is a default scalar to tensor conversion for all scalar inputs, which can be disabled by setting `disable_tensor_to_scalar` to `True`.

### Primitive Class and Function

A primitive class `class Foo(Primitive)` in `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` and a function `def foo(...)` in `mindspore/python/mindspore/ops/auto_generate/gen_ops.py` will be auto generated for a yaml file. They can be renamed or disabled by setting the `name` or `disable` field under `class` or `function` in yaml file.

The primitive class ties to the op's c++ operator. All c++ implementation including infer, kernel, etc. will be generated based on the primitive class. The function is the user-facing API that calls the primitive class.

### Dispatch

The `dispatch` field controls kernel-related code generation. It can have an `enable` key to turn on auto-generation, and device-specific keys (like `CPU`, `GPU`, `Ascend`) that specify customized function names to use instead of auto-generated ones for those targets. By default, `enable` is False, meaning no kernel code will be generated.

### Inplace Op

For inplace op, add fields `rw_write`, `side_effect_mem` and `inplace` for the output correspondingly.

Example `inplace_add_ext`, whose output tensor is actually the `input` tensor.

```yaml
inplace_add_ext:
  args:
    input:
      dtype: tensor
    other:
      dtype: tensor
    alpha:
      dtype: number
      default: 1
  args_signature:
    rw_write: input
  returns:
    output:
      dtype: tensor
      inplace: input
  labels:
    side_effect_mem: True
  class:
    name: InplaceAddExt
  dispatch:
    enable: True
    Ascend: InplaceAddExtAscend
    GPU: None
```

#### Ascend

Ascend backend dispatches kernel calculation to aclnn operator interface. Pyboost kernel(`mindspore/ops/kernel/ascend/aclnn/pyboost_impl`) for the pynative mode and aclnn kernelmod(`mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl`) for the graph mode(KBK mode) are generated for the op, in which aclnn interfaces are invoked.

When no customized function is specified, the auto-generated kernel will use the op name with prefix `aclnn` (e.g. `aclnnFoo`) as the aclnn interface, and pass all inputs that defined in the yaml file as-is to aclnn interface. Otherwise, the auto-generated kernel will instead pass inputs to the customized function, which is then responsible for calling aclnn interface and doing necessary input transformation.

## API Yaml (`mindspore/ops/api_def/`)

API yaml is used to provide a function and/or a tensor method interface for an operator, based on the primitive defined in op yaml.

An api yaml file can contain multiple overloads for the same operator to support api overloading, with each overload defined by the signature of its `op_yaml`.

> keywords: function interface, tensor method interface, api overload

Example:

```yaml
add:
  - op_yaml: add_scalar_op.yaml
    py_method: tensor_add_ext
    kwonlyargs: alpha
    Ascend: pyboost
    CPU: py_method
    GPU: py_method
    disable_scalar_tensor: other
    interface: tensor, function

  - op_yaml: add_ext_op.yaml
    py_method: tensor_add_ext
    kwonlyargs: alpha
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function

  - op_yaml: deprecated/add_method.yaml
    py_method: deprecated_tensor_add
    Ascend: py_method
    CPU: py_method
    GPU: py_method
    interface: tensor
```

The generated `add` interface checks inputs against the signatures defined in each `xxx_op.yaml`. When it matches, it invokes the corresponding primitive. The checking is done in order.

### device field

The device field('Ascend', 'CPU', 'GPU') can have value `pyboost` or `py_method`. For `pyboost`, the execution is dispatched to the primitive defined in `op_yaml`. When the value is `py_method`, the execution is dispatched to the customized python method defined in `py_method` field. If device field is not set, the execution is dispatched to the primitive by default.

#### deprecated entry

For **non-deprecated** op_yaml, the device field only take affect in the pynative mode. In graph mode(KBK mode),  the device field is ignored and the primitive is the only execution path. For deprecated op_yaml, this field takes affect in both mode, and the customized python method is the only allowed execution path. 

Moreover, the deprecated entry takes presedence over non-deprecated entries for signature checking in graph mode(KBK mode).

### `interface` field

The `interface` field decides what kind of interface will be generated for the api. It can be set as `tensor`, `function` or both. When `tensor` is included, a tensor method interface will be generated, which means the api can be called by `x.add(y)`. When `function` is included, a functional interface will be generated in `mindspore/python/mindspore/ops/functional_overload.py`.

### `kwonlyargs` field

The `kwonlyargs` field turns the specified arguments in `xxx_op.yaml` into keyword-only arguments in the generated interface.

### `disable_scalar_tensor` field

When the `disable_scalar_tensor` field is set in the op yaml file, it has also to be added in the corresponding section in API yaml file, to inform the dispatcher that the default scalar-tensor conversion is disabled for the specified input.

### `alias` field

When an api is an alias of another api(e.g. "__mul__" is an alias of "mul"), just set the `alias` field.

`__mul__`'s case:

```yaml
__mul__:
  alias: mul
```

## Doc YAML

### Op Doc

Every op yaml should register a corredsponding doc yaml in `mindspore/ops/op_def/doc/` with the same file name. The `description` field stores the op docstring in English. The docstring should contain a brief description of the operator, plus sections `Inputs`, `Outputs`,  `Supported Platforms` and `Examples`. If the op is not user facing(internal op or backprop-only op), the docstring can be set as empty string.

### API Doc

Each api yaml has a doc yaml in `mindspore/ops/api_def/function_doc` and/or `mindspore/ops/api_def/method_doc`, according to the interface type. 

When the api has overloaded signatures, all overload type should be documented in the same doc yaml file. Check `mindspore/ops/api_def/method_doc/add_doc.yaml` for example. When the api is an alias of another api, just mention it's an alias to another api, i.e.

```yaml
__mul__:
  description: |
    __mul__(other) -> Tensor

    Alias for :func:`mindspore.Tensor.mul`.
```