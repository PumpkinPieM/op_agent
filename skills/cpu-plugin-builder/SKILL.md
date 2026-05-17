---
name: cpu-plugin-builder
description: Build MindSpore CPU operators by adapting ATen (libtorch) operators via mindspore_op_plugin. Use when implementing ops in op_plugin/ops/kernel/, writing kernel .cc files
---

# CPU Plugin Builder

This skill helps you develop CPU operators for MindSpore's op_plugin that call ATen (libtorch) operators.

## When to Use

Use this skill when:
- Implementing CPU operators for mindspore_op_plugin
- Writing forward and backward (gradient) operators kernel `.cc` files under `op_plugin/ops/kernel/`

## Instructions

### Step 1: Load api-helper skill to find op name.

### Step 2: Find corresponding torch ATen Interface
must read ./reference/how_to_find_aten_interface.md

### Step 3: Write the Forward Operator
must read ./reference/how_to_write_forward_op.md

### Step 4: Write the Backward Operator
must read ./reference/how_to_write_backward_op.md

### Step 5: Build and run test

cd `mindspore_op_plugin`
build with `bash build.sh`
get env ready : `source env.source`
run test : `pytest tests/st/mint/test_{op_name}.py`

### Step 6: Write Report of Each Step
report contains: forward opname(list out kernel file name), backward op name(list out kernel file name), test result

