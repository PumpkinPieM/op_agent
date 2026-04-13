---
name: aclnn_full_process
description: Use this skill only when the user mention it directly. Auto-invocation is disabled.
---

Execute aclnn adaptation-task end-to-end. Finish the whole process, don't ask unless there's vital problem or key information missing.

Do not skip any step.

1. Create branch `aclnn_{op_name}_agent_task` from `HEAD`.
2. Run `python ./mindspore/python/mindspore/ops_generate/gen_ops.py` to refresh the auto-generated files.
3. Invoke skill[`aclnn-builder`] for aclnn adaptation code generation
4. use "bash build.sh -e ascend -j{n}" to build MindSpore, and fix compile error. `n` is the worker number, suggest value is 64.
5. commit with message `aclnn task: {op_name}`
6. Invoke skill[`op-info-test`] for ST test generation and validation.
