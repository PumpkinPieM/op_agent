---
name: aclnn_full_process
description: invoke this skill only when it's mentioned directly.
---

Execute aclnn adaptation-task end-to-end. Finish the whole process, don't ask unless there's vital problem or key information missing.

Do not skip any step.

1. Create branch `aclnn_{op_name}_agent_task`
2. Run `python ./mindspore/python/mindspore/ops_generate/gen_ops.py` to refresh the auto-generated files.
3. Invoke skill[`aclnn-builder`] for aclnn adaptation code generation
4. commit and push，message `aclnn task: {op_name}`
5. Invoke skill[`op-info-test`] for ST test generation and validation. The remote server IP is `8.92.7.131`, port `18081`.
