---
name: aclnn_full_process
description: Guides end-to-end ACLNN custom operator development and adaptation in MindSpore (PyBoost/Pynative + KBK/Graph paths), including YAML definitions, code generation, GeneralInfer, kernel registration, bprop wiring, tests (UT/ST), and docs. 
---

执行aclnn接入端到端闭环，优先直接落地，不在中途停下提问；仅在权限受限、关键信息缺失且无法合理推断时提问。

不允许跳过任何步骤。

1. 调用skill[`mindspore-aclnn-opeartor-devflow`]，执行接口接入任务。
2. commit修改，message `aclnn task: {op_name}`
3. 调用skill[`op_info_test`]，执行测试用例生成和测试任务。

执行完毕后检查，确保无遗漏任务。
