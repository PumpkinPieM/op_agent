---
name: op_info_test
description: Generate and validate MindSpore Python ST op_info tests end-to-end (case generation, temporary isolation patch, remote deploy/test, rerun, and final cleanup). Use when users ask to add/repair/verify op_info-based ST for one or more operators, run op_info smoketest, or produce remote test evidence and coverage summary for op_info cases.
---

生成 op_info 测试，优先直接落地，不在中途停下提问；仅在权限受限、关键信息缺失且无法合理推断时提问。

## 快速路由

- 指令包含 `smoketest` / `冒烟`：直接执行 [workflow/smoke.md](workflow/smoke.md)。
- 其他 op_info 用例新增/修复请求：执行下述完整闭环。

## 完整闭环流程

1. 从 `master` 新建分支：`add_opinfo_test_[op_name]`。
2. 按 [workflow/op_info_generation.md](workflow/op_info_generation.md) 生成/更新目标接口用例。
3. 在当前工作目录生成 `分支名.md` 总结文档（不加入 git）：
   - 新增/修改的接口与用例清单。
   - 覆盖场景与边界。
   - 未覆盖场景及原因。

## 执行约束

- 不改动与目标接口无关的测试内容。
- 检查生成用例是否覆盖[workflow/op_info_generation.md]中所要求的全部场景。如果有缺失场景，除非是信息缺失等无法解决原因，否则重新执行用例生成步骤直到场景全部放覆盖。
- 输出总结文档时明确写出“已覆盖/未覆盖/阻塞原因”，避免模糊表述。
- 仅生成测试，不要跑测试。
