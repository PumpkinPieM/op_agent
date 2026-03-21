---
name: aclnn_builder
description: Guides end-to-end ACLNN custom operator development and adaptation in MindSpore (PyBoost/Pynative + KBK/Graph paths), including YAML definitions, code generation, GeneralInfer, kernel registration, bprop wiring, tests (UT), and docs. Use when the user mentions ACLNN, Ascend, 算子适配/算子开发, PyBoost, KBK, op_def YAML, gen_ops.py, bprop, or Ascend operator adaptation tasks.
---

# ACLNN 算子开发全流程（MindSpore 适配）

## 目标
把一个 Ascend 平台 ACLNN 算子在 MindSpore 里**端到端落地**：前向/反向、PyBoost(Pynative) 与 KBK(Graph) 双路径、动态 shape/rank、UT、文档与导出，并完成必要的质量检查与验证。

## 使用方式（你要怎么用这份 skill）
- 当用户说"给MindSpore接入/适配一个 ACLNN 算子""给MS添加一个xxx接口，实现对标torch_npu""请通过skill帮我补一个npu算子""新增 xxx_op.yaml""PyBoost/KBK 怎么写""bprop 怎么注册""UT 怎么补"等，直接按本 skill 的步骤推进。

> **⚠️ 必读规则：执行每个 Step 前，必须读取对应的 workflow 文件**
> （`workflows/XX-xxx.md`）获取详细步骤、约束和成功标准。
> `reference.md`, `aclnn_doc`等共享文档保存在`../_shared/`路径下。

## 执行流程

### Workflow 执行清单

使用此 skill 开发 ACLNN 算子时，**创建 TODOLIST**，按顺序执行以下 workflow。
**注意：标记 `🔒不可跳过` 的步骤无论什么场景都必须执行，不能通过条件跳步裁剪。**
**注意：标记 `⛔ HARD GATE` 的地方必须完成前置产出后才能继续，否则停下等用户确认。**

- [ ] **[Pre](workflows/00-pre-checks.md)**`🔒不可跳过`：前置检查（Pre-A 存量检查 + Pre-B 方案设计 + Pre-C 调用链盘点）
  - 输入：算子名称、PTA 对标接口
  - **必须产出**：PTA 源码审查报告，已初始化 Feature 文档
    > **⛔ HARD GATE 1**：进入 Step 1 之前，必须确认该**文件已生成到工作区**：
  - 组合场景还需产出：`templates/aclnn-callchain-inventory.md` 调用链盘点表
  - 后续每完成一个 Step，回填 Feature 文档对应章节
- [ ] **[Step 1](workflows/01-yaml-definition.md)**：YAML 定义 → 回填 Feature [5. YAML 定义](templates/feature-document.md#feature-yaml-definition)
  - 输入：PTA 源码审查报告, Feature文档
  - 输出：op_def + api_def + doc YAML 文件
- [ ] **[Step 2](workflows/02-code-generation.md)**：代码生成
  - 输入：YAML 文件
  - 输出：gen_ops.py 运行成功
- [ ] **[Step 3](workflows/03-general-infer.md)**：GeneralInfer  → 回填 Feature [9. 动态 Shape/Rank 支持](templates/feature-document.md#feature-dynamic-shape) / [10. 异常与校验](templates/feature-document.md#feature-validation-and-errors)
  - 输入：YAML、PTA 输出 shape 逻辑
  - 输出：Infer 实现文件
- [ ] **[Step 4](workflows/04-pyboost.md)**：PyBoost（Pynative） → 回填 Feature [7. 执行模式与适配](templates/feature-document.md#feature-execution-modes)
  - **路径 1**：跳过手写，验证自动生成产物正确即可
  - **路径 2**：手写 customize 实现文件（参数转换 + ACLNN 调用）
  - 输入：YAML、ACLNN 调用细节
  - 输出：customize 实现文件（路径 2）/ 验证通过（路径 1）
- [ ] **[Step 5](workflows/05-kbk.md)**：KBK（Graph） → 回填 Feature [7. 执行模式与适配](templates/feature-document.md#feature-execution-modes)
  - **路径 1**：跳过手写，验证自动注册正确即可
  - **路径 2**：手写 kernel 文件（GetWorkSpaceInfo + Launch + 注册）
  - 输入：YAML、ACLNN 调用细节
  - 输出：kernel 实现文件
- [ ] **[Step 6](workflows/06-bprop.md)**：BPROP 注册 → 回填 Feature [11. 反向（BPROP）](templates/feature-document.md#feature-bprop)
  - 输入：derivatives.yaml 分析、反向 kernel
  - 输出：bprop 实现
- [ ] **[Step 7](workflows/07-export.md)**：导出与占位
  - 输入：算子实现
  - 输出：ops 包导出、接口文件、非 Ascend 占位；如涉及接口重载见 [`reference.md` 25 接口重载适配](reference.md#api-overload-adaptation)
- [ ] **[Step 8](workflows/08-testing.md)**：测试 → 回填 Feature [12. 测试方案](templates/feature-document.md#feature-test-plan)
  - 输入：全部实现、PTA 对标
  - 输出：C++ UT（必须新建）
    详见 `workflows/08-testing.md` Step 2。
- [ ] **[Step 9](workflows/09-docs.md)**：文档
  - 输入：接口实现
  - 输出：英文 function_doc（Step 1 已创建，此处完善）+ **中文 RST（公开 API 必须）**
  - **⚠️ 英文 doc YAML 不等于文档步骤完成——中文 RST 是独立产物，最容易遗漏**
  - **⚠️ mint/ops/nn/Tensor 公开接口不得跳过此步骤**（仅内部算子可跳过，见条件跳步表）
- [ ] **Feature 文档定稿** `🔒不可跳过`：补齐 [13. 代码与文件改动说明](templates/feature-document.md#feature-code-change-summary)、[14. 验收报告](templates/feature-document.md#feature-acceptance-report)、更新 [3. 任务清单](templates/feature-document.md#feature-task-list)
  - 即使 Step 9/10 被跳过或推迟，Feature 文档也必须在代码开发完成后补齐并输出给用户
- [ ] **[Step 10]**：补完Feature文档


## 验证闭环（每一步都要给证据）`🔒不可跳过`

每完成一个 Step，**必须**使用以下模板向用户展示执行报告（不可省略、不可合并、不可延后）。
**这是对用户的强制交付物，不是内部记录——必须在消息中直接输出给用户看到。**

```text
━━━ Step X 执行报告 ━━━

执行依据（我依据 skill 的哪条要求来执行）：
- workflow 文件：workflows/XX-xxx.md
- 对应的 skill 要求：（引用 SKILL.md / workflow 中的具体条目）
- 本步骤的成功标准：（从 workflow 成功标准中摘录）

我做了什么（产出清单）：
- ...

关键证据（代码片段/文件路径/搜索结果）：
- ...
- 对照了哪个已有算子的实际代码：...

验证结果：
- ...

成功标准逐项核对：
- [ ] 标准1：✅/❌
- [ ] 标准2：✅/❌
- ...

遗留问题/风险与下一步：
- ...
```

## 关键约束（必须遵守）

**以仓库实际代码为准，不要盲从文档流程。**
本 skill 的流程、模板、命名约定都可能因 MindSpore 版本迭代而过时。
发现文档描述与仓库现状不一致时，**以仓库现状为准**。


## 额外资料（按需读取）

- **知识参考与代码骨架**：`../_shared/reference.md`
- **触发样例**：`examples.md`
- **PTA 探测脚本模板**：`scripts/probe_pta_sparse_flash_attention.py`