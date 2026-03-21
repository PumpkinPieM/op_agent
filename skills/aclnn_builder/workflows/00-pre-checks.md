# Workflow 0: 前置检查（Pre-A / Pre-B / Pre-C）

## 目标

在写代码前，完成存量检查、对标分析、方案设计与组合场景调用链盘点。

## 输入

- **算子名称**：算子接口名（API、Primitive、ACLNN接口）
- **PTA 对标接口**：`torch_npu.npu_xxx` 或 `torch.xxx`

## 输出

- **存量检查结果**：该算子在 MS 仓库中已有 / 缺失的部分
- **方案设计文档**：接口类型、对接分类、影响面评估, 以md文件形式输出
- **ACLNN 调用链盘点表**（组合场景）：子算子覆盖状态与实施计划

## 约束

- **本地代码优先**：PTA（op-plugin）、PyTorch、MindSpore 的源码查阅**必须在本地工作区目录中搜索**。

---

## Pre-A：存量检查

当用户让你"新增 / 适配某算子"时，**先搜索确认**该算子在仓库中是否已存在。

### 执行步骤

1. **搜索 YAML**：在 `mindspore/ops/op_def/yaml/`, `mindspore/ops/api_def/yaml/` 中搜索算子名
2. **搜索 Infer**：在 `ops_func_impl` / `ops/infer` 目录搜索对应 infer 注册
3. **搜索 PyBoost**：搜索 `class OPS_ASCEND_API {OpName}`
4. **搜索 KBK**：搜索 `MS_ACLNN_KERNEL_FACTORY_REG({OpName}, ...)`, `MS_ACLNN_COMMON_KERNEL_FACTORY_REG({OpName}, ...)`
5. **搜索 BPROP**：搜索 `REG_BPROP_BUILDER("{OpName}")`
6. **搜索测试**：在 `tests/st/ops` 和`test/ut/cpp/ops` 下搜索算子名
7. **搜索文档**：在 `docs/api/` 下搜索算子名

### 输出模板

```text
算子存量检查：{OpName}

| 组件 | 状态 | 文件路径 | 备注 |
| ---- | ---- | -------- | ---- |
| YAML (op_def) | ✅/❌ | ... | |
| YAML (api_def) | ✅/❌ | ... | |
| Infer | ✅/❌ | ... | |
| PyBoost | ✅/❌ | ... | |
| KBK kernel | ✅/❌ | ... | |
| BPROP | ✅/❌ | ... | |
| 测试 (UT) | ✅/❌ | ... | |
| 测试 (ST) | ✅/❌ | ... | |
| 文档 (EN) | ✅/❌ | ... | |
| 文档 (CN) | ✅/❌ | ... | |

结论：{全新开发 / 需补齐xxx部分}
```

---

## Pre-B：方案设计与对标分析

分析 MS/PTA/ACLNN 的接口差异，决定原语/接口接入策略，**确定接入路径（路径 1 自动生成 / 路径 2 Customize）**，并初始化 Feature 文档。

### 执行步骤

1. **PTA 源码审查（必做）**：审查 op-plugin 三类关键文件（详见 [`reference.md` 19 PTA 源码审查方法](reference.md#pta-source-review)）
   - `op_plugin_functions.yaml`：函数签名、参数类型/默认值
   - `derivatives.yaml`：反向注册、可微输入
   - `XxxKernelNpuOpApi.cpp`：实际 ACLNN 调用、参数预处理
   - 注意 PTA 是否有**同名接口重载**（同函数名、不同参数签名）
   - **aclnn接口定义**: `aclnn_doc`中查找相关接口aclnn文档（如`aclnnAbs.md`）
2. **接口分析五要素（必做）**（[`reference.md` 15.4.1 接口分析五要素](reference.md#api-analysis-five-factors)）：
   - 功能 / 参数定义 / 数据类型是否一致
   - **是否要新增原语**；**是新增接口还是复用原有接口**
3. **确定 YAML 策略**（[`reference.md` 15.4.2 YAML 三种场景](reference.md#yaml-three-scenarios)）：
   - yaml接口定义见`mindspore/ops/op_def/yaml/README.md`
   - 已有 YAML + 复用原有原语 → 加 `dispatch` 字段
   - 已有 YAML + 新增原语 → 新建 YAML 加 `_ext` 后缀
   - 没有 YAML → 新建
   - 若需修改已有原语参数签名 → 参考 MS 仓库相似算子处理方式，具体分析兼容性（[`reference.md` 15.4.4 修改已有原语参数签名与接口重载](reference.md#existing-primitive-signature-change)）
   > 明确yaml与原语，函数接口关系:
   > yaml与原语强绑定，定义的接口即生成原语接口。yaml流程同时也可自动生成函数接口，在接口中直接调用生成的原语的实例，简化实现。可通过yaml定义中的"function"相关field控制。原语与函数接口非强相关。
   > 在后端接口与原语接口一致情况下，优先使用不带"Ext"后缀原语名。仅在当前已有同名原语且不能复用情况下，使用"Ext"后缀。在已有同名函数接口但无同名原语情况，仍优先使用不带"Ext"后缀原语名，区分原语和函数接口。
4. **确定接入路径（核心决策）**（[`reference.md` 2.3 两条接入路径](reference.md#dispatch-path-selection)）：
   - 分析 MindSpore API 参数能否**原样透传**给 ACLNN 接口
   - **路径 1（自动生成）**：参数直通 → YAML 不写 `Ascend` 字段 → pyboost和aclnn kernelmod自动生成
   - **路径 2（Customize）**：参数需预处理 → YAML 写 `Ascend: XxxAscend` → pyboost和aclnn kernelmod必须手写
   - 常见需预处理的情况：标量提取、参数重排、输出手动分配
   - **此决策直接决定后续整个开发工作量，必须在 Pre-B 阶段明确**
   - yaml提供type_cast接口支持简单输入类型转换, 转换后如果参数与aclnn接口匹配仍可以走自动生成路径。
5. **产出 PTA 差异记录**: 使用 `templates/pta-analysis-report.md` 模板, 生成文件（如 `{op_name}_pta_analysis.md`）

---

## 🔒 Feature 文档初始化（Pre-B 完成后必须执行，不可跳过）

> **这是评审和转测交付的必须产物。** 无论什么场景（前向/反向、单算子/组合、内部/公开），
> 都必须生成 Feature 文档。如果跳过此步，后续将无法通过评审。

### 执行步骤

1. 从 `templates/feature-document.md` 复制一份，命名为 `{算子名}_Feature.md`
2. 基于 Pre-B 的分析结果，填写以下章节：
   - [1. 背景描述](../templates/feature-document.md#feature-background)
   - [2. 标杆与接口](../templates/feature-document.md#feature-benchmark-api)
   - [3. 任务清单](../templates/feature-document.md#feature-task-list)（标准 13 大类表格，初始化每项状态）
   - [4. 功能与接口说明](../templates/feature-document.md#feature-functional-spec)（接口签名、参数说明）
   - [6. 约束与类型](../templates/feature-document.md#feature-constraints)（设备、dtype、shape 约束）
   - [8. 与 PTA 的差异与对齐](../templates/feature-document.md#feature-pta-alignment)（初始化版）

---

## Pre-C：ACLNN 调用链分析与子算子盘点（组合场景必做）

> 仅当 PTA C++ 实现中使用**多个 ACLNN 小算子串联**时执行。
> 仅调用单个 `aclnnXxx` 时跳过此步。

### 执行步骤

1. **提取 ACLNN 调用链**：从 PTA C++ 代码中提取前向+反向的全部
   `EXEC_NPU_CMD` / `aclnnXxx` 调用（详见 [`reference.md` 22.2 调用链提取方法](reference.md#aclnn-callchain-extraction)）
2. **盘点 MS 覆盖情况**：逐个搜索确认子算子是否已接入（[`reference.md` 22.3 MS 侧覆盖盘点方法](reference.md#ms-coverage-inventory)）
3. **产出覆盖盘点表**（使用 `templates/aclnn-callchain-inventory.md` 模板）
4. **规划实施顺序**：叶子算子先、组合算子后；按拓扑序（[`reference.md` 22.5 实施顺序原则](reference.md#callchain-rollout-order)）

---

## 成功标准

**⛔ HARD GATE：在进入 Step 1 之前，以下两项必须完成并交付给用户：**
1. ✅ PTA 源码审查报告（Pre-B 产出，使用 `templates/pta-analysis-report.md` 模板）
2. ✅ 已初始化 Feature 文档

**⚠️ "交付给用户"的含义：生成实际的 .md 文件到工作区，并告知用户文件路径。**
