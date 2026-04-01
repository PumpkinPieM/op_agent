# ACLNN Builder Pre 阶段蒸馏仓设计方案

## 1. 背景

`aclnn-builder` 当前的 `Pre` 阶段要求 agent 在本地直接搜索 `mindspore`、`op-plugin`、以及 ACLNN 文档，完成以下工作：

1. `Pre-A`：盘点 MindSpore 侧是否已经存在 YAML、Infer、PyBoost、KBK、BPROP、测试、文档。
2. `Pre-B`：审查 PTA 三件套，分析接口差异，选择 `auto` / `customize` 路径，输出 PTA 差异报告。
3. `Pre-C`：对于组合算子，抽取 ACLNN 调用链，并逐个子算子回查 MindSpore 覆盖状态。

这些要求分别定义在：

- `workflows/00-pre-checks.md`
- `templates/pta-analysis-report.md`
- `templates/aclnn-callchain-analysis.md`
- `../operator-agent/workflows/native-framework/mindspore/_shared/reference.md`

现状的问题不是“检索能力不够”，而是“每次单算子分析都重复扫描两个代码仓，成本高且噪声大”。

因此需要引入一个新的中间层：

- 离线全仓扫描
- 在线读取蒸馏结果
- 仅在低置信度或证据冲突时回退到定点源码搜索

## 2. 设计目标

蒸馏仓的目标不是替代源码仓，也不是构建一个泛化的 RAG 知识库，而是为 `Pre-A / Pre-B / Pre-C` 提供稳定、结构化、可复核的输入。

具体目标：

1. 避免 builder 在正常场景下对 `mindspore` 和 `op-plugin` 做全仓搜索。
2. 让单算子分析可以直接产出 `pta-analysis-report` 和 `callchain-analysis` 所需事实。
3. 保留证据链，保证每条结论都能回溯到源文件和行号。
4. 支持增量更新，避免每次都重建全量知识。

## 3. 设计原则

### 3.1 结构化优先，不做纯向量库

`Pre` 阶段关心的是确定性事实：

- API 身份解析
- branch 到 primitive 的映射
- `auto` / `customize` 路径
- PTA 前后向签名
- ACLNN 调用链
- BPROP 注册和 backward 形态

这些内容本质上都适合做成结构化表和事实包，不适合只做 embedding 检索。

向量检索最多作为补充，用于找相似算子模板，不应该承担主路径。

### 3.2 branch 是最小分析单元，不是 public API 名字

一个 public API 可能对应多个 `api_def` branch，不同 branch 的：

- `op_yaml`
- `primitive`
- Ascend dispatch
- backward 形态

都可能不同。

因此蒸馏索引不能仅用 `mindspore.mint.xxx` 作为 key。当前两张基础索引的主键约定为：

```text
api_identity: public_api + op_branch
ms_coverage: op + primitive
```

组合算子再附加：

```text
pta_entry + callchain_id
```

### 3.3 builder 消费 bundle，不直接消费原始索引

批量索引的作用是全局路由、筛选、join。
builder 真正读取的应该是单算子 `bundle`，而不是自己去拼多张表。

## 4. 推荐目录结构

建议引入一个新的蒸馏目录，例如：

```text
distill/
  snapshots/
    2026-03-31/
      manifest.json
      indexes/
        torch_coverage.parquet
        ms_coverage.jsonl
        ms_coverage.csv
        api_identity.jsonl
        api_identity.csv
        branch_dispatch.parquet
        bprop_inventory.parquet
        callchain_inventory.parquet
      ops/
        add/
          add__add_op__Add/
            bundle.json
            evidence.json
            render/
              pta_analysis.md
              callchain.md
        sparse_flash_attention/
          npu_sparse_flash_attention__custom/
            bundle.json
            evidence.json
            render/
              pta_analysis.md
              callchain.md
      patterns/
        elementwise_binary_auto.yaml
        reduction_customize.yaml
        composite_attention.yaml
```

## 5. 目录与文件说明

### 5.1 `snapshots/<date>/`

表示一次完整的离线扫描产物。

职责：

- 固化一次扫描时对应的源码版本
- 支持回溯“某条知识来自哪个版本”
- 支持 diff 两次快照之间的变化

### 5.2 `manifest.json`

记录这次快照的元信息，建议包含：

- `mindspore_sha`
- `op_plugin_sha`
- `aclnn_doc_version`
- `generated_at`
- `schema_version`
- `scanner_version`

作用：

- 判断知识是否过期
- 支持 builder 在运行时做 freshness check

### 5.3 `indexes/`

这是“批量索引区”，面向全量查询，不面向最终单算子消费。

当前 `api_identity` 和 `ms_coverage` 已直接产出为：

- `JSONL`
- `CSV`

原因：

- `JSONL` 适合脚本直接消费
- `CSV` 适合人工抽查和回归对比
- 当前阶段先把字段和主键稳定下来，再考虑切换到 `Parquet` / `DuckDB`

### 5.4 `ops/`

这是“单算子事实包区”，面向 builder。

组织方式建议：

- 第一层：按 canonical op 名或用户常见名分组
- 第二层：按 branch-level 唯一 key 建目录

### 5.5 `bundle.json`

这是 builder 的主输入文件。

它应该直接回答以下问题：

- 这个分析单元是谁
- PTA 实际调用了什么
- MindSpore 当前覆盖到什么程度
- 应该走 `auto` 还是 `customize`
- 是否是组合算子
- 需要哪些后续工作

### 5.6 `evidence.json`

记录事实与证据的映射关系，建议包含：

- `repo`
- `path`
- `line_start`
- `line_end`
- `kind`
- `excerpt_hash`
- `parser_rule`

作用：

- 每个结论都可复核
- 后续模板渲染可以直接插入证据路径
- 源码变化时可以快速定位失效条目

### 5.7 `render/*.md`

面向人类审阅的渲染结果。

建议至少产出：

- `pta_analysis.md`
- `callchain.md`

它们可以由 `bundle.json + evidence.json` 渲染生成，用于：

- 方案评审
- 与现有模板对齐
- 在需要时直接交付给用户

### 5.8 `patterns/`

存的是“模式模板”，不是“事实证据”。

例如：

- `elementwise_binary_auto`
- `reduction_customize`
- `composite_attention`
- `loss_with_reduction_and_cache`

用途：

- 给 builder 提供相似实现参考
- 辅助生成初始决策建议
- 不能代替真实证据

## 6. 批量索引是什么

批量索引本质上是把源码扫描结果抽成表。

它索的不是全文，而是 `Pre` 真正需要的实体和关系。

### 6.1 `torch_coverage.parquet`

来源：

- `op_plugin/config/op_plugin_functions.yaml`
- `op_plugin/config/derivatives.yaml`
- `op_plugin/ops/opapi/*.cpp`

关注事实：

- `aclnn_api -> torch/torch_npu entry`
- 前后向入口
- 命中的 `cpp_files`
- PTA 侧是否直接调用该 ACLNN
- 是否疑似融合

这个层可以复用现有 `aclnn-dashboard` 的 Torch-NPU 覆盖扫描结果。

### 6.2 `ms_coverage`

来源：

- `mindspore/ops/op_def/yaml/`
- `mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml`
- `mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/auto_generate/`
- `mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize/`
- `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/`
- `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/`
- Infer 实现
- BPROP 注册
- PyBoost
- KBK
- 测试和文档

关注事实：

- 主键：`op + primitive`
- 某个 `op / primitive` branch 是否已有：
- YAML
- Infer
- PyBoost
- KBK
- BPROP
- UT / ST
- 文档
- `dispatch_enable`
- `dispatch_kind`
- `dispatch_ascend`
- `aclnn`

其中：

- `aclnn` 字段名固定叫 `aclnn`
- 单值和多值都以 JSON 数组字符串保存，例如 `["aclnnSoftmax"]`
- 当前阶段不拆附表，多个 ACLNN 直接保存在同一行

当前输出路径：

- `operator-facts/data/ms_coverage.jsonl`
- `operator-facts/data/ms_coverage.csv`

这个层可以参考现有 `aclnn-dashboard` 的 MindSpore 覆盖结果，但 Ascend 侧判定逻辑以仓库真实代码和 `api-knowledge/backend-lens-ascend.md` 为准。

### 6.3 `api_identity`

来源：

- `mindspore/python/mindspore/mint/__init__.py`
- public export
- wrapper function
- `api_def`
- `op_yaml`

关注事实：

- 主键：`public_api + op_branch`
- `mindspore.mint.xxx -> internal symbol -> op_yaml -> primitive`
- `mindspore.ops.xxx -> op_yaml -> primitive`
- `mindspore.Tensor.xxx -> op_yaml -> primitive`
- 是否 alias export
- 是否 overload
- 哪些 branch 仍然 active
- `py_method`
- `interface`
- `target_module`
- `target_symbol`
- `resolver_kind`
- `resolver_path`

当前输出路径：

- `operator-facts/data/api_identity.jsonl`
- `operator-facts/data/api_identity.csv`

这个层可以基于现有 `api-knowledge/api-opname-inventory.md` 进一步结构化。


### 6.4 `branch_dispatch.parquet`

来源：

- `op_yaml`
- `dispatch.enable`
- `Ascend: XxxAscend`
- `aclnn_config.yaml`
- customize 源码路径

关注事实：

- branch 的 Ascend 路径是 `auto_generate`、`customize` 还是 unsupported
- PyBoost / KBK 的静态证据是否存在

### 6.5 `bprop_inventory.parquet`

来源：

- `REG_BPROP_BUILDER`
- backward body 中的 `Emit("XxxGrad")`
- inline backward 中的组合算子

关注事实：

- primitive 是否有 bprop
- backward 是 dedicated grad 还是 inline composition
- backward 实际依赖哪些 operator

### 6.6 `callchain_inventory.parquet`

来源：

- PTA C++ 中的 `EXEC_NPU_CMD`
- `EXEC_NPU_NO_FORMAT_CHECK_CMD`
- 相关 helper 函数链

关注事实：

- 某个 PTA 接口是否为组合算子
- 前向调用链节点
- 反向调用链节点
- 每个子算子在 MindSpore 的覆盖情况

这是 `Pre-C` 最关键也最贵的索引层。

## 7. bundle 是什么

`bundle` 是“一个单算子的完整蒸馏事实包”。

关系如下：

- 索引是全量表，负责路由和筛选
- bundle 是单算子物化结果，负责被 builder 直接消费

一个最小可用的 `bundle.json` 示例：

```json
{
  "snapshot": {
    "mindspore_sha": "xxx",
    "op_plugin_sha": "yyy",
    "generated_at": "2026-03-31T12:00:00+08:00"
  },
  "identity": {
    "public_api": "mindspore.mint.add",
    "pta_api": "torch.add",
    "op_yaml": "add_op.yaml",
    "primitive": "Add"
  },
  "inventory": {
    "yaml": true,
    "infer": true,
    "pyboost": true,
    "kbk": true,
    "bprop": true,
    "ut": true,
    "st": true,
    "docs_en": true,
    "docs_cn": true
  },
  "pta": {
    "forward_signature": "...",
    "backward_signature": "...",
    "forward_aclnn_calls": ["aclnnAdd"],
    "backward_aclnn_calls": [],
    "param_preprocess": [],
    "hardcoded_args": [],
    "mismatches": []
  },
  "decision": {
    "integration_type": "reuse_primitive",
    "yaml_strategy": "modify_existing",
    "integration_path": "auto",
    "composite": false
  },
  "callchain": {
    "forward": [],
    "backward": []
  },
  "similar_patterns": [
    "elementwise_binary_auto"
  ],
  "evidence_refs": [
    {
      "repo": "mindspore",
      "path": "...",
      "line": 12
    },
    {
      "repo": "op-plugin",
      "path": "...",
      "line": 34
    }
  ]
}
```

## 8. bundle 字段和 Pre 阶段的映射

### 8.1 映射到 `Pre-A`

`Pre-A` 需要的内容主要来自：

- `identity`
- `inventory`
- `evidence_refs`

builder 可以直接用它填 inventory 表，而不用再在 MindSpore 仓全仓搜。

### 8.2 映射到 `Pre-B`

`Pre-B` 需要的内容主要来自：

- `pta.forward_signature`
- `pta.backward_signature`
- `pta.forward_aclnn_calls`
- `pta.backward_aclnn_calls`
- `pta.param_preprocess`
- `pta.hardcoded_args`
- `pta.mismatches`
- `decision`

builder 可以直接渲染 PTA 差异报告，并给出：

- primitive 复用还是新增
- YAML 策略
- `auto` / `customize`

### 8.3 映射到 `Pre-C`

`Pre-C` 需要的内容主要来自：

- `callchain.forward`
- `callchain.backward`
- 子算子覆盖状态
- rollout 顺序

组合算子场景下，builder 可以直接生成 call-chain inventory，而不用重新分析 C++ 调用链。

## 9. builder 怎么消费

建议 builder 的运行流程改成三段式。

### 9.1 第一步：输入标准化

输入可能是：

- `mindspore.mint.xxx`
- `torch.xxx`
- `torch_npu.npu_xxx`
- `aclnnXxx`

先查索引，得到 canonical 分析单元 key：

```text
public_api + op_branch
```

### 9.2 第二步：读取 bundle

如果对应 `bundle.json` 存在：

- 直接读 bundle
- 填 `Pre-A`
- 填 `Pre-B`
- 如果 `decision.composite == true`，继续填 `Pre-C`

### 9.3 第三步：按条件回退源码搜索

只有以下场景才回源码：

- 没有 bundle
- bundle 的 snapshot 过旧
- `pta.mismatches` 非空
- 证据不完整
- 组合调用链解析置信度低

并且回退也应该是：

- 定点文件搜索
- 定点行号验证

而不是再次对两个仓做全仓搜索。

## 10. builder 的直接输入建议

builder 最好不要直接读取多张索引表，而是接收一个标准化的 `pre_bundle`。

推荐格式：

```json
{
  "bundle_path": "distill/.../bundle.json",
  "render_hint": {
    "emit_pta_report": true,
    "emit_callchain_report": true
  },
  "fallback_policy": {
    "allow_targeted_source_check": true,
    "allow_repo_wide_search": false
  }
}
```

这样 builder 的职责会清晰很多：

- 负责消费蒸馏结果
- 负责补齐模板输出
- 不负责重新建立知识

## 11. 离线构建流水线建议

建议把蒸馏仓的生成拆成 4 个离线步骤。

### 11.1 Step 1：源扫描器

扫描：

- MindSpore
- op-plugin
- ACLNN 文档

抽出原子事实。

### 11.2 Step 2：规范化

完成：

- 命名统一
- branch 解析
- primitive 解析
- `auto/customize` 解析
- backward 形态归类

### 11.3 Step 3：聚合成 bundle

把多张索引 join 成单算子事实包。

### 11.4 Step 4：模板渲染

从 bundle 渲染：

- `pta_analysis.md`
- `callchain.md`

## 12. 与现有资产的关系

当前工作区已有两个可以复用的基础层：

### 12.1 `aclnn-dashboard`

适合作为 `L0` 覆盖索引来源：

- Torch-NPU 覆盖
- MindSpore 覆盖
- 基础证据字段

### 12.2 `api-knowledge`

适合作为 `L1` 规则层来源：

- API identity
- Ascend backend lens
- backward inventory

真正缺的是 `L2` 单算子 bundle 层。

## 13. 分阶段落地建议

### Phase 1：先跑通直接单 ACLNN 算子

范围：

- 非组合
- 无 overload 或 overload 较少
- `auto/customize` 易判断

目标：

- 先产出稳定的 `bundle.json`
- 让 builder 能跳过大部分 Pre-A / Pre-B 全仓搜索

### Phase 2：补组合算子调用链

范围：

- PTA C++ 中有多个 `EXEC_NPU_CMD`
- 反向是组合式

目标：

- 建 `callchain_inventory`
- 让 `Pre-C` 也可离线消费

### Phase 3：加相似模式和增量更新

目标：

- 增量构建
- 模式推荐
- 低置信度回退策略

## 14. 非目标

以下内容不建议一开始就做：

- 通用大模型知识库
- 全文 embedding 作为主检索路径
- 图数据库先行
- 自动替 builder 做所有设计决策

第一阶段更重要的是：

- 把事实抽对
- 把证据带全
- 把 builder 的输入稳定下来

## 15. 结论

推荐方案可以概括成一句话：

```text
离线扫描两仓 -> 抽结构化索引 -> 聚合单算子 bundle -> builder 在线消费 bundle -> 仅在异常场景回退定点源码检查
```

这个方案的核心价值不在“检索更聪明”，而在“builder 不再重复建立知识”。

对 `aclnn-builder` 来说，最值得先做的不是更强的搜索器，而是先定义一份稳定的 `bundle schema`，让 `Pre-A / Pre-B / Pre-C` 都能围绕这份 schema 来消费。
