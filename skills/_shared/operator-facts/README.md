# operator-facts

`operator-facts` 是 ACLNN `Pre` 阶段的第一版蒸馏库。

当前阶段只产出两张索引：

- `api_identity`
- `ms_coverage`
- `op_bundle`

## 当前主键约定

### `api_identity`

- 主键：`public_api + op_branch`
- `public_api` 是全限定公共接口，例如 `mindspore.mint.max`
- `op_branch` 是 `api_def` 或 direct `op_def` 落到的分支文件名，例如 `max_dim_op.yaml`

### `ms_coverage`

- 主键：`op + primitive`
- `op` 是 `op_def` 顶层算子名，例如 `max_dim`
- `primitive` 是该分支对应的 Primitive 名，例如 `MaxDim`

## `aclnn` 字段约定

- 字段名统一叫 `aclnn`
- 单值和多值都用 JSON 数组字符串保存
- 例如：`["aclnnSoftmax"]`
- 例如：`["aclnnSoftmax", "aclnnCast", "aclnnReduceSum"]`

当前阶段不拆附表，不做 `branch -> aclnn` 正规化。

## 输出路径

生成结果写到：

- `operator-facts/data/api_identity.jsonl`
- `operator-facts/data/api_identity.csv`
- `operator-facts/data/ms_coverage.jsonl`
- `operator-facts/data/ms_coverage.csv`
- `operator-facts/data/op_bundles.jsonl`
- `operator-facts/bundles/<public_api>/<op_branch>.json`

## 使用方式

```bash
python operator-facts/scripts/build_phase1.py
```

也可以单独运行：

```bash
python operator-facts/scripts/build_api_identity.py
python operator-facts/scripts/build_ms_coverage.py
```

## 当前覆盖范围

### `api_identity`

第一版覆盖：

- `mindspore.ops.*`
- `mindspore.Tensor.*`
- `mindspore.mint.*`

其中：

- `ops/Tensor` 主要来自 `api_def`
- `mint` 主要来自 `mint/__init__.py` 的实际导出关系
- wrapper 解析采用“直接符号 -> `api_def` / `op_def` -> 一层到数层 return 调用追踪”的轻量策略

### `ms_coverage`

第一版覆盖：

- `dispatch`
- `aclnn`
- `infer`
- `pyboost`
- `kbk`
- `bprop`
- `ut`
- `st`
- `docs_cn`
- `docs_en`

其中 `dashboard` 只作为扫描经验参考，不作为 source-of-truth。
Ascend 侧判断优先遵循 `api-knowledge/backend-lens-ascend.md` 的规则：

- 先看 `op_yaml.dispatch`
- 再看 `aclnn_config.yaml`
- customize 走源码取证

### `op_bundle`

第一版 `bundle` 只聚合 `api_identity + ms_coverage`，不包含：

- PTA
- callchain
- decision

当前字段分成：

- `identity`
- `resolver`
- `coverage`
- `evidence`
- `refs`

## 已知限制

- `api_identity` 目前没有覆盖所有复杂 wrapper 的深层控制流。
- `ms_coverage` 当前不区分多个 `aclnn` 的角色，只做 branch 级摘要。
- `op_bundle` 目前只是 phase-1 最小事实包，还没有 PTA 和组合调用链信息。
- 复杂组合算子的调用链还未展开到单独附表。
