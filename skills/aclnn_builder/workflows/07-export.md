# Workflow 7: 导出与占位

## 目标

确保算子通过 mint 包正确导出，非 Ascend 设备有清晰错误占位；接口满足对标 PyTorch 要求。

## 输出

- **ops 包导出**：`__init__.py` / `__all__` 更新
- **接口文件**：functional / nn / Tensor 方法（按需）

**接口对标约束**：接口名、参数名、顺序与默认值、输入范围（dtype）与pta一致

---

## 执行步骤

### Step 1：mint 显式导出

- 在 `mindspore/python/mindspore/mint/` 相关 `__init__.py` 中**对应算子类别**和 **`__all__` 两处**添加算子名。
- 确保 `__all__` 列表包含新算子

### Step 2：接口开发（[`reference.md` 15 接口开发要点](reference.md#api-development)）

| 接口类型 | 要点 |
| --- | --- |
| **functional** | 内部用 `_get_cache_prim` 获取 Primitive（避免反复 __init__） |
| **nn** | Cell 子类；`construct` 中不直接 `raise`，用 `@constexpr` |
| **Tensor 方法** | 覆盖 PyNative/KBK/GE 模式（按项目要求）。**GE 模式**需在 `resource.cc` 注册映射、`standard_method.py` 实现；该校验函数不能接收 Tensor 类型入参（见 2. 接口开发 2.4） |

### Step 2.5：接口重载配置（如涉及同名多签名，[`reference.md` 25 接口重载适配](reference.md#api-overload-adaptation)）

若目标算子存在同名接口重载（如 Tensor-Scalar / Tensor-Tensor 两种入参，或有/无 keyword-only 参数等），按以下步骤处理：

1. **分析重载场景**：确认属于 [`reference.md` 25.2 四种典型重载场景](reference.md#api-overload-scenarios) 的哪种场景（入参类型不同 / kwonlyargs / 新旧兼容 / 符号别名）
2. **编写 api_def YAML**：在 `ops/api_def/{op_name}.yaml` 中定义多条 `op_yaml` 条目，每条对应一种签名
3. **如旧接口不兼容** → 新增 `ops/op_def/deprecated/{op_name}_method.yaml`，并在 `deprecated_tensor_method.py` 中注册旧接口映射
4. **如有符号别名** → 新增 alias YAML（如 `__mul__.yaml: alias: mul`）
5. **如涉及 functional 重载** → `interface` 字段增加 `function`，更新 `mint/__init__.py` 导入源为 `functional_overload`

## 限制

`xxxExt`, `xxx_ext`接口只是内部接口，export为对外接口时应该`import xxx_ext as xxx`，移除ext后缀。永远不要直接对外ext接口。

---

## 成功标准

- [ ] 可从mint正常 import 算子
- [ ] functional/nn/Tensor 接口可用（按需）
- [ ] `_get_cache_prim` 使用正确（functional 接口）
- [ ] 接口重载：api_def 多条目配置正确、deprecated YAML 参数与 py_method 一致（如涉及）

---