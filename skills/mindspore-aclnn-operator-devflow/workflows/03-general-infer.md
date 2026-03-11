# Workflow 3: GeneralInfer（C++ 推导）

## 目标

实现算子的输出shape/dtype推导（C++），需要注意输入为动态shape/rank场景(dim, rank在当前阶段未知)。
可选：实现 InferValue 常量折叠。

## 输入

- **YAML 定义**：参数列表、输出结构
- **PTA 源码分析**：输出 shape 推导逻辑

## 输出

- **Infer 实现文件**：`op_name_general_infer.cc`（或项目对应路径）
- **（可选）InferValue 实现**

---

## 执行步骤

GeneralInfer中主要通过inferinfo类接口获取输入信息进行推导。 常用 API（[`reference.md` 4.3 常用 InferInfo API](reference.md#general-infer-api)）

代码骨架见 [`reference.md` 18.2 GeneralInfer 骨架](reference.md#general-infer-skeleton)。

### Step 1：实现 InferShape

职责边界（[`reference.md` 4.1 职责边界](reference.md#general-infer-responsibilities)）：
- **只做推导**，不做运行时合法性校验（交给 ACLNN/运行时）
- 报错使用框架异常宏，包含：参数名、期望、实际

### Step 2：处理动态 shape/rank

三种动态类型及策略（[`reference.md` 21 动态 shape 分类与处理策略](reference.md#dynamic-shape-strategy)）：

| 类型 | Infer 策略 |
| --- | --- |
| InputDynamic | 输出对应维度设为 `kShapeDimAny` |
| Input Value Depend | `GetShapeValue()` 取值；unknown 时回退 |
| Compute Depend | 分配最大可能 size + 运行后 SyncOutputShape |

快速回退策略（[`reference.md` 4.2 动态 shape / 动态 rank](reference.md#general-infer-dynamic-shape-rank)）：
- 动态 rank → 返回 `kShapeRankAny`
- 关键参数 unknown → 对应维度回退 `kShapeDimAny`
- 参数都已知 → 返回精确 shape

### Step 3：实现 InferType

通常输出 dtype 与输入一致或按算子语义确定。

---

## 成功标准

- [ ] InferShape 实现完成，覆盖精确推导和动态回退
- [ ] InferType 实现完成
- [ ] 编译通过，无链接错误
- [ ] C++ UT 可构造 unknown/None 输入并验证推导结果
- [ ] （可选）InferValue 实现并验证

---

## 注意事项

- **InferInfo API**：以`mindspore/core/include/ops/infer_info/infer_info.h`中接口定义为主，按项目已有用法写（如 `GetScalarValueWithCheck` / `GetArrayValue` / `HasUnknownValue` / `IsNone`），不要臆造 API。
- **Infer 职责**：只做推导，不做运行时合法性校验（合法性让ACLNN接口内部处理）。