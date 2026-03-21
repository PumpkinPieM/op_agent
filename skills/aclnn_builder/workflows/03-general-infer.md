# Workflow 3: GeneralInfer（C++ 推导）

## 目标

使用GeneralInfer流程实现算子的输出shape/dtype推导（C++），需要注意输入为动态shape/rank场景(dim, rank在当前阶段未知)。

## 输入

- **YAML 定义**：参数列表、输出结构
- **PTA 源码分析**：输出 shape 推导逻辑

## 输出

- **Infer 实现文件**：`mindspore/ops/infer/ops_func_impl`

---

## 执行步骤

GeneralInfer中主要通过inferinfo类接口获取输入信息进行推导。 

基础接口：
```cpp
class OPS_API XXX : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; };
};
```
- 使用InferInfoPtrList入参的infer接口，同时重载GeneralInferRegistered接口返回true，注册GeneralInfer流程。
- inferinfo定义： `mindspore/core/include/ops/infer_info/infer_info.h`
- infer基类定义: `mindspore/core/include/ops/ops_func_impl/op_func_impl.h`


### Step 1：实现 InferShape

职责边界（[`reference.md` 4.1 职责边界](reference.md#general-infer-responsibilities)）：
- **只做推导**，不做入参合法性校验（交给kernel）
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

---