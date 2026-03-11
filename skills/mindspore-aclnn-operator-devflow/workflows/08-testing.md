# Workflow 8: 测试

## 目标

完成 C++ UT ，确保功能、动态 shape 全覆盖。

## 输入

- **算子实现**：YAML / Infer / PyBoost / KBK / BPROP

## 输出（两类测试，逐项确认）

> **⚠️ 以下两类测试是 Step 8 的必须产出，每一类都要明确标注状态。**

| 类型 | 文件位置 | 必须程度 | 状态标注 |
| --- | --- | --- | --- |
| **C++ UT** | `tests/ut/cpp/ops/test_ops_{op_name}.cc` | `[MUST]` 必须新建 | ✅已写 / ❌未写（说明原因） |

不生成st用例。ST用例由其他任务负责。

## 执行步骤

### Step 1：C++ UT（[`reference.md` 8.1 C++ UT](reference.md#testing-cpp-ut)）—— 必须新建

典型构造：
- 标量：`ShapeVector{}` + `CreateScalar<T>(value)`
- tuple：`ShapeArray{{}}` + `ValuePtrList{...}`
- None：`kMetaTypeNone` + `kNone`
- unknown：`kValueAny`

参照同类算子的已有 C++ UT 文件确认测试宏和参数结构。

---

## 🔒 Step 8 完成前强制检查（不可跳过）

**在标记 Step 8 为完成之前，必须逐项确认以下清单：**

测试产出检查清单：

C++ UT 文件：
  - 文件路径：tests/ut/cpp/ops/test_ops_{op_name}.cc
  - 状态：✅已新建 / ❌未写（原因：___）

> 如果 C++ UT 的状态为 ❌，**必须说明原因并暂停等用户确认后再继续**。
> 不允许静默跳过。

## 成功标准

- [ ] **C++ UT 文件已产出**（Infer 推导覆盖 unknown/None/动态shape）
- [ ] 覆盖场景：动态 shape / 静态 shape / 非连续 tensor / 空 tensor / 特殊值

---