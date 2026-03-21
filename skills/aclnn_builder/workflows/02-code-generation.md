# Workflow 2: 代码生成

## 目标

运行 `mindspore/python/mindspore/ops_generate/gen_ops.py`，基于 YAML 生成算子代码。**gen_ops.py 在两条路径下作用不同：**
- **路径 1（自动生成）**：生成完整的 PyBoost/KBK 调用代码 + 注册代码 + Python 接口
- **路径 2（Customize）**：生成包装代码（调用手写的 Customize 类）+ Python 接口

## 输入

- **YAML 文件**：Workflow 1 产出的 op_def / api_def / function_doc
- **接入路径**：auto/customize

## 输出

- **gen_ops.py 运行成功**

**重要**：MindSpore编译和接口调用依赖自动生成文件。每次修改 YAML 后都必须重新运行 gen_ops.py 更新自动生成文件。

---

## 执行步骤

### Step 1：运行 gen_ops.py

```bash
python mindspore/ops/op_def/gen_ops.py
```

### Step 2：确认自动生成产物

运行完成后，**必须确认**以下文件已正确生成：

| 文件 | 路径 1 | 路径 2 | 说明 |
| --- | --- | --- | --- |
| PyBoost 调用代码 | **完整生成** | 生成包装 | 路径 1 直接调用 ACLNN；路径 2 调用 Customize 类 |
| KBK 自动注册 | **完整生成** | 不生成 | 路径 2 需手写 kernel 并手动注册 |


---

## 成功标准

- [ ] `gen_ops.py` 运行无报错
- [ ] 路径 1：确认 PyBoost 调用代码和 aclnn kernelmod 注册已自动生成
  - aclnn kernelmod: mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen, mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/auto_generate
  - pyboost: mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate

---
