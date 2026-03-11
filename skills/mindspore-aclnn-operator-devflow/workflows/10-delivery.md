# Workflow 10: 转测交付

## 目标

完成 Feature issue内容填写、PR 提交、验收 checklist，确保交付件齐全。

## 输入

- **全部实现代码**：YAML / Infer / PyBoost / KBK / BPROP / 接口 / 测试 / 文档

## 输出

- **Feature 文档**：完整填写所有章节（评审与转测必须产物）
- **PR**：代码提交（通过质量门禁）
- **验收 checklist**：填写完整

---

## 执行步骤

### Step 1：质量门禁检查（[`checklists.md` 11 质量门禁](../checklists.md#checklists-quality-gates)）

提交前确保通过：
- 行长 ≤ 120
- UTF-8 编码
- 无行尾空格
- Check_Pylint / Check_Notebooklint / Check_Shellcheck 等

### Step 2：补齐 Feature 文档（[`reference.md` 24 Feature 文档](reference.md#feature-document-reference)）

Feature 文档是评审与转测交付的**必须产物**，此时需补齐开发过程中各章节：

1. **检查前半部分**：确认 Pre-B 阶段填写的 [1. 背景描述](../templates/feature-document.md#feature-background)、[2. 标杆与接口](../templates/feature-document.md#feature-benchmark-api)、[3. 任务清单](../templates/feature-document.md#feature-task-list)、[4. 功能与接口说明](../templates/feature-document.md#feature-functional-spec)、[6. 约束与类型](../templates/feature-document.md#feature-constraints)、[8. 与 PTA 的差异与对齐](../templates/feature-document.md#feature-pta-alignment) 是否需要根据实际开发更新
2. **补齐开发章节**：
   - [5. YAML 定义](../templates/feature-document.md#feature-yaml-definition) ← Step 1 产出
   - [7. 执行模式与适配](../templates/feature-document.md#feature-execution-modes) ← Step 4/5 产出
   - [9. 动态 Shape/Rank 支持](../templates/feature-document.md#feature-dynamic-shape) ← Step 3 产出
   - [10. 异常与校验](../templates/feature-document.md#feature-validation-and-errors) ← Step 3/4 产出
   - [11. 反向（BPROP）](../templates/feature-document.md#feature-bprop) ← Step 6 产出
   - [12. 测试方案](../templates/feature-document.md#feature-test-plan) ← Step 8 产出
3. **填写 [13. 代码与文件改动说明](../templates/feature-document.md#feature-code-change-summary)**：列出所有新增/修改的文件路径
4. **填写 [14. 验收报告](../templates/feature-document.md#feature-acceptance-report)**：
   - 资料验证表（17 项）：逐项自测并标注结果
   - 功能验证表（26 项）：逐项自测并标注结果
   - 安全编码检视表（12 项）：逐项自测并标注结果
5. **更新 [3. 任务清单](../templates/feature-document.md#feature-task-list)**：每项的最终状态改为"新增/修改/无变更/不涉及"

### Step 3：交付件检查（[`reference.md` 9.3 交付件与验证范围](reference.md#delivery-artifacts-and-scope)）

转测覆盖范围：
- 接口形态：NN / functional / Tensor
- 后端：Ascend（复用存量原语/接口时须确保 CPU/GPU 不回退；全新算子仅 Ascend）
- 模式：动态图 / 静态图 / KBK
- shape：动态/静态
- 维度：泛化性 + 精度 + 性能

### Step 4：PR 提交

- 提交前过 `checklists.md` 末尾的"提交前必检 Top-25"
- PR 描述包含 Feature 文档链接和关键变更说明

---

## 成功标准

- [ ] Feature 文档所有 14 个章节已填写完整
- [ ] Feature 文档验收报告四张表（资料验证 + 功能验证 + 安全编码）已逐项自测
- [ ] 质量门禁全部通过
- [ ] 提交前 Top-25 checklist 全部 ✅
- [ ] PR 描述清晰，关联 Feature 文档
- [ ] 交付件覆盖范围确认

---