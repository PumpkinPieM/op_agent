# Workflow 10: 远程编译验证

## 目标

将代码变更推送到 Ascend 服务器，触发源码编译，自动获取编译结果。
若编译失败，拉取错误日志、定位并修复代码问题后重新推送编译。
**最多重试 3 次**，3 次仍失败则停止并报告最后一次编译错误。

## 输入

- **本地 git 仓库**：已完成 Step 1-9 代码变更并 commit
- **远程服务器信息**：IP、账号、密码、远程仓库目录

## 输出

- 编译成功：报告成功，进入下一步
- 编译失败且已自动修复：给出修复说明 + 最终成功的编译日志
- 3 次重试后仍失败：输出最后一次编译错误详情，停止流程

---

## 前置条件

1. **代码已 commit**：所有 Step 1-9 的改动必须已经 `git commit`
   （脚本基于最新 commit 提取变更文件列表，未 commit 的改动不会被推送）
2. **服务器配置已就绪**：`servers.json`（skill 根目录）中已包含目标服务器信息。
   脚本会自动从中读取 IP、账号密码、远程目录和编译命令，**无需用户手动提供**。
   - 默认使用 `servers.json` 中 `"default"` 字段指定的设备（当前为 `910b`）
   - 若需指定其他设备，用 `--device` 参数（如 `--device 910a`、`--device gpu`）

## 执行步骤

### Step 1：部署并编译（首次）

使用 `scripts/remote_deploy_build.py` 脚本执行部署和编译：

```bash
# 使用默认服务器（servers.json 中的 default）
python <skill_dir>/scripts/remote_deploy_build.py ^
    --local-repo <LOCAL_REPO> ^
    --log-file <WORKSPACE>/build_output.log

# 指定特定设备类型
python <skill_dir>/scripts/remote_deploy_build.py ^
    --device 910b ^
    --local-repo <LOCAL_REPO> ^
    --log-file <WORKSPACE>/build_output.log
```

> **参数说明**：
> - `--device`：可选，从 `servers.json` 中选择服务器（`910b`/`910a`/`gpu` 等），
>   不指定则使用 `"default"` 字段对应的设备
> - `--local-repo`：本地 git 仓库路径
> - `--log-file`：编译日志输出路径
> - `--remote-dir`/`--build-cmd`：可选，覆盖 `servers.json` 中的默认值

**脚本自动完成以下操作**：
1. 从 `servers.json` 读取服务器连接信息
2. 从最新 commit 提取变更文件列表（`git diff-tree`）
3. 打包变更文件（`git archive`）
4. 通过 SCP 上传到服务器（使用 `SSH_ASKPASS` 机制传递密码）
5. 在服务器上解压到目标目录
6. 在服务器上执行编译命令，捕获全部输出到本地 `build_output.log`

### Step 2：检查编译结果

- **脚本退出码 = 0**：编译成功 → 跳到 Step 5
- **脚本退出码 = 1**：编译失败 → 进入 Step 3（错误分析）
- **脚本退出码 = 2**：部署/网络问题 → 停止，报告基础设施问题给用户

### Step 3：错误分析与代码修复（重试循环）

> **最多执行 3 次**。用计数器 `retry_count` 跟踪，初始值为 0。

**3a. 读取编译日志**

用 Read 工具读取 `build_output.log`，重点关注：
- `error:` 开头的行（编译错误）
- `undefined reference` / `no matching function` / `no member named` 等典型 C++ 编译错误
- 文件名和行号信息（定位到具体源文件）

**3b. 定位问题文件**

从错误信息中提取：
- 出错的源文件路径（相对于仓库根目录）
- 行号
- 错误类型（语法错误/类型不匹配/符号未定义/头文件缺失等）

**3c. 修复代码**

- 使用 Read 工具读取出错的源文件
- 分析错误原因，使用 StrReplace 工具修复
- 如果错误涉及多个文件，逐个修复
- **修复后必须 commit**（`git add` + `git commit`），脚本只推送 committed 的变更

**3d. 关键约束**

- 每次修复后**必须**重新 commit，否则改动不会被推送
- 修复应当**最小化**——只修编译错误，不要趁机重构其他代码
- 如果错误原因不明（如环境/依赖问题），不要盲目改代码，停下报告给用户

### Step 4：重新部署并编译（重试）

```
retry_count += 1
if retry_count > 3:
    → 停止！输出最后一次编译错误，告知用户，退出流程
else:
    → 回到 Step 1 重新执行
```

### Step 5：编译成功，输出报告

编译成功后，按 SKILL.md 验证闭环模板输出报告，包含：
- 推送的变更文件数量
- 编译命令和结果
- 若有重试：每次修复的错误和修复方式
- 编译日志路径

---

## 重试失败时的终止行为

当 3 次重试后编译仍然失败，**必须**：

1. **输出最后一次编译错误的关键信息**（错误类型、文件、行号、错误消息）
2. **说明已尝试的修复措施**（每次修复了什么、为什么仍然失败）
3. **明确告知用户**："远程编译在 3 次重试后仍然失败，请检查以下编译错误并手动处理"
4. **不得继续后续步骤**——流程在此终止

---

## 成功标准

- [ ] 变更文件已通过 SCP 推送到远程服务器
- [ ] 编译命令已在远程服务器执行
- [ ] 编译通过（exit code 0），或在 ≤3 次自动修复重试后通过
- [ ] 编译日志已保存到本地 `build_output.log`
- [ ] 若有修复，修复改动已 commit

---

## 下一步

编译成功后，进入 **[Workflow 11: 转测交付](./11-delivery.md)**
