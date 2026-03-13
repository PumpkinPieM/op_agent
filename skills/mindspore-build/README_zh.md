# mindspore-build

用于从源码编译 MindSpore 的 AI Coding Agent Skill，适用于 Cursor、Claude Code、Trae、OpenCode 等支持 skill 的 AI 编程工具。

## 概述

根据目标平台、硬件环境和用户意图，自动选择正确的编译命令。支持本地编译（全量/增量/
仅插件/UT）和远程编译（通过 SSH 部署代码并编译）。

## 支持的平台

| 平台 | 芯片版本 | 操作系统 | 验证状态 |
|------|---------|---------|---------|
| Ascend (NPU) | 910, 910b, a5, 310 | Linux | 已验证（910B3, openEuler aarch64） |
| GPU (CUDA) | CUDA 11.1, 11.6 | Linux | 未验证 |
| CPU | x86_64, ARM | Linux, macOS, Windows | 未验证 |

## 部署模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **本地编译** | Agent 运行在编译机上 | Agent 部署在 Ascend/GPU 服务器上 |
| **远程编译** | Agent 在外部机器，通过 SSH 编译 | Agent 在外网，编译服务器在内网 |

## 使用方式

### 用户使用

使用自然语言：

- "编译 MindSpore" — 自动检测环境并编译
- "在 910b 上编译" — Ascend 编译，使用 `-V 910b`
- "编译 UT" — C++ 单元测试编译
- "把代码推到服务器编译" — 通过 SSH 远程编译

Skill 会自动：
1. 检测硬件和已安装的工具包
2. 根据代码变更范围选择全量/增量/仅插件编译
3. 执行编译并验证结果
4. 编译失败时诊断错误并建议修复方案

### 被其他 Skill 调用

当其他 skill 需要编译步骤时调用：

> "使用 mindspore-build skill 编译 Ascend 版本，然后运行 ST 测试。"

返回编译成功/失败及安装路径。

### 远程编译配置

当 agent 无法在本地编译时使用：

1. 复制 `servers.example.json` → `servers.json`
2. 填入服务器 IP、用户名、仓库路径和编译命令
3. 选择认证方式：

| 方式 | `auth_method` | 配置方法 |
|------|---------------|---------|
| SSH 密钥（推荐） | `"ssh_key"` | 执行 `ssh-copy-id user@host` |
| 环境变量 | `"env"`（默认） | 导出 `MS_SSH_PASS_<设备>`（如 `MS_SSH_PASS_910B`） |
| JSON 明文密码 | `"password"` | 添加 `"password"` 字段（不推荐） |

4. 将 `servers.json` 添加到 `.gitignore`

详见 `workflows/remote-build.md`。

## 文件结构

```
mindspore-build/
├── SKILL.md                         # 入口：决策树 + 自动检测逻辑
├── README.md                        # 英文文档
├── README_zh.md                     # 中文文档
├── servers.example.json             # 远程服务器配置模板
├── scripts/
│   ├── remote_deploy_build.py       # 远程部署 + 编译脚本
│   ├── verify_build.py              # 编译后验证（导入 → 张量 → 设备 → 网络）
│   ├── analyze_build_log.py         # [实验性] 编译日志分析脚本
│   ├── probe_env.sh                 # 环境探测脚本（只读）
│   └── setup_build.sh               # [实验性] Docker 环境设置脚本
└── workflows/
    ├── server-init-ascend.md        # 裸机服务器初始化（依赖、Conda、CANN、克隆）
    ├── docker-build-ascend.md       # Docker 编译环境（工具链版本隔离）
    ├── build-ascend.md              # Ascend 编译指南
    ├── build-gpu.md                 # GPU (CUDA) 编译指南
    ├── build-cpu.md                 # CPU 编译指南（Linux/macOS/Windows）
    ├── build-ut.md                  # C++ UT 编译指南
    ├── remote-build.md              # 远程 SSH 编译指南
    └── version-matrix.md            # 版本兼容性矩阵
```

## 设计原则

- **极简入口**：`SKILL.md`（约 140 行）是路由器，只选择 workflow。
- **单文件自足**：每个 workflow 包含该场景的全部信息，无需跨文件跳转。
- **最小上下文消耗**：AI 每次只需加载 `SKILL.md` + 1 个 workflow。
- **自动检测**：自动判断部署模式和目标平台，无需用户手动指定。

## 知识来源

- `mindspore/build.sh` 和 `scripts/build/*.sh` — 编译系统源码分析
- `mindspore/cmake/external_libs/*.cmake` — 依赖下载机制
- `docs/install/mindspore_*_install_source*.md` — 官方安装文档
