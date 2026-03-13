# mindspore-build

AI coding agent skill for building MindSpore from source. Works with Cursor, Claude Code, Trae, OpenCode, and other skill-compatible AI coding tools.

## Overview

Automatically selects the correct build command based on target platform,
hardware environment, and user intent. Supports local builds (full/incremental/
plugin-only/UT) and remote builds (deploy code via SSH and compile).

## Supported Platforms

| Platform | Chip Versions | OS | Verified |
|----------|--------------|-----|----------|
| Ascend (NPU) | 910, 910b, a5, 310 | Linux | Yes (910B3, openEuler aarch64) |
| GPU (CUDA) | CUDA 11.1, 11.6 | Linux | Not yet |
| CPU | x86_64, ARM | Linux, macOS, Windows | Not yet |

## Deployment Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Local** | Agent runs on the build machine | Agent is deployed on an Ascend/GPU server |
| **Remote** | Agent runs elsewhere, builds via SSH | Agent is on an external machine, build server is on internal network |

## Usage

### For Users

Use natural language:

- "编译 MindSpore" — auto-detects environment and builds
- "在 910b 上编译" — Ascend build with `-V 910b`
- "编译 UT" — C++ unit test build
- "把代码推到服务器编译" — remote build via SSH

The skill will:
1. Detect hardware and installed toolkits
2. Choose full/incremental/plugin-only based on what changed
3. Execute the build and verify the result
4. On failure, diagnose errors and suggest fixes

### For Other Skills

Invoke this skill when a compilation step is needed:

> "Use the mindspore-build skill to compile for Ascend before running ST tests."

Returns success/failure and the install path.

### Remote Build Setup

Required when the agent cannot build locally:

1. Copy `servers.example.json` → `servers.json`
2. Fill in server IP, user, repo path, and build command
3. Choose an authentication method:

| Method | `auth_method` | Setup |
|--------|---------------|-------|
| SSH Key (recommended) | `"ssh_key"` | Run `ssh-copy-id user@host` |
| Environment variable | `"env"` (default) | Export `MS_SSH_PASS_<DEVICE>` (e.g. `MS_SSH_PASS_910B`) |
| JSON plaintext | `"password"` | Add `"password"` field (not recommended) |

4. Add `servers.json` to `.gitignore`

See `workflows/remote-build.md` for full details.

## File Structure

```
mindspore-build/
├── SKILL.md                         # Entry point: decision tree + auto-detection
├── README.md                        # English documentation
├── README_zh.md                     # Chinese documentation
├── servers.example.json             # Remote server config template
├── scripts/
│   ├── remote_deploy_build.py       # Remote deploy + build script
│   ├── verify_build.py              # Post-build verification (import → tensor → device → network)
│   ├── analyze_build_log.py         # [EXPERIMENTAL] Build log analyzer
│   ├── probe_env.sh                 # Environment probe script (read-only)
│   └── setup_build.sh               # [EXPERIMENTAL] Docker environment setup script
└── workflows/
    ├── server-init-ascend.md        # Bare-metal server setup (deps, conda, CANN, clone)
    ├── docker-build-ascend.md       # Docker build environment (toolchain version isolation)
    ├── build-ascend.md              # Ascend build guide
    ├── build-gpu.md                 # GPU (CUDA) build guide
    ├── build-cpu.md                 # CPU build guide (Linux/macOS/Windows)
    ├── build-ut.md                  # C++ UT build guide
    ├── remote-build.md              # Remote build via SSH
    └── version-matrix.md            # Version compatibility matrix
```

## Design Principles

- **Minimal entry point**: `SKILL.md` (~140 lines) is a router that selects the workflow.
- **Self-contained workflows**: Each workflow contains all information for its scenario.
  No cross-file references.
- **Low context cost**: `SKILL.md` + 1 workflow is all the AI needs to load per task.
- **Auto-detection**: Determines deployment mode and target without user input.

## Knowledge Sources

- `mindspore/build.sh` and `scripts/build/*.sh` — build system source analysis
- `mindspore/cmake/external_libs/*.cmake` — dependency download mechanism
- `docs/install/mindspore_*_install_source*.md` — official installation guides
