---
name: mindspore-build
description: >
  Builds MindSpore from source for any target (Ascend/GPU/CPU) and mode (full/incremental/UT).
  Auto-detects the current machine environment and selects the optimal build command.
  Supports local and remote (SSH) deployment modes.
  Use this skill whenever the user mentions compiling, building, or 编译 MindSpore,
  running C++ UT tests, diagnosing build failures, setting up a new build server,
  or any other skill needs a "compile MindSpore" step (e.g. aclnn operator dev, kernel dev).
  Also use when the user encounters build errors, linker failures, missing dependencies,
  or asks about build commands, build options, ccache, incremental builds, or environment setup.
---

# MindSpore Build

## When to Use

- User says "编译" / "build" / "compile"
- Another skill needs a compilation step (e.g. aclnn workflow step 8)
- Build failure diagnosis
- User asks "which build command should I use"
- User needs to set up a new build server → `workflows/server-init-ascend.md`

## Deployment Modes

This skill supports two deployment scenarios:

| Mode | Agent Location | Build Machine | Workflow |
|------|---------------|---------------|----------|
| **Local** | On the build machine (has NPU/GPU/CPU) | Same machine | `build-ascend/gpu/cpu/ut.md` |
| **Remote** | On a different machine (e.g. external network) | Accessed via SSH | `remote-build.md` |

## HARD RULES

1. **Never downgrade target platform.** NPU machine = Ascend build. Missing deps = install them.
2. **Minimum impact principle.** Fix environment issues with the least invasive method:
   - PREFER: `export PATH=...` / `export CC=...` in env.sh (session-only, zero side effects)
   - OK: `pip install` / `conda install` (user-scoped) or `yum/apt install` (system package)
   - **NEVER**: `ln -sf`, `update-alternatives`, edit `/etc/profile` or `/etc/ld.so.conf`,
     or anything that changes global symlinks / system-wide paths. These break other users.

## Decision Tree

```
Step 1: What hardware does this machine have?
│
├─ npu-smi info succeeds → target = Ascend (LOCKED, no fallback)
├─ nvidia-smi succeeds   → target = GPU    (LOCKED, no fallback)
├─ Neither, but servers.json exists → target = remote (see remote-build.md)
└─ Neither, no servers.json → target = CPU
│
Step 2: Is the build environment ready?
│  Run the checks from the target workflow's "Prerequisites" section.
│
├─ All checks pass → Step 3
│
└─ Some checks fail (missing GCC, CMake, CANN, Python deps, etc.)
   │  DO NOT change target. DO NOT skip to a different platform.
   │  Install the missing dependencies directly:
   │
   ├─ Default: install what's missing on the machine
   │  → workflows/build-ascend.md "Install Missing Dependencies" section
   │  → or workflows/server-init-ascend.md for full from-scratch setup
   │  (System packages like GCC/CMake/git-lfs are safe to install —
   │   they don't conflict with other users. Conda envs use
   │   <user>_py<ver> naming to avoid collisions. env.sh only
   │   affects the current shell session.)
   │
   └─ Only if you need a DIFFERENT toolchain version than the host
      → workflows/docker-build-ascend.md (rare, for version isolation)
│
Step 3: What build mode?
│
├─ Ascend (NPU)  → workflows/build-ascend.md
├─ GPU (CUDA)    → workflows/build-gpu.md
├─ CPU           → workflows/build-cpu.md
└─ C++ UT        → workflows/build-ut.md
```

## Auto-Detection

When the user or calling skill doesn't specify, follow the Decision Tree above.
Key: **hardware determines target** (Step 1), **software gaps are fixed** (Step 2).

Infer build intent from context:
- Caller is aclnn skill → Ascend
- User modified `tests/ut/cpp/` → UT build
- User modified `.py` test files → no build needed (Python tests run directly)
- User said "跑 ST" → production build (ST needs installed framework)

## Install Methods (source build is the only one for dev)

| Method | Use Case | Can compile local changes? |
|--------|----------|---------------------------|
| **Source build** | Edit code → compile → test | **Yes** — this skill |
| pip / conda / docker | Use released version only | No |

See `docs/install/mindspore_ascend_install_*.md` for pip/conda/docker details.

## Speed Optimization

Always prefer the fastest correct build:

| Situation | Strategy |
|-----------|----------|
| `build/` exists with matching config | Incremental (`-i`) — 5-10x faster |
| Only changed plugin code (ops/kernel/) | Plugin-only (`-f`) — skips core rebuild |
| First build or changed CMake config | Full build (no `-i`) |
| User explicitly asks for clean build | Full build + `-k on` |

## After Build Succeeds

Run the verification script (auto-detects device):
```
python <skill_dir>/scripts/verify_build.py --build-dir <repo>/build/package
```
Then: report build time, `.whl` location, installed version.
If called by another skill: return success + install path.
For dev iteration: use `PYTHONPATH=<repo>/build/package` (no pip install needed).

## On Build Failure

Check these causes in order:

1. **Missing CANN/CUDA toolkit** — check environment paths
2. **OOM during compilation** — reduce `-j` thread count
3. **Dependency download failure** — try with/without `-S on` (see note below)
4. **Submodule not initialized** — `git submodule update --init`
5. **git-lfs files not pulled** — build reaches 100% but CPack fails with `No such file or directory` for `custom_ascendc_910b` or `prebuild_ascendc.tar.gz` → `git lfs install && git lfs pull`
6. **Linker errors** (`cannot find -latomic`, `cannot find -lnuma`, `vtable undefined`) — missing system libs, conda lib not in path, or git-lfs not pulled → see workflow Troubleshooting
7. **tar ownership errors** in sandbox/container — add `--no-same-owner` to tar commands in `cmake/utils.cmake`, `cmake/external_libs/*.cmake`, and `scripts/build/check_and_build_ms_kernels_internal.sh`
8. **Stale CMake cache** — delete `build/` and retry full build
9. **Version mismatch** — see `workflows/version-matrix.md`

**Note on `-S on`**: Switches ~29/38 dependencies to Gitee mirrors. 7 deps have no
mirror and always need GitHub/GitLab. `-S on` also disables `$MSLIBS_SERVER` local cache.
If build previously worked without `-S on`, don't add it.

If the error doesn't match any pattern, show the last 50 lines of build output to the user.
