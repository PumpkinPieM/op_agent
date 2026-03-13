# Ascend Server Initialization (from scratch to build-ready)

Self-contained guide. Follow steps in order on a fresh Ascend server.

> **Shared machine?** This guide is safe for shared machines. System packages
> (GCC, CMake, git-lfs) are shared tools that benefit all users. Conda environments
> use `<user>_py<ver>` naming to avoid collisions. The generated `env.sh` only
> affects the current shell session — other users are not impacted.
> Docker (`docker-build-ascend.md`) is only needed if you require a different
> toolchain version than what's installed on the host.

## Step 0: Gather Environment Info

Run these first to understand what you're working with:

```bash
# OS and architecture
cat /etc/os-release
uname -m                    # x86_64 or aarch64

# NPU hardware
npu-smi info                # shows NPU model, driver version, health
cat /usr/local/Ascend/version.info 2>/dev/null   # CANN version if pre-installed

# Existing tools
gcc --version 2>/dev/null
cmake --version 2>/dev/null
python3 --version 2>/dev/null
git --version 2>/dev/null
conda --version 2>/dev/null

# Disk space (source + build + deps ≈ 20-30GB)
df -h .

# Memory (each -j thread uses 2-4GB)
free -h
nproc                       # available CPU cores
```

Record the results. You'll need them to choose the right install commands below.

## Step 1: Create Project Directory

```bash
mkdir -p ~/mindspore_dev
cd ~/mindspore_dev
```

## Step 2: Install System Dependencies

### Debian/Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y \
    gcc-7 g++-7 \
    git tcl patch libnuma-dev flex libatomic1 \
    curl wget

# Use GCC 7 via PATH in env.sh (do NOT use update-alternatives on shared machines)
# env.sh will set: export PATH=/usr/bin/gcc-7-dir:$PATH  (or CC/CXX variables)
```

### CentOS 7

```bash
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-7
sudo yum install -y git tcl patch numactl-devel flex libatomic curl wget

# devtoolset-7 installs to /opt/rh/devtoolset-7/
# Activate via env.sh (do NOT use 'scl enable' in global bashrc)
```

### openEuler / EulerOS

```bash
sudo yum install -y gcc g++ git tcl patch numactl-devel flex libatomic curl wget
```

### Verify

```bash
gcc --version     # expect 7.3.0+
g++ --version
```

> **Important**: If the system has multiple GCC versions, do NOT change the default
> with `update-alternatives` or symlinks. Instead, set `CC` and `CXX` in env.sh
> (see Step 8). This only affects your build session, not other users.

## Step 3: Install git-lfs

### Debian/Ubuntu

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs -y
git lfs install
```

### CentOS 7

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs -y
git lfs install
```

### openEuler / EulerOS (manual install)

```bash
# x86_64
curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-amd64-v3.1.2.tar.gz
# aarch64
# curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-arm64-v3.1.2.tar.gz

mkdir git-lfs && tar xf git-lfs-linux-*-v3.1.2.tar.gz -C git-lfs
cd git-lfs && sudo bash install.sh && cd ..
```

### Verify

```bash
git lfs version
```

## Step 4: Install CMake (3.22.3+)

Check if already installed:
```bash
cmake --version
```

If missing or too old:

```bash
# x86_64
curl -O https://cmake.org/files/v3.22/cmake-3.22.3-linux-x86_64.sh
# aarch64
# curl -O https://cmake.org/files/v3.22/cmake-3.22.3-linux-aarch64.sh

sudo mkdir -p /usr/local/cmake-3.22.3
sudo bash cmake-3.22.3-linux-*.sh --prefix=/usr/local/cmake-3.22.3 --exclude-subdir

echo 'export PATH=/usr/local/cmake-3.22.3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify

```bash
cmake --version    # expect 3.22.3+
```

## Step 5: Install / Verify CANN

NPU driver & firmware must be pre-installed by admin (`npu-smi info` must work).
CANN toolkit itself has two install methods:

### Check if already installed

```bash
npu-smi info    # driver OK?

# System-level CANN (admin-installed)
ls /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && echo "system ascend-toolkit found"
ls /usr/local/Ascend/cann/set_env.sh 2>/dev/null && echo "system cann found"
cat /usr/local/Ascend/version.info 2>/dev/null
```

If CANN is already installed, skip to Step 6.

### Install via Conda (recommended — no sudo needed)

```bash
# Add Ascend channel (one-time)
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/

# Install toolkit + ops for your chip (run AFTER Step 6 conda env is created)
conda activate <your_env>
conda install ascend::cann-toolkit==8.5.0
# Choose ONE ops package matching your chip:
conda install ascend::cann-910b-ops==8.5.0   # Atlas A2 (910B)
# conda install ascend::cann-a3-ops==8.5.0   # Atlas A3
# conda install ascend::cann-910-ops==8.5.0   # Atlas training series (910)
# conda install ascend::cann-310p-ops==8.5.0  # Atlas inference (310P)
# conda install ascend::cann-310b-ops==8.5.0  # Atlas 200I/500 A2 inference
```

After install, configure and install runtime deps:
```bash
source $(python -c "import sys;print(sys.prefix)")/Ascend/cann/set_env.sh
pip install attrs cython "numpy>=1.19.2,<2.0" decorator sympy cffi pyyaml \
    pathlib2 psutil "protobuf==3.20.0" scipy requests absl-py
```

Verify:
```bash
python -c "import acl;print(acl.get_soc_name())"
```

CANN installs into the conda env's Ascend dir:
```bash
# Verify — path is like <conda_env>/Ascend/cann/set_env.sh
CONDA_PREFIX=$(python -c "import sys;print(sys.prefix)")
ls ${CONDA_PREFIX}/Ascend/cann/set_env.sh && echo "conda CANN found"
```

### Install via manual download (alternative)

Download from [CANN Community](https://www.hiascend.com/developer/download/community/result?module=cann).
Recommended version: **8.5.0**. Follow the
[install guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html).
Default path: `/usr/local/Ascend`.

## Step 6: Install Conda and Create Python Environment

### Install Miniconda

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_25.7.0-2-Linux-$(arch).sh
bash Miniconda3-py39_25.7.0-2-Linux-$(arch).sh -b
cd -

# Initialize conda
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### (Optional) Set Tsinghua mirror for faster downloads

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
```

### Create environment

```bash
# Naming convention: <user>_py<version>, e.g. alice_py39
conda create -c conda-forge -n alice_py39 python=3.9.11 -y
conda activate alice_py39
```

### Verify

```bash
python --version          # expect 3.9.11
which python              # should point to conda env
```

## Step 7: Install Python Dependencies

```bash
conda activate alice_py39

# Build tools
pip install wheel setuptools pyyaml "numpy>=1.20.0,<2.0.0" asttokens

# CANN Python dependencies (uninstall stale versions first)
pip uninstall te topi hccl -y 2>/dev/null
pip install sympy protobuf attrs cloudpickle decorator \
    ml-dtypes psutil scipy tornado jinja2
```

### Verify

```bash
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import yaml; print('pyyaml OK')"
```

## Step 8: Generate `env.sh`

Create a single `env.sh` in your project directory. This script is the **shared entry point**:
- You验证编译通过后，任何人只需 `source /home/alice/mindspore_dev/env.sh` 就能获得
  完整的编译环境（conda + CANN + CMake），然后 `cd` 到任意 MindSpore 源码目录直接编译。
- 不 source 则完全不影响当前 shell，不污染其他人的环境。
- 所有路径用**绝对路径硬编码**（不依赖 `$HOME`），确保任何用户 source 都指向同一套环境。

### 8a. Detect paths on this machine

```bash
# Find Conda base
CONDA_BASE="$(conda info --base 2>/dev/null)"
echo "Conda base: ${CONDA_BASE:-NOT FOUND}"

# Find CANN set_env.sh (system-level OR conda-level)
CONDA_PREFIX="$(python -c 'import sys;print(sys.prefix)' 2>/dev/null)"
CANN_ENV=""
for p in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/cann/set_env.sh \
         ${CONDA_PREFIX}/Ascend/cann/set_env.sh \
         ${CONDA_PREFIX}/Ascend/ascend-toolkit/set_env.sh; do
    [ -f "$p" ] && CANN_ENV="$p" && break
done
echo "CANN env script: ${CANN_ENV:-NOT FOUND}"
```

If CANN not found → go back to Step 5 (install via conda or manual download).
If Conda not found → go back to Step 6.

### 8b. Write `env.sh`

**Replace the paths below with the actual values from 8a.**
The key design: all paths are absolute, not relative to `$HOME`.

```bash
# ── Replace these 4 values with your actual paths ──
MY_DIR="/home/alice/mindspore_dev"        # your project directory (absolute)
CONDA_BASE_DIR="/home/alice/miniconda3"   # conda install location (absolute)
CONDA_ENV_NAME="alice_py39"                # naming: <user>_py<version>
OWNER="alice"                              # who set this up (for identification)
# ────────────────────────────────────────────────────

cat > ${MY_DIR}/env.sh << ENVEOF
#!/bin/bash
# MindSpore build environment — configured by ${OWNER}
# Usage: source ${MY_DIR}/env.sh
# Then cd to ANY MindSpore source directory and run: bash build.sh -e ascend ...
#
# This script only modifies the CURRENT shell session.
# Close the terminal or open a new one to return to a clean environment.

# ── Conda (use ${OWNER}'s verified environment) ──────────
_CONDA_SH="${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
if [ -f "\$_CONDA_SH" ]; then
    source "\$_CONDA_SH"
    conda activate ${CONDA_ENV_NAME}
else
    echo "[env.sh] ERROR: conda not found at \$_CONDA_SH"
    echo "[env.sh] This env.sh was set up by ${OWNER}. Contact them if paths changed."
    return 1
fi

# ── CANN / Ascend ────────────────────────────────────────
LOCAL_ASCEND=/usr/local/Ascend
export GLOG_v=2

# Auto-detect CANN: system-level (/usr/local/Ascend) OR conda-level
_CONDA_PREFIX="\$(python -c 'import sys;print(sys.prefix)' 2>/dev/null)"
_CANN_SOURCED=0
for _p in "\${LOCAL_ASCEND}/ascend-toolkit/set_env.sh" \
          "\${LOCAL_ASCEND}/cann/set_env.sh" \
          "\${_CONDA_PREFIX}/Ascend/cann/set_env.sh" \
          "\${_CONDA_PREFIX}/Ascend/ascend-toolkit/set_env.sh"; do
    if [ -f "\$_p" ]; then
        source "\$_p"
        _CANN_SOURCED=1
        _CANN_ENV_PATH="\$_p"
        break
    fi
done
if [ "\$_CANN_SOURCED" -eq 0 ]; then
    echo "[env.sh] WARNING: CANN set_env.sh not found (checked \$LOCAL_ASCEND and \$_CONDA_PREFIX/Ascend)"
fi
if [ "\$_CANN_SOURCED" -eq 1 ]; then
    export ASCEND_CUSTOM_PATH="\$(dirname "\$(dirname "\$(dirname "\$_CANN_ENV_PATH")")")"
else
    export ASCEND_CUSTOM_PATH=\${LOCAL_ASCEND}
fi
unset _CANN_SOURCED _p _CONDA_SH _CONDA_PREFIX _CANN_ENV_PATH

# ── libatomic (conda toolchains may have it outside system path) ──
_CONDA_LIB="\$(python -c 'import sys;print(sys.prefix)' 2>/dev/null)/lib"
if [ -d "\$_CONDA_LIB" ]; then
    export LIBRARY_PATH="\$_CONDA_LIB:\$LIBRARY_PATH"
    export LD_LIBRARY_PATH="\$_CONDA_LIB:\$LD_LIBRARY_PATH"
fi
unset _CONDA_LIB

# ── CMake (if installed to custom path) ──────────────────
if [ -d "/usr/local/cmake-3.22.3/bin" ]; then
    export PATH=/usr/local/cmake-3.22.3/bin:\$PATH
fi

# ── GCC version selection ─────────────────────────────────
# CentOS/EulerOS devtoolset-7:
if [ -f "/opt/rh/devtoolset-7/enable" ]; then
    source /opt/rh/devtoolset-7/enable
# Ubuntu/Debian with gcc-7 alongside other versions:
elif [ -x "/usr/bin/gcc-7" ]; then
    export CC="/usr/bin/gcc-7"
    export CXX="/usr/bin/g++-7"
fi

# ── ccache (PATH wrapper, same approach as CI) ───────────
# ccache wraps gcc/g++ to cache compilation results (2-5x faster on rebuilds).
# CI uses PATH-based wrappers: a directory with gcc/g++ symlinks pointing to ccache
# is prepended to PATH, so CMake sees "gcc" but actually runs ccache transparently.
if command -v ccache &>/dev/null; then
    # Prefer distro ccache wrapper dir if it exists
    for _d in /usr/local/ccache/bin /usr/lib64/ccache /usr/lib/ccache; do
        if [ -d "\$_d" ] && [ -L "\$_d/gcc" ] 2>/dev/null; then
            export PATH="\$_d:\$PATH"
            break
        fi
    done
    # If no wrapper dir found, create one under user home
    if ! ccache -p 2>/dev/null | grep -q 'compiler'; then
        _WRAPPER="${MY_DIR}/.ccache_wrappers"
        if [ ! -d "\$_WRAPPER" ]; then
            mkdir -p "\$_WRAPPER"
            ln -sf "\$(which ccache)" "\$_WRAPPER/gcc"
            ln -sf "\$(which ccache)" "\$_WRAPPER/g++"
            ln -sf "\$(which ccache)" "\$_WRAPPER/cc"
            ln -sf "\$(which ccache)" "\$_WRAPPER/c++"
        fi
        export PATH="\$_WRAPPER:\$PATH"
        unset _WRAPPER
    fi
    # ccache config (referencing CI best practices)
    export CCACHE_DIR="${MY_DIR}/.ccache"
    export CCACHE_MAXSIZE="50G"
    export CCACHE_COMPRESS=1
    export CCACHE_COMPRESSLEVEL=1
    export CCACHE_SLOPPINESS="include_file_ctime,time_macros"
    export CCACHE_NOHASHDIR=1
fi

# ── mold linker (optional, significantly faster linking) ──
# mold is a high-speed drop-in replacement for ld. If installed, prepend to PATH.
if [ -d "/usr/local/mold/bin" ]; then
    export PATH="/usr/local/mold/bin:\$PATH"
fi

# ── PYTHONPATH (use build output without pip install) ────
# After build succeeds, set this to skip 'pip install output/*.whl':
#   export PYTHONPATH=<ms_source>/build/package:\$PYTHONPATH
#   export LD_LIBRARY_PATH=<ms_source>/build/package/mindspore/lib:\$LD_LIBRARY_PATH
# Uncomment and set the path after your first successful build:
# export PYTHONPATH=${MY_DIR}/mindspore/build/package:\$PYTHONPATH
# export LD_LIBRARY_PATH=${MY_DIR}/mindspore/build/package/mindspore/lib:\$LD_LIBRARY_PATH

# ── Ready ────────────────────────────────────────────────
echo "[env.sh] Environment ready (configured by ${OWNER})"
echo "  Python : \$(python --version 2>&1)"
echo "  CMake  : \$(cmake --version 2>&1 | head -1)"
echo "  GCC    : \$(gcc --version 2>&1 | head -1)"
echo "  CANN   : \${ASCEND_HOME:-not set}"
echo "  ccache : \$(ccache --version 2>&1 | head -1 || echo 'not installed')"
command -v mold &>/dev/null && echo "  mold   : \$(mold --version 2>&1)"
echo ""
echo "Now cd to your MindSpore source dir and run:"
echo "  bash build.sh -e ascend -V 910b -j64"
ENVEOF

chmod +x ${MY_DIR}/env.sh
echo "env.sh written to ${MY_DIR}/env.sh"
```

### 8c. Verify

```bash
source /home/alice/mindspore_dev/env.sh

# Should see:
# [env.sh] Environment ready (configured by alice)
#   Python : Python 3.9.11
#   CMake  : cmake version 3.22.3
#   GCC    : gcc (GCC) ...
#   CANN   : /usr/local/Ascend/ascend-toolkit/...

python --version          # expect 3.9.11
cmake --version           # expect 3.22.3+
echo $ASCEND_CUSTOM_PATH  # expect /usr/local/Ascend
npu-smi info              # NPU accessible
```

### 8d. How others use your env.sh

After you verify the build works, tell your teammates:

```bash
# One command to get a working build environment:
source /home/alice/mindspore_dev/env.sh

# Then go to your own MindSpore source directory:
cd /home/someone_else/mindspore
bash build.sh -e ascend -V 910b -j64
```

Key properties:
- **No install needed**: teammates don't need to install conda, cmake, or python deps
- **No pollution**: only affects the current shell session; close terminal = clean state
- **Any source dir**: works with any MindSpore clone, not tied to your directory
- **Identified**: output shows who configured this env, so people know who to ask

### 8e. (Optional) Auto-source on login

Only do this for YOUR login, not for shared accounts like root:

```bash
echo '[ -f /home/alice/mindspore_dev/env.sh ] && source /home/alice/mindspore_dev/env.sh' >> ~/.bashrc
```

**WARNING**: On shared root accounts, do NOT add this to `/root/.bashrc` — it would
affect everyone. Each person should source manually or add it to their own shell init.

## Step 9: Clone MindSpore Source

```bash
cd ~/mindspore_dev
git clone https://atomgit.com/mindspore/mindspore.git
cd mindspore
git submodule update --init

# CRITICAL: Pull git-lfs files (AscendC prebuild kernels, internal kernels)
# Without this, build compiles to 100% but fails at CPack packaging.
git lfs install
git lfs pull
```

If the server has restricted network access:
- Clone on an external machine first, then rsync/scp the repo
- Or set up a git mirror

### Verify

```bash
ls build.sh                # should exist
git log --oneline -3       # recent commits visible

# Verify lfs files are real binaries, not pointer stubs
file mindspore/ops/kernel/ascend/ascendc/prebuild/$(arch)/prebuild_ascendc.tar.gz
# Expected: "gzip compressed data"
# If "ASCII text" → lfs pointer, run: git lfs pull
```

## Step 10: Test Build

Full builds take 30-60 min. Use tmux to survive SSH disconnects:

```bash
# Install tmux if not available
# openEuler/CentOS: sudo yum install tmux -y
# Ubuntu/Debian:    sudo apt install tmux -y

tmux new -s build
source ~/mindspore_dev/env.sh
cd ~/mindspore_dev/mindspore

# Full build (adjust -j based on your RAM, see thread count guideline below)
bash build.sh -e ascend -V 910b -j64

# If SSH disconnects, reconnect with: tmux attach -t build
```

### Thread Count Guideline

| Machine RAM | Recommended -j |
|-------------|----------------|
| 64 GB | 16-32 |
| 128 GB | 32-64 |
| 256 GB+ | 64-128 |

Each thread uses 2-4 GB RAM. If build is killed by OOM, halve the count.

### On Success

```bash
# Install
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/

# Verify
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

Expected output:
```
MindSpore version: x.x.x
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

## Full Checklist

Use this to verify all steps are complete:

```
[ ] Step 0: Environment info recorded (OS, arch, NPU model, disk, RAM)
[ ] Step 1: Project directory created
[ ] Step 2: GCC 7.3+ installed
[ ] Step 3: git-lfs installed
[ ] Step 4: CMake 3.22.3+ installed
[ ] Step 5: CANN installed and NPU visible
[ ] Step 6: Conda installed, '<user>_py<ver>' environment created (e.g. alice_py39)
[ ] Step 7: Python deps installed (numpy, pyyaml, wheel, CANN deps)
[ ] Step 8: env.sh generated, source it works (Python + CMake + CANN all ready)
[ ] Step 9: MindSpore repo cloned, submodules initialized
[ ] Step 10: Build succeeded, run_check() passed
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `npu-smi: command not found` | CANN driver not installed | Contact server admin |
| `conda: command not found` | Conda not in PATH | Run `source ~/.bashrc` or re-install |
| `gcc-7: No such file` | Wrong package name for your OS | See Step 2 for OS-specific commands |
| CMake download fails | Network restricted | Download on external machine, scp to server |
| `pip install` timeout | No PyPI mirror | Add `-i https://repo.huaweicloud.com/repository/pypi/simple/` |
| Build OOM (Killed) | Too many -j threads | Halve the -j number |
| `set_env.sh: No such file` | CANN path differs | Run `find /usr/local/Ascend -name set_env.sh` |
| `Permission denied` on `/usr/local/Ascend` | User not in Ascend group | `sudo usermod -aG HwHiAiUser $USER` then re-login |
| `cannot find -latomic` | libatomic not in linker path | `export LIBRARY_PATH=$(python -c "import sys;print(sys.prefix)")/lib:$LIBRARY_PATH` |
| `custom_ascendc_910b: No such file` at CPack | git-lfs files not pulled | `git lfs install && git lfs pull` |
| `robin_hood_hashing.../src/include: No such file` | MSLIBS_CACHE_PATH cross-directory issue | `unset MSLIBS_CACHE_PATH && rm -rf build` then rebuild |
| sqlite download hangs | No Gitee mirror for sqlite | Copy cached file to `$MSLIBS_CACHE_PATH` or use proxy |
