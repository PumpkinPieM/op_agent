# Build MindSpore for Ascend

Self-contained guide. No other files needed.

## Prerequisites

**If any check below fails, install the missing dependency. Do NOT skip Ascend build.**

### Quick Check (run all at once)

```bash
npu-smi info                    # NPU visible? (MUST pass on Ascend machine)
gcc --version                   # GCC 7.3+?
cmake --version                 # CMake 3.22.3+?
python --version                # Python 3.9-3.12?
git lfs version                 # git-lfs installed?
python -c "import numpy"       # numpy available?
ldconfig -p | grep libatomic    # libatomic available to linker?
rpm -q numactl-devel 2>/dev/null || dpkg -s libnuma-dev 2>/dev/null  # NUMA dev headers?
# CANN installed? (system-level or conda-level)
ls /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
ls /usr/local/Ascend/cann/set_env.sh 2>/dev/null || \
ls $(python -c "import sys;print(sys.prefix)" 2>/dev/null)/Ascend/cann/set_env.sh 2>/dev/null || \
echo "CANN NOT FOUND"
```

### Install Missing Dependencies

If everything passes, skip to [Environment Setup](#environment-setup).
Otherwise, install what's missing:

#### CANN Toolkit (Ascend-specific, REQUIRED)

NPU driver & firmware must be pre-installed by admin. CANN toolkit has two install methods:

**Method A: Conda install (recommended — no sudo needed)**

```bash
# Add Ascend channel (one-time)
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/

# Install toolkit
conda install ascend::cann-toolkit==8.5.0

# Install ops — choose ONE matching your chip:
conda install ascend::cann-910b-ops==8.5.0   # Atlas A2 (910B)
# conda install ascend::cann-a3-ops==8.5.0   # Atlas A3
# conda install ascend::cann-910-ops==8.5.0   # Atlas training series (910)
# conda install ascend::cann-310p-ops==8.5.0  # Atlas inference (310P)
# conda install ascend::cann-310b-ops==8.5.0  # Atlas 200I/500 A2 inference

# CANN installs into conda env's Ascend dir, e.g.:
#   <conda_env_path>/Ascend/cann/set_env.sh
```

After install, configure environment and install runtime deps:
```bash
source $(python -c "import sys;print(sys.prefix)")/Ascend/cann/set_env.sh
pip install attrs cython "numpy>=1.19.2,<2.0" decorator sympy cffi pyyaml \
    pathlib2 psutil "protobuf==3.20.0" scipy requests absl-py
```

Verify CANN works:
```bash
python -c "import acl;print(acl.get_soc_name())"
```

**Method B: Manual download (if conda channel not available)**

| Software | Version | Download |
|----------|---------|----------|
| CANN | 8.5.0 (recommended), 8.3.RC1, 8.2.RC1 | [CANN Community](https://www.hiascend.com/developer/download/community/result?module=cann) |
| Firmware & Driver | Matching CANN version | [Firmware & Driver](https://www.hiascend.com/hardware/firmware-drivers/community) |
| Install guide | — | [CANN Install Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html) |

Default install path: `/usr/local/Ascend`.

**After either method, verify:**
```bash
# Check system-level install
ls /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
ls /usr/local/Ascend/cann/set_env.sh 2>/dev/null || \
# Check conda-level install
ls $(conda info --base)/Ascend/cann/set_env.sh 2>/dev/null || \
ls $(python -c "import sys;print(sys.prefix)")/Ascend/cann/set_env.sh 2>/dev/null || \
echo "CANN NOT FOUND"
```

CANN Python dependencies (run after CANN install):
```bash
pip uninstall te topi hccl -y 2>/dev/null
pip install sympy protobuf attrs cloudpickle decorator ml-dtypes psutil scipy tornado jinja2
```

#### Python (via Conda)

```bash
# Install Miniconda (if no conda)
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_25.7.0-2-Linux-$(arch).sh
bash Miniconda3-py39_25.7.0-2-Linux-$(arch).sh -b
cd -
~/miniconda3/bin/conda init bash && source ~/.bashrc

# Create environment
# Naming: <user>_py<version>, e.g. alice_py39
conda create -c conda-forge -n alice_py39 python=3.9.11 -y
conda activate alice_py39
```

#### GCC (7.3.0+)

```bash
# Ubuntu / Debian
sudo apt-get install gcc-7 g++-7 -y

# CentOS 7
sudo yum install centos-release-scl && sudo yum install devtoolset-7
scl enable devtoolset-7 bash

# openEuler / EulerOS
sudo yum install gcc g++ -y
```

#### CMake (3.22.3+)

```bash
# x86_64
curl -O https://cmake.org/files/v3.22/cmake-3.22.3-linux-x86_64.sh
# aarch64
# curl -O https://cmake.org/files/v3.22/cmake-3.22.3-linux-aarch64.sh

sudo mkdir -p /usr/local/cmake-3.22.3
sudo bash cmake-3.22.3-linux-*.sh --prefix=/usr/local/cmake-3.22.3 --exclude-subdir
export PATH=/usr/local/cmake-3.22.3/bin:$PATH
```

#### git, tclsh, patch, NUMA, Flex, libatomic

```bash
# Ubuntu / Debian
sudo apt-get install -y git tcl patch libnuma-dev flex libatomic1

# CentOS / openEuler / EulerOS
sudo yum install -y git tcl patch numactl-devel flex libatomic
```

> **libatomic note**: On some systems (especially with conda-installed toolchains),
> `libatomic.so` exists in conda's lib dir but not in the system linker path.
> If you get `cannot find -latomic` during build, add conda's lib to the linker path:
> ```bash
> export LIBRARY_PATH=$(python -c "import sys;print(sys.prefix)")/lib:$LIBRARY_PATH
> export LD_LIBRARY_PATH=$(python -c "import sys;print(sys.prefix)")/lib:$LD_LIBRARY_PATH
> ```

#### git-lfs

```bash
# Ubuntu / Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs -y && git lfs install

# CentOS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs -y && git lfs install

# openEuler / EulerOS (manual)
# aarch64:
curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-arm64-v3.1.2.tar.gz
# x86_64:
# curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-amd64-v3.1.2.tar.gz
mkdir git-lfs && tar xf git-lfs-linux-*-v3.1.2.tar.gz -C git-lfs
cd git-lfs && sudo bash install.sh && cd ..
```

#### Python Build Packages

```bash
pip install wheel setuptools pyyaml "numpy>=1.20.0,<2.0.0" asttokens
```

### Environment Setup

**Option A**: Source a pre-configured `env_ms.sh` (if someone already set up the environment):

```bash
# Generic env script — takes MindSpore path as argument:
source /home/alice/env_ms.sh /home/alice/mindspore
source /home/alice/env_ms.sh /home/alice/test_ms/mindspore  # different source tree

# This activates conda, CANN, sets BUILD_PATH/PYTHONPATH/ccache,
# and prints verification output. Then build directly.
```

The `env_ms.sh` script auto-detects CANN (conda or system), configures ccache,
sets `BUILD_PATH` (needed by `runtest.sh`), and `PYTHONPATH` (for dev iteration
without pip install). See `server-init-ascend.md` Step 8 for how to create one.

**Option B**: Set up manually (if no env.sh exists yet, or you want your own):

```bash
LOCAL_ASCEND=/usr/local/Ascend
CONDA_PREFIX="$(python -c 'import sys;print(sys.prefix)' 2>/dev/null)"

# Source CANN environment (system-level or conda-level)
for p in ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh \
         ${LOCAL_ASCEND}/cann/set_env.sh \
         ${CONDA_PREFIX}/Ascend/cann/set_env.sh \
         ${CONDA_PREFIX}/Ascend/ascend-toolkit/set_env.sh; do
    [ -f "$p" ] && source "$p" && break
done

export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}
export GLOG_v=2    # log level: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# libatomic / conda lib (needed for linking)
export LIBRARY_PATH=${CONDA_PREFIX}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

# GCC version selection (session-only, do NOT change system symlinks)
# CentOS devtoolset-7:
[ -f "/opt/rh/devtoolset-7/enable" ] && source /opt/rh/devtoolset-7/enable
# Ubuntu with gcc-7 alongside other versions:
# export CC=/usr/bin/gcc-7 CXX=/usr/bin/g++-7
```

To create your own `env.sh`, see `server-init-ascend.md` Step 8.

### Final Verification

```bash
npu-smi info                                   # NPU visible
gcc --version | head -1                        # GCC 7.3+
cmake --version | head -1                      # CMake 3.22.3+
python --version                               # Python 3.9-3.12
echo $ASCEND_HOME                              # set by CANN set_env.sh
```

All must pass before proceeding to build.

## Build Command

```bash
cd <mindspore-repo-root>
bash build.sh -e ascend [-V <version>] -j<threads>
```

### Parameter Reference

| Param | Meaning | Values | Default |
|-------|---------|--------|---------|
| `-e ascend` | Target device | `ascend` (or legacy `d`) | — |
| `-V` | Chip version | `910`, `910b`, `a5`, `310` | `910` |
| `-j` | Parallel threads | integer | `8` |
| `-i` | Incremental build (skip cmake config) | flag | off |
| `-f` | Plugin-only build (skip core) | flag | off |
| `-d` | Debug mode | flag | off (Release) |
| `-S on` | Gitee mirror for ~29/38 deps; 7 deps still need GitHub; disables `$MSLIBS_SERVER` (see details below) | `on`/`off` | `off` |
| `-k on` | Clean before build | `on`/`off` | `off` |
| `-p on` | Enable profiling (ENABLE_PROFILE) | `on`/`off` | `off` |
| `-o` | Enable AIO (async I/O, ENABLE_AIO) | flag | off |

### About `-S on` (ENABLE_GITEE)

`-S on` switches ~29 of 38 external dependencies from GitHub to Gitee mirrors.
**It is NOT a universal fix.** Key caveats:

- **7 deps have no gitee mirror** (sqlite, mockcpp, eigen, pocketfft, jemalloc, ffmpeg, limonp)
  — these still download from GitHub/GitLab regardless of `-S on`
- **Disables `MSLIBS_SERVER`**: if your build machine has a local dependency cache
  (via `$MSLIBS_SERVER` env var), `-S on` will bypass it and go to public Gitee instead
- **Gitee mirror URLs may be stale**: the `gitee.com/mirrors/xxx/repository/archive/` format
  is an older Gitee API that may break without notice
- **If build fails with `-S on` but succeeds without it** (or vice versa), the issue is
  likely a specific gitee mirror being down or having a different archive format

**Recommendation**: Try building without `-S on` first. Only add it if GitHub downloads
fail due to network restrictions. If your company has `MSLIBS_SERVER`, do NOT use `-S on`.

### Chip Version Details

`-V` controls which features get compiled. The actual compile flags:

| Version | ENABLE_D | ENABLE_ACL | ENABLE_AKG | ENABLE_CPU | ENABLE_MPI | Notes |
|---------|----------|------------|------------|------------|------------|-------|
| `910` | on | on | — | on | on | Default. For Atlas 900/800 A1 |
| `910b` | on | on | — | on | on | Same flags as 910. For Atlas 800 A2 |
| `a5` | on | on | — | on | on | Same + EXPERIMENT_A5. Atlas A5 |
| `310` | on | on | **on** | on | on | Inference card. Adds AKG |

Key insight: **910 and 910b produce identical compile flags.** The difference is
the `ASCEND_VERSION` CMake variable, which affects runtime library linking.
If you're on a 910b machine, `-V 910b` is recommended but omitting `-V` also works.

## Use tmux (prevent SSH disconnect from killing the build)

Full builds take 30-60 min. Always run inside tmux to survive SSH disconnects:

```bash
tmux new -s build              # start a named session
# ... run build commands below ...

# If disconnected, reconnect with:
tmux attach -t build
```

If tmux is not installed: `sudo yum install tmux -y` (openEuler/CentOS)
or `sudo apt install tmux -y` (Ubuntu/Debian).

## Common Recipes

### Full build on 910b (most common for aclnn dev)
```bash
bash build.sh -e ascend -V 910b -j128
```

> If GitHub is blocked on your network, try adding `-S on` for Gitee mirrors.
> But see the `-S on` caveat above — it doesn't cover all deps and may conflict
> with local cache servers.

### Incremental build (after first full build, 5-10x faster)
```bash
bash build.sh -e ascend -V 910b -j128 -i
```

### Plugin-only build (only changed ops/kernel code, fastest)
```bash
bash build.sh -e ascend -V 910b -j128 -f
```

### Debug build
```bash
bash build.sh -e ascend -V 910b -j128 -d
```

### Clean rebuild (when cmake cache is stale)
```bash
bash build.sh -e ascend -V 910b -j128 -k on
```

## Choosing the Right Recipe

```
Have you built before with the same -V and -e?
│
├─ No → Full build
│
└─ Yes
   │
   ├─ Changed CMakeLists.txt or build scripts? → Full build
   ├─ Changed only files under ops/kernel/ ? → Plugin-only (-f)
   └─ Changed other C++ files? → Incremental (-i)
```

## Thread Count Guideline

| Machine RAM | Recommended -j |
|-------------|----------------|
| 64 GB | 16–32 |
| 128 GB | 32–64 |
| 256 GB+ | 64–128 |

Each compile thread can use 2-4 GB RAM. If build is killed by OOM, halve the thread count.

## Build Acceleration

### ccache (2-5x faster for repeated builds)

ccache caches gcc/g++ compilation results. The CI uses ccache with PATH wrappers
and achieves ~50G cache with high hit rates.

```bash
# Install
sudo yum install ccache -y    # openEuler / CentOS / EulerOS
# sudo apt install ccache -y  # Ubuntu / Debian
```

**Enable via PATH** (recommended — same as CI, doesn't interfere with CMake):
```bash
# ccache creates symlinks: gcc -> ccache, g++ -> ccache
# Put ccache wrapper dir BEFORE real gcc in PATH
CCACHE_WRAPPER_DIR=$(dirname $(which ccache))  # usually /usr/lib64/ccache or /usr/local/ccache/bin
export PATH="${CCACHE_WRAPPER_DIR}:$PATH"
```

**ccache config** (in env.sh, referencing CI best practices):
```bash
export CCACHE_DIR=~/mindspore_dev/.ccache
export CCACHE_MAXSIZE="50G"
export CCACHE_COMPRESS=1
export CCACHE_COMPRESSLEVEL=1
export CCACHE_SLOPPINESS="include_file_ctime,time_macros"
export CCACHE_NOHASHDIR=1          # don't hash build dir path → cache survives dir renames
```

First full build populates the cache. Subsequent builds (including `git checkout`
to a different branch) reuse cached objects. Check: `ccache -s`

> If ccache wrapper dir doesn't exist, create it:
> ```bash
> sudo mkdir -p /usr/local/ccache/bin
> sudo ln -s $(which ccache) /usr/local/ccache/bin/gcc
> sudo ln -s $(which ccache) /usr/local/ccache/bin/g++
> sudo ln -s $(which ccache) /usr/local/ccache/bin/cc
> sudo ln -s $(which ccache) /usr/local/ccache/bin/c++
> ```

### MSLIBS_CACHE_PATH (avoid re-downloading dependencies)

```bash
export MSLIBS_CACHE_PATH=~/mindspore_dev/.mslib
```

Third-party dependencies are cached here. Shared across branches and rebuilds.

> **Cross-directory trap**: CMake writes absolute paths from `MSLIBS_CACHE_PATH` into
> `cmake_install.cmake`. If you set a shared `MSLIBS_CACHE_PATH` (e.g. `/home/alice/dev/.mslib`)
> but build in a different source tree, the CPack packaging step may fail with
> `file INSTALL cannot find ".../robin_hood_hashing_.../src/include"`.
> **Fix**: either `unset MSLIBS_CACHE_PATH` (let each build use its own `build/.mslib`),
> or `rm -rf build` and rebuild from scratch when switching source directories.

### git-lfs (REQUIRED for Ascend builds)

MindSpore stores large binary files (AscendC prebuild kernels, internal kernels) via git-lfs.
**After cloning, you MUST pull lfs files**, otherwise the build will compile to 100% but
fail at the CPack packaging step with `custom_ascendc_910b: No such file or directory`.

```bash
cd <mindspore-repo-root>
git lfs install
git lfs pull

# Verify lfs files are real binaries, not pointer stubs:
file mindspore/ops/kernel/ascend/ascendc/prebuild/$(arch)/prebuild_ascendc.tar.gz
# Expected: "gzip compressed data"
# Bad:      "ASCII text" (means lfs pointer, not actual file)
```

## Build Output & Installation

On success, the script:
1. Compiles into `build/mindspore/`
2. Packages into `build/package/mindspore/` (complete Python package + native libs)
3. Generates `.whl` in `output/`
4. Prints "success building mindspore project!"

### Option A: PYTHONPATH (fastest for dev iteration, no install step)
```bash
export PYTHONPATH=<repo>/build/package:$PYTHONPATH
export LD_LIBRARY_PATH=<repo>/build/package/mindspore/lib:$LD_LIBRARY_PATH
```

Rebuild → immediately available. No `pip install` needed between edits.
Add these to env.sh for persistence across sessions.

### Option B: Install the whl package (for ST testing or sharing)
```bash
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/
```

Needed when running ST tests that `import mindspore` from site-packages.

### Verify Installation

**Quick check** (CPU-only, works even without NPU access):
```bash
python -c "import mindspore as ms; print('MindSpore', ms.__version__); \
import numpy as np; x=ms.Tensor(np.ones([2,3],dtype=np.float32)); print('Tensor OK:', x.shape)"
```

**Full NPU check** (requires NPU device access):
```bash
python -c "
import mindspore as ms
print('MindSpore version:', ms.__version__)
ms.set_context(device_target='Ascend')
print('Device target:', ms.get_context('device_target'))
import mindspore.nn as nn
import numpy as np
net = nn.Dense(10, 5)
x = ms.Tensor(np.random.randn(2, 10).astype(np.float32))
out = net(x)
print('Dense output shape:', out.shape)
print('Ascend NPU test PASSED')
"
```

> **Note**: `ms.set_context(device_target='Ascend')` works but is deprecated in 2.9+.
> Use `ms.set_device('Ascend')` for newer versions. If running in a sandbox/container
> without `/dev/davinci*` device mounts, the NPU test will fail with `ErrCode=507899`
> — this is expected; test on the host or a properly configured container instead.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ascend-toolkit not found` | CANN not installed or env not sourced | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `Killed` or OOM | Too many threads | Reduce `-j` |
| CMake error about missing package | Dependency download failed | Check network; try with/without `-S on`; check `$MSLIBS_SERVER` |
| `fatal: not a git repository` in submodule | Submodule not init | `git submodule update --init` |
| Linker errors after switching branch | Stale build cache | Delete `build/` dir, full rebuild |
| `ENABLE_D` related errors | Wrong `-V` for this machine | Check `npu-smi info` for actual chip |
| `tar: Cannot change ownership` (isl, mkl_dnn, ms_kernels_internal, etc.) | Sandbox/container restricts `chown` | Add `--no-same-owner` to tar commands in `cmake/utils.cmake`, `cmake/external_libs/*.cmake`, and `scripts/build/check_and_build_ms_kernels_internal.sh` |
| `cannot find -latomic` | libatomic not in linker path | `export LIBRARY_PATH=$(python -c "import sys;print(sys.prefix)")/lib:$LIBRARY_PATH` |
| `Numa package not found` | numactl-devel not installed | `sudo yum install numactl-devel` (CentOS/openEuler) or `sudo apt install libnuma-dev` |
| `No module named 'asttokens'` | Python dep missing | `pip install asttokens` |
| sqlite download hangs/fails | sqlite has no Gitee mirror, GitHub blocked | Copy cached `.tar.gz` to `$MSLIBS_CACHE_PATH`; or use proxy |
| `robin_hood_hashing.../src/include: No such file` | `MSLIBS_CACHE_PATH` points to a different build tree | `unset MSLIBS_CACHE_PATH` then `rm -rf build` and rebuild |
| `custom_ascendc_910b: No such file` at CPack | git-lfs files not pulled | `git lfs install && git lfs pull` then rebuild with `-i` |
| `kGetLcocWorkspaceSizeName undeclared` in hcom_matmul | Source bug: LCOC code missing `#ifdef ENABLE_INTERNAL_KERNELS` | Add `#ifdef ENABLE_INTERNAL_KERNELS` guards around LCOC code blocks in `hcom_matmul_all_reduce.cc` |
| `rtGetDeviceCount: ErrCode=507899` | NPU device not accessible (Docker without device mount, or driver issue) | Check `npu-smi info`; in Docker add `--device=/dev/davinci*` flags |

## What Happens Inside (for understanding)

`build.sh` is a thin orchestrator that sources 6 sub-scripts:

1. `scripts/build/default_options.sh` — sets all flags to defaults
2. `scripts/build/process_options.sh` — parses CLI args into env vars
3. `scripts/build/parse_device.sh` — translates `-e ascend -V 910b` into specific ENABLE_* flags
4. `scripts/build/build_mindspore.sh` — assembles CMAKE_ARGS and runs cmake + cmake --build

The actual build is two cmake commands:
```bash
cmake ${CMAKE_ARGS} ${BASEPATH}          # Configure (skipped with -i)
cmake --build . --target package -j128   # Compile
```
