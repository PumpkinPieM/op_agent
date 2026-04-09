# Build MindSpore for CPU

> **Status: NOT YET VERIFIED.** This workflow is based on official docs but has not been
> tested end-to-end. Contributions welcome.

Self-contained guide. No other files needed.

## Prerequisites

### Linux

| Software | Version | Install (Ubuntu 18.04) |
|----------|---------|----------------------|
| Python | 3.9–3.12 | `conda create -n <user>_py39 python=3.9.11 -y` |
| GCC | 7.3.0–9.4.0 | `sudo apt-get install gcc-7 -y` |
| CMake | 3.22.3+ | See [kitware APT](https://apt.kitware.com/) |
| git, tclsh, patch, NUMA | — | `sudo apt-get install git tcl patch libnuma-dev -y` |
| wheel, setuptools, PyYAML, numpy | — | `pip install wheel setuptools pyyaml "numpy>=1.20.0,<2.0.0"` |
| LLVM 12 (optional, for graph-kernel fusion) | 12.0.1 | See install docs |

### macOS

| Software | Version | Install |
|----------|---------|---------|
| Xcode | 12.4+ (Intel), 13.0+ (M1) | App Store |
| Command Line Tools | — | `sudo xcode-select --install` |
| CMake | 3.22.3+ | `brew install cmake` |
| patch, autoconf | — | `brew install patch autoconf` |
| Python | 3.9–3.12 | Conda (Miniforge for M1, Miniconda for Intel) |

macOS requires setting the compiler before build:
```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
```

### Windows

| Software | Version |
|----------|---------|
| Windows 10 | x86_64 |
| Visual Studio 2019 Community | With "Desktop C++" and "Universal Windows Platform" workloads |
| CMake | 3.22.3+ (add to PATH) |
| Python | 3.9+ |
| git | Add `Git\usr\bin` to PATH |
| [MSYS2](https://www.msys2.org/) | — |

Windows uses `build.bat` instead of `build.sh`:
```cmd
call build.bat ms_vs_cpu
```

## Build Command

```bash
cd <mindspore-repo-root>
bash build.sh -e cpu -j<threads>
```

### Parameter Reference

| Param | Meaning | Values | Default |
|-------|---------|--------|---------|
| `-e cpu` | Target device | `cpu` | — |
| `-j` | Parallel threads | integer | `8` |
| `-i` | Incremental build | flag | off |
| `-d` | Debug mode | flag | off |
| `-W` | SIMD instruction set | `sse`, `neon`, `avx`, `avx512`, `off` | `avx` (cloud) |
| `-S on` | Gitee mirror for ~29/38 deps; 7 deps still need GitHub; disables `$MSLIBS_SERVER` | `on`/`off` | `off` |

### What `-e cpu` Enables

| Flag | Value |
|------|-------|
| ENABLE_CPU | on |
| ENABLE_MPI | on |

CPU build is the lightest — no CANN or CUDA dependencies.

### SIMD Selection (`-W`)

| Value | When to use |
|-------|-------------|
| `avx` | Default for x86_64 cloud servers |
| `avx512` | Servers with AVX-512 support (check `lscpu`) |
| `neon` | ARM platforms (e.g. Kunpeng) |
| `sse` | Older x86 machines |
| `off` | Disable SIMD entirely |

## Common Recipes

### Linux x86_64
```bash
bash build.sh -e cpu -j32
```

### Linux ARM (Kunpeng)
```bash
bash build.sh -e cpu -j32 -W neon
```

### macOS
```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -j4
```

### Windows
```cmd
call build.bat ms_vs_cpu
```

### Incremental (any platform, after first build)
```bash
bash build.sh -e cpu -j32 -i
```

## Build Output & Installation

```bash
# Install whl package
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/

# Or use from build output (fastest for dev iteration, Linux/macOS only)
export PYTHONPATH=<repo>/build/package:$PYTHONPATH
export LD_LIBRARY_PATH=<repo>/build/package/mindspore/lib:$LD_LIBRARY_PATH

# Verify
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

> **Windows note**: Run `import mindspore` from **outside** the source tree.
> Python on Windows treats the current directory as an execution environment,
> which causes import conflicts if you're inside the MindSpore repo root.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Illegal instruction` at runtime | SIMD mismatch | Rebuild with correct `-W` for your CPU |
| GCC version errors | Too old GCC | Upgrade to GCC 7.3+ |
| Python.h not found | Missing python-dev | `apt install python3-dev` |
