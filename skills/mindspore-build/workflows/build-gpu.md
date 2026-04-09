# Build MindSpore for GPU (CUDA)

> **Status: NOT YET VERIFIED.** This workflow is based on official docs but has not been
> tested on an actual GPU machine. Contributions welcome.

Self-contained guide. No other files needed.

## Prerequisites

### System Dependencies

| Software | Version | Install (Ubuntu 18.04) |
|----------|---------|----------------------|
| Python | 3.9–3.11 | `conda create -n <user>_py39 python=3.9.11 -y` |
| GCC | 7.3.0–9.4.0 | `sudo apt-get install gcc-7 -y` |
| CMake | 3.22.2+ | See [kitware APT](https://apt.kitware.com/) |
| CUDA | 11.1 or 11.6 | [NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-toolkit-archive) |
| cuDNN | 8.0.x (CUDA 11.1) or 8.5.x (CUDA 11.6) | [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) |
| Autoconf, Libtool, Automake | — | `sudo apt-get install automake autoconf libtool -y` |
| git, tclsh, patch, NUMA, Flex | — | `sudo apt-get install git tcl patch libnuma-dev flex -y` |
| wheel, setuptools, PyYAML, numpy | — | `pip install wheel setuptools pyyaml "numpy>=1.19.3,<=1.26.4"` |
| LLVM 12 (optional, for graph-kernel fusion) | 12.0.1 | See install docs |
| TensorRT (optional, for Serving) | 7.2.2 or 8.4 | [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) |

### Environment Setup (required before build)

```bash
# Set CUDA paths (adjust version number to match your install)
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.6
```

### Quick Verification

```bash
nvcc --version      # CUDA compiler available?
nvidia-smi          # GPU visible?
echo $CUDA_HOME     # Points to CUDA installation?
```

## Build Command

```bash
cd <mindspore-repo-root>
bash build.sh -e gpu [-V <cuda_version>] -j<threads>
```

### Parameter Reference

| Param | Meaning | Values | Default |
|-------|---------|--------|---------|
| `-e gpu` | Target device | `gpu` | — |
| `-V` | CUDA version | `10.1`, `11.1`, `11.6` | `10.1` |
| `-j` | Parallel threads | integer | `8` |
| `-i` | Incremental build | flag | off |
| `-d` | Debug mode | flag | off |
| `-M on` | Enable MPI+NCCL (distributed) | `on`/`off` | `on` for GPU |
| `-G` | GPU architecture | `auto`, `common`, `ptx` | `auto` |
| `-S on` | Gitee mirror for ~29/38 deps; 7 deps still need GitHub; disables `$MSLIBS_SERVER` | `on`/`off` | `off` |

### What `-e gpu` Enables

| Flag | Value | Notes |
|------|-------|-------|
| ENABLE_GPU | on | |
| GPU_BACKEND | cuda | (or `rocm` if `-e rocm`) |
| ENABLE_CPU | on | CPU backend always included |
| ENABLE_MPI | on | For distributed training |

### GPU Architecture (`-G`)

| Value | Behavior |
|-------|----------|
| `auto` | Detect installed GPU and compile for that arch (default) |
| `common` | Compile for sm_53, sm_60, sm_62, sm_70, sm_72, sm_75 |
| `ptx` | Only build PTX (JIT compiled at runtime, smaller binary) |

Use `auto` unless cross-compiling for a different GPU.

## Common Recipes

### CUDA 11.6 full build (recommended)
```bash
bash build.sh -e gpu -V 11.6 -j64
```

> If GitHub is blocked, try `-S on`. But it only covers ~29/38 deps and disables `$MSLIBS_SERVER`.

### Incremental build
```bash
bash build.sh -e gpu -V 11.6 -j64 -i
```

### ROCm (AMD GPU) build
```bash
bash build.sh -e rocm -j64
```

## Build Output & Installation

Compiled `.so` files are copied to `mindspore/python/mindspore/`.
A `.whl` package is generated in `output/`.

```bash
# Install
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/

# Or use from build output (fastest for dev iteration)
export PYTHONPATH=<repo>/build/package:$PYTHONPATH
export LD_LIBRARY_PATH=<repo>/build/package/mindspore/lib:$LD_LIBRARY_PATH

# Verify
python -c "import mindspore;mindspore.set_device(device_target='GPU');mindspore.run_check()"
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `nvcc not found` | CUDA not in PATH | `export PATH=$CUDA_HOME/bin:$PATH` |
| `Unsupported gpu architecture` | CUDA version mismatch | Match `-V` to installed CUDA |
| NCCL errors | MPI/NCCL not installed | Install NCCL, or `-M off` if single-GPU |
| OOM during compile | Too many threads | Reduce `-j` |
