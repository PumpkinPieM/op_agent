# MindSpore Version Compatibility Matrix

Self-contained reference for version requirements. No other files needed.

## Quick Reference

| Component | Version Range | Notes |
|-----------|---------------|-------|
| **Python** | 3.9 - 3.12 | 3.9.11 recommended for stability |
| **GCC** | 7.3.0 - 9.4.0 | GCC 7 recommended; GCC 10+ may have issues |
| **CMake** | 3.18+ | 3.22.3+ recommended |
| **git-lfs** | 2.0+ | Required for large files in repo |

## MindSpore x CANN x CUDA Compatibility

### Ascend (NPU)

| MindSpore | CANN | Python | GCC | NPU Supported |
|-----------|------|--------|-----|---------------|
| **2.5.x** | 8.0.RC1 - 8.2.RC1 | 3.9 - 3.11 | 7.3 - 9.4 | 910, 910B, A5, 310 |
| **2.4.x** | 7.0.RC1 - 8.0.RC1 | 3.9 - 3.11 | 7.3 - 9.4 | 910, 910B, 310 |
| **master** | 8.2.RC1 - 8.5.0 | 3.9 - 3.12 | 7.3 - 9.4 | 910, 910B, A5, 310 |

**CANN x Driver Compatibility** (critical):

| CANN Version | Driver Version | Firmware Version |
|--------------|----------------|------------------|
| 8.5.0 | 23.0.3+ | 7.3.0+ |
| 8.3.RC1 | 23.0.0+ | 7.1.0+ |
| 8.2.RC1 | 22.0.0+ | 7.0.0+ |
| 8.0.RC1 | 21.0.0+ | 6.3.0+ |

> **Important**: CANN version MUST match driver version. Mismatch causes runtime errors.

### GPU (CUDA)

| MindSpore | CUDA | cuDNN | Python | GCC |
|-----------|------|-------|--------|-----|
| **2.5.x** | 11.1, 11.6 | 8.0.x (11.1), 8.5.x (11.6) | 3.9 - 3.11 | 7.3 - 9.4 |
| **master** | 11.1, 11.6 | 8.0.x, 8.5.x | 3.9 - 3.12 | 7.3 - 9.4 |

### CPU

| MindSpore | Python | GCC | OS |
|-----------|--------|-----|-----|
| **2.5.x** | 3.9 - 3.11 | 7.3 - 9.4 | Linux, macOS, Windows |
| **master** | 3.9 - 3.12 | 7.3 - 9.4 | Linux, macOS, Windows |

## Common Version Mismatch Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ImportError: libascendcl.so` | CANN version != driver version | Reinstall matching CANN/driver |
| `undefined symbol: ...` | GCC version mismatch | Use GCC 7.x for compilation |
| `numpy.ndarray size changed` | NumPy version mismatch | Reinstall: `pip install numpy==1.23.5` |
| `RuntimeError: CUDA error` | CUDA/cuDNN mismatch | Verify with `nvcc --version` |
| `Illegal instruction` | SIMD mismatch | Rebuild with correct `-W` flag |
| `Python.h: No such file` | Missing python-dev | `apt install python3.9-dev` |

## Checking Your Versions

```bash
# Ascend
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
npu-smi info

# GPU
nvcc --version
nvidia-smi

# Common
python --version
gcc --version
cmake --version
pip list | grep -E "numpy|pyyaml"
```

## Version-Specific Build Flags

| Situation | Flag | Example |
|-----------|------|---------|
| Ascend 910B | `-V 910b` | `bash build.sh -e ascend -V 910b -j128` |
| Ascend A5 | `-V a5` | `bash build.sh -e ascend -V a5 -j128` |
| CUDA 11.6 | `-V 11.6` | `bash build.sh -e gpu -V 11.6 -j64` |
| ARM CPU | `-W neon` | `bash build.sh -e cpu -W neon -j32` |
