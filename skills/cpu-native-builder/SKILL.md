---
name: cpu-native-builder
description: Build native CPU kernels for MindSpore using Eigen/SLEEF or hand-written implementations. Use when implementing ops directly in mindspore/ repository without external ATen dependency.
---

# MindSpore CPU Native Builder

Build native CPU kernels directly in the mindspore repository without ATen/libtorch dependency.

## When to Use

Use this skill when:
- Implementing kernels in `mindspore/ccsrc/plugin/device/cpu/kernel/`
- Writing optimized kernels with Eigen or SLEEF
- No external ATen/libtorch dependency is desired
- Full control over implementation is needed
- Building production-optimized kernels

## Quick Start

1. Identify the operator from `mindspore/ops/op_def/yaml/`
2. Create kernel class in `mindspore/ccsrc/plugin/device/cpu/kernel/`
3. Inherit from `CPUKernelMod` or `NativeCpuKernelMod`
4. Register with `MS_KERNEL_FACTORY_REG`
5. Add tests in `tests/st/ops/cpu/`

## Instructions

### Step 1: Identify the Operator

Find the operator definition:

```bash
# Search for operator definition
find mindspore/ops/op_def/yaml -name "*<op_name>*"
```

### Step 2: Create Kernel File

Create `mindspore/ccsrc/plugin/device/cpu/kernel/<op_name>_cpu_kernel.cc`:

```cpp
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

#include "plugin/device/cpu/kernel/<op_name>_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {

bool OpNameCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                               const std::vector<KernelTensor *> &outputs) {
  // Initialize kernel parameters
  return true;
}

int OpNameCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) {
  // Handle dynamic shapes
  return KRET_OK;
}

bool OpNameCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                 const std::vector<kernel::KernelTensor *> &workspace,
                                 const std::vector<kernel::KernelTensor *> &outputs) {
  // Get input/output pointers
  auto input = reinterpret_cast<float *>(inputs[0]->device_ptr());
  auto output = reinterpret_cast<float *>(outputs[0]->device_ptr());

  // Implement kernel logic
  size_t elem_num = inputs[0]->size() / sizeof(float);
  for (size_t i = 0; i < elem_num; ++i) {
    output[i] = /* computation */;
  }

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, OpName, OpNameCpuKernelMod);

}  // namespace kernel
}  // namespace mindspore
```

### Step 3: Create Header File

Create `mindspore/ccsrc/plugin/device/cpu/kernel/<op_name>_cpu_kernel.h`:

```cpp
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * ...
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_OPNAME_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_OPNAME_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {

class OpNameCpuKernelMod : public NativeCpuKernelMod {
 public:
  OpNameCpuKernelMod() = default;
  ~OpNameCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs,
            const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs,
             const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    };
    return support_list;
  }

 private:
  // Kernel parameters
};

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_OPNAME_CPU_KERNEL_H_
```

### Step 4: Using Eigen for Optimization

For vectorized operations, use Eigen:

```cpp
#include "Eigen/Core"

bool OpNameCpuKernelMod::Launch(...) {
  auto input = reinterpret_cast<float *>(inputs[0]->device_ptr());
  auto output = reinterpret_cast<float *>(outputs[0]->device_ptr());
  size_t elem_num = inputs[0]->size() / sizeof(float);

  // Eigen map for vectorized operations
  Eigen::Map<Eigen::ArrayXf> input_array(input, elem_num);
  Eigen::Map<Eigen::ArrayXf> output_array(output, elem_num);

  // Vectorized computation
  output_array = input_array.sin();  // Example: element-wise sin

  return true;
}
```

### Step 5: Using SLEEF for SIMD

For explicit SIMD with SLEEF:

```cpp
#include "sleef.h"

bool OpNameCpuKernelMod::Launch(...) {
  auto input = reinterpret_cast<float *>(inputs[0]->device_ptr());
  auto output = reinterpret_cast<float *>(outputs[0]->device_ptr());
  size_t elem_num = inputs[0]->size() / sizeof(float);

  // Process 8 floats at a time with AVX
  size_t i = 0;
  for (; i + 8 <= elem_num; i += 8) {
    __m256 x = _mm256_loadu_ps(input + i);
    __m256 y = Sleef_sinf8_u10(x);  // SLEEF vectorized sin
    _mm256_storeu_ps(output + i, y);
  }

  // Handle remainder
  for (; i < elem_num; ++i) {
    output[i] = sinf(input[i]);
  }

  return true;
}
```

### Step 6: Multi-threaded Execution

Use OpenMP for parallelization:

```cpp
#include <omp.h>

bool OpNameCpuKernelMod::Launch(...) {
  auto input = reinterpret_cast<float *>(inputs[0]->device_ptr());
  auto output = reinterpret_cast<float *>(outputs[0]->device_ptr());
  size_t elem_num = inputs[0]->size() / sizeof(float);

  #pragma omp parallel for
  for (size_t i = 0; i < elem_num; ++i) {
    output[i] = /* computation */;
  }

  return true;
}
```

### Step 7: Write Tests

Create `tests/st/ops/cpu/test_<op_name>_op.py`:

```python
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
import torch


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_op_name_float32():
    """Test op_name with float32."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    # Input data
    input_np = np.random.randn(3, 4).astype(np.float32)

    # MindSpore
    ms_input = Tensor(input_np)
    ms_output = ops.op_name(ms_input)

    # PyTorch reference
    torch_input = torch.from_numpy(input_np)
    torch_output = torch.op_name(torch_input)

    # Compare
    assert np.allclose(ms_output.asnumpy(), torch_output.numpy(), rtol=1e-5, atol=1e-8)
```

### Step 8: Build and Test

```bash
cd mindspore
bash build.sh -e cpu -j 16

# Run tests
pytest tests/st/ops/cpu/test_<op_name>_op.py -v
```

## Key Locations

| Component | Location |
|-----------|----------|
| Kernels | `mindspore/ccsrc/plugin/device/cpu/kernel/` |
| Base class | `mindspore/ccsrc/plugin/device/cpu/kernel/cpu_kernel.h` |
| Eigen | `third_party/eigen/` |
| Tests | `tests/st/ops/cpu/` |

## Test Coverage Checklist

- [ ] All supported dtypes (float16, float32, float64, int32, int64)
- [ ] Various tensor shapes (0D to 8D)
- [ ] Empty tensors
- [ ] Non-contiguous tensors
- [ ] Large tensors (stress test)
- [ ] Numerical accuracy vs PyTorch
- [ ] Edge cases (NaN, Inf, zeros)

## Performance Tips

1. **Use Eigen** for automatic vectorization
2. **Use SLEEF** for explicit SIMD control
3. **Use OpenMP** for multi-threading
4. **Minimize memory allocation** in Launch()
5. **Preallocate workspace** in Resize()
6. **Cache kernel parameters** in Init()

## Troubleshooting

**Kernel not found**: Check `MS_KERNEL_FACTORY_REG` registration

**Type mismatch**: Verify `GetOpSupport()` includes all needed types

**Performance issues**: Profile with `perf` or Intel VTune

## References

- `mindspore/ccsrc/plugin/device/cpu/kernel/` - Existing kernel implementations
- `third_party/eigen/` - Eigen documentation
- MindSpore CPU kernel development guide (internal docs)
