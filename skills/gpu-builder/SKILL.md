---
name: gpu-builder
description: Build MindSpore GPU operators using CUDA. Use when implementing CUDA kernels in mindspore/ccsrc/plugin/device/gpu/kernel/.
---

# MindSpore GPU Builder

Build GPU operators with CUDA kernels for NVIDIA GPUs.

## When to Use

Use this skill when:
- Implementing CUDA kernels in `mindspore/ccsrc/plugin/device/gpu/kernel/`
- Writing `.cu` and `.cuh` files
- Optimizing operators for NVIDIA GPUs
- Using cuDNN, cuBLAS, or custom CUDA kernels

## Quick Start

1. Identify the operator from `mindspore/ops/op_def/yaml/`
2. Create kernel in `mindspore/ccsrc/plugin/device/gpu/kernel/`
3. Implement CUDA kernel in `.cu` file
4. Register with `MS_REG_GPU_KERNEL`
5. Add tests in `tests/st/ops/gpu/`

## Instructions

### Step 1: Create CUDA Kernel File

Create `mindspore/ccsrc/plugin/device/gpu/kernel/cuda_impl/<op_name>_impl.cu`:

```cuda
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * ...
 */

#include "<op_name>_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void OpNameKernel(const T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count;
       i += blockDim.x * gridDim.x) {
    output[i] = /* computation */;
  }
}

template <typename T>
cudaError_t CalOpName(const T *input, T *output, size_t count,
                       cudaStream_t cuda_stream) {
  int thread_num = 256;
  int block_num = (count + thread_num - 1) / thread_num;

  OpNameKernel<<<block_num, thread_num, 0, cuda_stream>>>(input, output, count);

  return GetCudaStatus();
}

// Explicit instantiation
template cudaError_t CalOpName<float>(const float *, float *, size_t, cudaStream_t);
template cudaError_t CalOpName<half>(const half *, half *, size_t, cudaStream_t);
```

### Step 2: Create Header File

Create `mindspore/ccsrc/plugin/device/gpu/kernel/cuda_impl/<op_name>_impl.cuh`:

```cuda
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_OPNAME_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_OPNAME_IMPL_CUH_

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalOpName(const T *input, T *output, size_t count,
                                       cudaStream_t cuda_stream);

#endif
```

### Step 3: Create GPU Kernel Module

Create `mindspore/ccsrc/plugin/device/gpu/kernel/<op_name>_gpu_kernel.cc`:

```cpp
#include "plugin/device/gpu/kernel/<op_name>_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/<op_name>_impl.cuh"

namespace mindspore {
namespace kernel {

bool OpNameGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                               const std::vector<KernelTensor *> &outputs) {
  return true;
}

bool OpNameGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs,
                                 void *cuda_stream) {
  auto input = GetDeviceAddress<float>(inputs, 0);
  auto output = GetDeviceAddress<float>(outputs, 0);
  size_t count = inputs[0]->size() / sizeof(float);

  auto status = CalOpName(input, output, count,
                           reinterpret_cast<cudaStream_t>(cuda_stream));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, OpName, OpNameGpuKernelMod);

}  // namespace kernel
}  // namespace mindspore
```

### Step 4: Using cuDNN

For operations with cuDNN support:

```cpp
#include <cudnn.h>

bool OpNameGpuKernelMod::Launch(...) {
  cudnnHandle_t cudnn_handle = GetCudnnHandle();

  // Create descriptors
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateTensorDescriptor(&output_desc);

  // Set descriptor parameters
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, n, c, h, w);

  // Execute operation
  float alpha = 1.0f, beta = 0.0f;
  cudnnOpTensor(cudnn_handle, op_desc, &alpha, input_desc, input,
                &alpha, input_desc, input, &beta, output_desc, output);

  // Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);

  return true;
}
```

### Step 5: Write Tests

Create `tests/st/ops/gpu/test_<op_name>_op.py`:

```python
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
import torch


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_op_name_gpu():
    """Test op_name on GPU."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

    input_np = np.random.randn(3, 4).astype(np.float32)

    ms_output = ops.op_name(Tensor(input_np))
    torch_output = torch.op_name(torch.from_numpy(input_np).cuda())

    assert np.allclose(ms_output.asnumpy(), torch_output.cpu().numpy(),
                       rtol=1e-5, atol=1e-8)
```

## Key Locations

| Component | Location |
|-----------|----------|
| CUDA kernels | `mindspore/ccsrc/plugin/device/gpu/kernel/cuda_impl/` |
| Kernel modules | `mindspore/ccsrc/plugin/device/gpu/kernel/` |
| cuDNN wrappers | `mindspore/ccsrc/plugin/device/gpu/kernel/cuda_impl/cuda_ops/` |
| Tests | `tests/st/ops/gpu/` |

## Performance Tips

1. **Coalesced memory access** - Ensure threads access contiguous memory
2. **Shared memory** - Use for frequently accessed data
3. **Occupancy** - Balance threads per block with register usage
4. **Streams** - Use CUDA streams for async execution
5. **cuDNN** - Use cuDNN for standard operations when available

## TODO

This skill is a placeholder. Detailed instructions will be added based on actual GPU kernel development patterns in MindSpore.
