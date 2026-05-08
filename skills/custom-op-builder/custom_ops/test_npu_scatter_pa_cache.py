import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu
import mindspore as ms
from mindspore import Tensor, context

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_scatter_pa_cache.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_scatter_pa_cache")

if HAS_TORCH_NPU_INTERFACE:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    context.set_context(device_target="Ascend", device_id=DEVICE_ID)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    _custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_scatter_pa_cache_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
else:
    _custom_ops = None

def npu_scatter_pa_cache(*args, **kwargs):
    return _custom_ops.npu_scatter_pa_cache(*args, **kwargs)

@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()

def test_npu_scatter_pa_cache_metadata_smoke():
    if not HAS_TORCH_NPU_INTERFACE:
        pytest.skip("torch_npu on this host does not expose npu_scatter_pa_cache")
    key=np.ones((2,2),np.float16); cache=np.zeros((4,2),np.float16); sm=np.array([0,1],np.int32)
    out=npu_scatter_pa_cache(Tensor(key),Tensor(sm),key_cache=Tensor(cache))
    assert out.asnumpy().shape == cache.shape

