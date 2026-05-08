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
KERNEL_SOURCE = Path(__file__).with_name("npu_dynamic_mx_quant_with_dual_axis.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_dynamic_mx_quant_with_dual_axis")

if HAS_TORCH_NPU_INTERFACE:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    context.set_context(device_target="Ascend", device_id=DEVICE_ID)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    _custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_dynamic_mx_quant_with_dual_axis_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
else:
    _custom_ops = None

def npu_dynamic_mx_quant_with_dual_axis(*args, **kwargs):
    return _custom_ops.npu_dynamic_mx_quant_with_dual_axis(*args, **kwargs)

@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()

def test_npu_dynamic_mx_quant_with_dual_axis_metadata_smoke():
    if not HAS_TORCH_NPU_INTERFACE:
        pytest.skip("torch_npu on this host does not expose npu_dynamic_mx_quant_with_dual_axis")
    x=np.ones((64,64),np.float16)
    out=npu_dynamic_mx_quant_with_dual_axis(Tensor(x))
    assert len(out)==4

