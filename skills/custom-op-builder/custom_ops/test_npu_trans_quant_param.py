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
KERNEL_SOURCE = Path(__file__).with_name("npu_trans_quant_param.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_trans_quant_param")

if HAS_TORCH_NPU_INTERFACE:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    context.set_context(device_target="Ascend", device_id=DEVICE_ID)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    _custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_trans_quant_param_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
else:
    _custom_ops = None

def npu_trans_quant_param(*args, **kwargs):
    return _custom_ops.npu_trans_quant_param(*args, **kwargs)

@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()

def test_npu_trans_quant_param_metadata_smoke():
    if not HAS_TORCH_NPU_INTERFACE:
        pytest.skip("torch_npu on this host does not expose npu_trans_quant_param")
    s=np.ones((8,),np.float32); z=np.zeros((8,),np.float32)
    out=npu_trans_quant_param(Tensor(s), Tensor(z), None)
    assert out.asnumpy().shape == s.shape
