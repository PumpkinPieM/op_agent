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
KERNEL_SOURCE = Path(__file__).with_name("npu_dynamic_quant.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_dynamic_quant_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_dynamic_quant(input, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken", dst_type_max=0.0):
    return _custom_ops.npu_dynamic_quant(input, smooth_scales, group_index, dst_type, quant_mode, dst_type_max)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,value_mode,use_smooth",
    [
        ((1, 2, 8), "zeros", False),
        ((2, 4, 8), "normal", False),
        ((2, 3, 16), "small", True),
    ],
)
def test_npu_dynamic_quant_matches_torch_npu(shape, value_mode, use_smooth):
    rng = np.random.default_rng(27)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    smooth = rng.uniform(0.5, 1.5, size=(shape[-1],)).astype(np.float16) if use_smooth else None
    expected = torch_npu.npu_dynamic_quant(
        torch.from_numpy(x).npu(), smooth_scales=None if smooth is None else torch.from_numpy(smooth).npu()
    )
    actual = npu_dynamic_quant(Tensor(x), smooth_scales=None if smooth is None else Tensor(smooth))
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a.asnumpy(), e.cpu().numpy(), rtol=1e-3, atol=1e-3)
