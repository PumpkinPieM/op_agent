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
KERNEL_SOURCE = Path(__file__).with_name("npu_gelu_quant.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_gelu_quant_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_gelu_quant(self, *, input_scale=None, input_offset=None, approximate="none", quant_mode="dynamic", dst_type=None, round_mode="rint"):
    return _custom_ops.npu_gelu_quant(self, input_scale, input_offset, approximate, quant_mode, dst_type, round_mode)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,value_mode,approximate",
    [
        ((1, 2, 8), "zeros", "none"),
        ((2, 4, 8), "normal", "none"),
        ((2, 3, 16), "small", "tanh"),
    ],
)
def test_npu_gelu_quant_smoke(shape, value_mode, approximate):
    rng = np.random.default_rng(32)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    input_scale = rng.uniform(0.25, 0.75, size=(shape[-1],)).astype(np.float16)
    input_offset = rng.uniform(-1, 1, size=(shape[-1],)).astype(np.float16)
    actual = npu_gelu_quant(
        Tensor(x), input_scale=Tensor(input_scale), input_offset=Tensor(input_offset), approximate=approximate
    )
    assert len(actual) == 2
    y_np = actual[0].asnumpy()
    scale_np = actual[1].asnumpy()
    assert y_np.shape == x.shape
    assert y_np.dtype == np.int8
    assert scale_np.shape == shape[:-1]
    assert scale_np.dtype == np.float32
