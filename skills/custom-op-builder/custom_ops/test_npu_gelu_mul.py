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
KERNEL_SOURCE = Path(__file__).with_name("npu_gelu_mul.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_gelu_mul_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_gelu_mul(input, *, approximate="none"):
    return _custom_ops.npu_gelu_mul(input, approximate)


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
        ((2, 4, 16), "normal", "none"),
        ((2, 3, 16), "small", "tanh"),
    ],
)
def test_npu_gelu_mul_matches_torch_npu(shape, value_mode, approximate):
    rng = np.random.default_rng(31)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    expected = torch_npu.npu_gelu_mul(torch.from_numpy(x).npu(), approximate=approximate)
    actual = npu_gelu_mul(Tensor(x), approximate=approximate)
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
