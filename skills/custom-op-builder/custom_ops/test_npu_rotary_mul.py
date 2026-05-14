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
KERNEL_SOURCE = Path(__file__).with_name("npu_rotary_mul.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_rotary_mul_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_rotary_mul(self, r1, r2, rotary_mode="half", rotate=None):
    return _custom_ops.npu_rotary_mul(self, r1, r2, rotary_mode, rotate)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,r_shape,value_mode,rotary_mode",
    [
        ((1, 2, 8), (1, 2, 8), "zeros", "half"),
        ((2, 4, 8), (2, 4, 8), "normal", "half"),
        ((2, 2, 5, 128), (1, 2, 1, 128), "small", "interleave"),
    ],
)
def test_npu_rotary_mul_matches_torch_npu(shape, r_shape, value_mode, rotary_mode):
    rng = np.random.default_rng(36)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
        r1 = np.ones(r_shape, dtype=np.float16)
        r2 = np.zeros(r_shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
        r1 = rng.uniform(-0.01, 0.01, size=r_shape).astype(np.float16)
        r2 = rng.uniform(-0.01, 0.01, size=r_shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
        r1 = rng.normal(size=r_shape).astype(np.float16)
        r2 = rng.normal(size=r_shape).astype(np.float16)
    expected = torch_npu.npu_rotary_mul(
        torch.from_numpy(x).npu(), torch.from_numpy(r1).npu(), torch.from_numpy(r2).npu(), rotary_mode
    )
    actual = npu_rotary_mul(Tensor(x), Tensor(r1), Tensor(r2), rotary_mode)
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
