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
KERNEL_SOURCE = Path(__file__).with_name("npu_geglu.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_geglu_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_geglu(self, dim=-1, approximate=1, activate_left=False):
    return _custom_ops.npu_geglu(self, dim, approximate, activate_left)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,value_mode,approximate,activate_left",
    [
        ((1, 2, 8), "zeros", 1, False),
        ((2, 4, 16), "normal", 1, False),
        ((2, 3, 16), "small", 0, True),
    ],
)
def test_npu_geglu_matches_torch_npu(shape, value_mode, approximate, activate_left):
    rng = np.random.default_rng(29)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    expected = torch_npu.npu_geglu(torch.from_numpy(x).npu(), -1, approximate, activate_left)
    actual = npu_geglu(Tensor(x), -1, approximate, activate_left)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a.asnumpy(), e.cpu().numpy(), rtol=1e-3, atol=1e-3)
