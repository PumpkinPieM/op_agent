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
KERNEL_SOURCE = Path(__file__).with_name("npu_gemma_rms_norm.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_gemma_rms_norm_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_gemma_rms_norm(self, gamma, epsilon=1e-6):
    return _custom_ops.npu_gemma_rms_norm(self, gamma, epsilon)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,epsilon,value_mode",
    [
        ((1, 2, 8), 1e-6, "zeros"),
        ((2, 4, 8), 1e-6, "normal"),
        ((2, 3, 16), 1e-4, "small"),
    ],
)
def test_npu_gemma_rms_norm_matches_torch_npu(shape, epsilon, value_mode):
    rng = np.random.default_rng(33)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    gamma = rng.uniform(0.5, 1.5, size=(shape[-1],)).astype(np.float16)
    expected = torch_npu.npu_gemma_rms_norm(torch.from_numpy(x).npu(), torch.from_numpy(gamma).npu(), epsilon)
    actual = npu_gemma_rms_norm(Tensor(x), Tensor(gamma), epsilon)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a.asnumpy(), e.cpu().numpy(), rtol=1e-3, atol=1e-3)
