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
KERNEL_SOURCE = Path(__file__).with_name("npu_geglu_grad.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_geglu_grad_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_geglu_grad(grad_output, self, gelu, dim=-1, approximate=1, activate_left=False):
    return _custom_ops.npu_geglu_grad(grad_output, self, gelu, dim, approximate, activate_left)


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
def test_npu_geglu_grad_matches_torch_npu(shape, value_mode, approximate, activate_left):
    rng = np.random.default_rng(30)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
        grad = np.zeros(shape[:-1] + (shape[-1] // 2,), dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
        grad = rng.uniform(-0.01, 0.01, size=shape[:-1] + (shape[-1] // 2,)).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
        grad = rng.normal(size=shape[:-1] + (shape[-1] // 2,)).astype(np.float16)
    _, gelu = torch_npu.npu_geglu(torch.from_numpy(x).npu(), -1, approximate, activate_left)
    expected = torch_npu.npu_geglu_grad(torch.from_numpy(grad).npu(), torch.from_numpy(x).npu(), gelu, -1, approximate, activate_left)
    actual = npu_geglu_grad(Tensor(grad), Tensor(x), Tensor(gelu.cpu().numpy()), -1, approximate, activate_left)
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
