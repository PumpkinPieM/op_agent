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
KERNEL_SOURCE = Path(__file__).with_name("npu_clipped_swiglu.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_clipped_swiglu_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True):
    return _custom_ops.npu_clipped_swiglu(x, group_index, dim, alpha, limit, bias, interleaved)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,value_mode,alpha,limit,bias,interleaved",
    [
        ((1, 2, 8), "zeros", 1.702, 7.0, 1.0, True),
        ((2, 4, 16), "normal", 1.702, 7.0, 1.0, True),
        ((2, 3, 16), "small", 1.0, 3.0, 0.5, False),
    ],
)
def test_npu_clipped_swiglu_matches_torch_npu(shape, value_mode, alpha, limit, bias, interleaved):
    rng = np.random.default_rng(24)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    expected = torch_npu.npu_clipped_swiglu(
        torch.from_numpy(x).npu(), alpha=alpha, limit=limit, bias=bias, interleaved=interleaved
    )
    actual = npu_clipped_swiglu(Tensor(x), alpha=alpha, limit=limit, bias=bias, interleaved=interleaved)
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
