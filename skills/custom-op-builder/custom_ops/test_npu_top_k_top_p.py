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
KERNEL_SOURCE = Path(__file__).with_name("npu_top_k_top_p.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_top_k_top_p_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_top_k_top_p(logits, p=None, k=None):
    return _custom_ops.npu_top_k_top_p(logits, p, k)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,value_mode,p,k",
    [
        ((1, 8), "zeros", np.array([0.9], dtype=np.float16), np.array([4], dtype=np.int32)),
        ((2, 8), "normal", np.array([0.9, 0.8], dtype=np.float16), np.array([4, 3], dtype=np.int32)),
        ((3, 16), "small", np.array([0.95, 0.85, 0.75], dtype=np.float16), np.array([8, 4, 2], dtype=np.int32)),
    ],
)
def test_npu_top_k_top_p_matches_torch_npu(shape, value_mode, p, k):
    rng = np.random.default_rng(39)
    if value_mode == "zeros":
        logits = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        logits = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        logits = rng.normal(size=shape).astype(np.float16)
    expected = torch_npu.npu_top_k_top_p(torch.from_numpy(logits).npu(), torch.from_numpy(p).npu(), torch.from_numpy(k).npu())
    actual = npu_top_k_top_p(Tensor(logits), Tensor(p), Tensor(k))
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
