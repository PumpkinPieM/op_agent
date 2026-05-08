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
KERNEL_SOURCE = Path(__file__).with_name("npu_fast_gelu.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_fast_gelu_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_fast_gelu(input):
    return _custom_ops.npu_fast_gelu(input)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_pt(arr, dtype):
    return torch.from_numpy(arr).npu().to(dtype)


def _to_ms(arr, dtype):
    out = Tensor(arr)
    if dtype == torch.float16:
        return out.astype(ms.float16)
    if dtype == torch.bfloat16:
        return out.astype(ms.bfloat16)
    return out.astype(ms.float32)


def _pt_np(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy()
    return tensor.cpu().numpy()


def _ms_np(tensor):
    if tensor.dtype == ms.bfloat16:
        return tensor.astype(ms.float32).asnumpy()
    return tensor.asnumpy()


@pytest.mark.parametrize("shape", [(0,), (8,), (2, 3, 4)])
@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float16, 1e-3, 1e-3), (torch.float32, 1e-4, 1e-4)])
def test_npu_fast_gelu_matches_torch_npu(shape, dtype, rtol, atol):
    rng = np.random.default_rng(0)
    arr = rng.uniform(-8, 8, shape).astype(np.float32)

    expected = torch_npu.npu_fast_gelu(_to_pt(arr, dtype))
    actual = npu_fast_gelu(_to_ms(arr, dtype))

    np.testing.assert_allclose(_ms_np(actual), _pt_np(expected), rtol=rtol, atol=atol)
