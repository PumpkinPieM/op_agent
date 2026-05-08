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
KERNEL_SOURCE = Path(__file__).with_name("npu_scaled_masked_softmax.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_scaled_masked_softmax_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_scaled_masked_softmax(x, mask, scale=1.0, fixed_triu_mask=False):
    return _custom_ops.npu_scaled_masked_softmax(x, mask, float(scale), fixed_triu_mask)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("shape,mask_shape", [((2, 2, 8, 8), (2, 2, 8, 8)), ((2, 4, 8, 16), (2, 1, 8, 16))])
@pytest.mark.parametrize("dtype,rtol,atol", [(np.float16, 1e-3, 1e-3), (np.float32, 1e-4, 1e-4)])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_npu_scaled_masked_softmax_matches_torch_npu(shape, mask_shape, dtype, rtol, atol, scale):
    rng = np.random.default_rng(2)
    x_np = rng.normal(size=shape).astype(dtype)
    mask_np = rng.integers(0, 2, size=mask_shape).astype(np.bool_)

    expected = torch_npu.npu_scaled_masked_softmax(
        torch.from_numpy(x_np).npu(), torch.from_numpy(mask_np).npu(), scale, False
    )
    actual = npu_scaled_masked_softmax(Tensor(x_np), Tensor(mask_np), scale, False)

    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=rtol, atol=atol)
