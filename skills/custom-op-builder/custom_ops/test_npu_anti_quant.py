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
KERNEL_SOURCE = Path(__file__).with_name("npu_anti_quant.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_anti_quant_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_anti_quant(x, scale, *, offset=None, dst_dtype=None, src_dtype=None):
    return _custom_ops.npu_anti_quant(x, scale, offset, dst_dtype, src_dtype)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "x,scale,offset",
    [
        (np.array([[1, -2, 3, -4]], dtype=np.int8), np.array([0.5], dtype=np.float32), None),
        (np.zeros((2, 4), dtype=np.int8), np.array([1.0], dtype=np.float32), None),
        (
            np.array([[8, -8, 4, -4], [2, -2, 1, -1]], dtype=np.int8),
            np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
            np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32),
        ),
    ],
)
def test_npu_anti_quant_matches_torch_npu(x, scale, offset):
    expected = torch_npu.npu_anti_quant(
        torch.from_numpy(x).npu(),
        torch.from_numpy(scale).npu(),
        offset=None if offset is None else torch.from_numpy(offset).npu(),
    )
    actual = npu_anti_quant(Tensor(x), Tensor(scale), offset=None if offset is None else Tensor(offset))
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
