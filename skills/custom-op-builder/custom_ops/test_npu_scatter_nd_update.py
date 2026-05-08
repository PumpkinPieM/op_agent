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
KERNEL_SOURCE = Path(__file__).with_name("npu_scatter_nd_update.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_scatter_nd_update_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_scatter_nd_update(self, indices, updates):
    return _custom_ops.npu_scatter_nd_update(self, indices, updates)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "x,indices,updates",
    [
        (
            np.zeros((4, 3), dtype=np.float16),
            np.array([[0], [2]], dtype=np.int64),
            np.ones((2, 3), dtype=np.float16),
        ),
        (
            np.arange(12, dtype=np.float16).reshape(4, 3),
            np.array([[1], [3]], dtype=np.int64),
            np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.float16),
        ),
        (
            np.zeros((2, 3, 4), dtype=np.float16),
            np.array([[0, 0], [1, 2]], dtype=np.int64),
            np.ones((2, 4), dtype=np.float16),
        ),
    ],
)
def test_npu_scatter_nd_update_matches_torch_npu(x, indices, updates):
    expected = torch_npu.npu_scatter_nd_update(torch.from_numpy(x).npu(), torch.from_numpy(indices).npu(), torch.from_numpy(updates).npu())
    actual = npu_scatter_nd_update(Tensor(x), Tensor(indices), Tensor(updates))
    np.testing.assert_array_equal(actual.asnumpy(), expected.cpu().numpy())
