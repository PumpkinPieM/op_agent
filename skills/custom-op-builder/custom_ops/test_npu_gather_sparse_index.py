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
KERNEL_SOURCE = Path(__file__).with_name("npu_gather_sparse_index.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_gather_sparse_index_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_gather_sparse_index(input, index):
    return _custom_ops.npu_gather_sparse_index(input, index)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "x,index",
    [
        (np.arange(24, dtype=np.float16).reshape(6, 4), np.array([0, 2, 5], dtype=np.int64)),
        (np.zeros((4, 3), dtype=np.float16), np.array([1, 1, 3], dtype=np.int64)),
        (np.arange(60, dtype=np.float16).reshape(5, 3, 4), np.array([[0, 4], [2, 1]], dtype=np.int64)),
    ],
)
def test_npu_gather_sparse_index_matches_torch_npu(x, index):
    expected = torch_npu.npu_gather_sparse_index(torch.from_numpy(x).npu(), torch.from_numpy(index).npu())
    actual = npu_gather_sparse_index(Tensor(x), Tensor(index))
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
