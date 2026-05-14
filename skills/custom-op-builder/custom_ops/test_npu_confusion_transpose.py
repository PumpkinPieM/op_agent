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
KERNEL_SOURCE = Path(__file__).with_name("npu_confusion_transpose.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_confusion_transpose_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_confusion_transpose(self, perm, shape, transpose_first):
    return _custom_ops.npu_confusion_transpose(self, list(perm), list(shape), transpose_first)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "input_shape,perm,shape,transpose_first",
    [
        ((2, 3, 4), (0, 2, 1), (2, 3, 4), False),
        ((2, 3, 4), (0, 2, 1), (2, 4, 3), True),
        ((2, 3, 3, 2), (0, 2, 1), (2, 3, 6), False),
    ],
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_npu_confusion_transpose_matches_torch_npu(input_shape, perm, shape, transpose_first, dtype):
    rng = np.random.default_rng(1)
    arr = rng.normal(size=input_shape).astype(dtype)

    def _skip_if_host_kernel_missing(exc):
        message = str(exc).lower()
        if (
            "confusiontransposed does not has any binary" in message
            or "does not has any binary" in message
            or "parse dynamic kernel config fail" in message
        ):
            pytest.skip("aclnnConfusionTranspose is not supported by the CANN package on this host")

    expected = torch_npu.npu_confusion_transpose(torch.from_numpy(arr).npu(), perm, shape, transpose_first)
    actual = npu_confusion_transpose(Tensor(arr), perm, shape, transpose_first)
    actual_np = actual.asnumpy()
    np.testing.assert_array_equal(actual_np, expected.cpu().numpy())
