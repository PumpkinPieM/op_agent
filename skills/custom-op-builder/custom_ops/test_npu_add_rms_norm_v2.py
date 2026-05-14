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
KERNEL_SOURCE = Path(__file__).with_name("npu_add_rms_norm_v2.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_add_rms_norm_v2_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_add_rms_norm_v2(x1, x2, gamma, epsilon=1e-6):
    return _custom_ops.npu_add_rms_norm_v2(x1, x2, gamma, epsilon)


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
def test_npu_add_rms_norm_v2_smoke(shape, epsilon, value_mode):
    rng = np.random.default_rng(21)
    if value_mode == "zeros":
        x1 = np.zeros(shape, dtype=np.float16)
        x2 = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x1 = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
        x2 = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x1 = rng.normal(size=shape).astype(np.float16)
        x2 = rng.normal(size=shape).astype(np.float16)
    gamma = rng.uniform(0.5, 1.5, size=(shape[-1],)).astype(np.float16)
    actual = npu_add_rms_norm_v2(Tensor(x1), Tensor(x2), Tensor(gamma), epsilon)
    assert actual.asnumpy().shape == shape[:-1] + (1,)
    assert actual.asnumpy().dtype == np.float32
