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
KERNEL_SOURCE = Path(__file__).with_name("npu_rms_norm_quant.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_rms_norm_quant_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon=1e-6, *, dst_dtype=None):
    return _custom_ops.npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon, dst_dtype)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,epsilon,value_mode,scale_value,offset_value",
    [
        ((16,), 1e-6, "zeros", 0.25, 0),
        ((32,), 1e-6, "normal", 0.25, 0),
        ((64,), 1e-4, "small", 0.5, 1),
    ],
)
def test_npu_rms_norm_quant_matches_torch_npu(shape, epsilon, value_mode, scale_value, offset_value):
    rng = np.random.default_rng(35)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x = rng.uniform(-0.01, 0.01, size=shape).astype(np.float16)
    else:
        x = rng.normal(size=shape).astype(np.float16)
    gamma = rng.uniform(0.5, 1.5, size=(shape[-1],)).astype(np.float16)
    beta = rng.uniform(-0.5, 0.5, size=(shape[-1],)).astype(np.float16)
    scale = np.array([scale_value], dtype=np.float16)
    offset = np.array([offset_value], dtype=np.int8)
    expected = torch_npu.npu_rms_norm_quant(
        torch.from_numpy(x).npu(), torch.from_numpy(gamma).npu(), torch.from_numpy(beta).npu(),
        torch.from_numpy(scale).npu(), torch.from_numpy(offset).npu(), epsilon
    )
    actual = npu_rms_norm_quant(Tensor(x), Tensor(gamma), Tensor(beta), Tensor(scale), Tensor(offset), epsilon)
    np.testing.assert_array_equal(actual.asnumpy(), expected.cpu().numpy())
