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
KERNEL_SOURCE = Path(__file__).with_name("npu_dequant_bias.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_dequant_bias_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_dequant_bias(x, weight_scale, activation_scale=None, bias=None, *, output_dtype=None):
    return _custom_ops.npu_dequant_bias(x, weight_scale, activation_scale, bias, output_dtype)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,value_mode",
    [
        ((1, 4), "zeros"),
        ((2, 4), "mixed"),
        ((3, 8), "positive"),
    ],
)
def test_npu_dequant_bias_matches_torch_npu(shape, value_mode):
    rng = np.random.default_rng(26)
    if value_mode == "zeros":
        x = np.zeros(shape, dtype=np.int32)
    elif value_mode == "positive":
        x = rng.integers(0, 16, size=shape, dtype=np.int32)
    else:
        x = rng.integers(-16, 16, size=shape, dtype=np.int32)
    weight_scale = rng.uniform(0.25, 1.25, size=(shape[-1],)).astype(np.float32)
    expected = torch_npu.npu_dequant_bias(torch.from_numpy(x).npu(), torch.from_numpy(weight_scale).npu(), None, None)
    actual = npu_dequant_bias(Tensor(x), Tensor(weight_scale))
    np.testing.assert_allclose(actual.asnumpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
