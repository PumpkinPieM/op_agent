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
KERNEL_SOURCE = Path(__file__).with_name("npu_group_norm_swish.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_group_norm_swish_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


def npu_group_norm_swish(input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0):
    return _custom_ops.npu_group_norm_swish(input, num_groups, weight, bias, eps, swish_scale)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("shape,num_groups,swish_scale", [((1, 4, 2, 2), 1, 1.0), ((2, 8, 3, 2), 4, 1.5)])
def test_npu_group_norm_swish_matches_torch_npu(shape, num_groups, swish_scale):
    rng = np.random.default_rng(41)
    x = rng.normal(size=shape).astype(np.float16)
    weight = rng.uniform(0.5, 1.5, size=(shape[1],)).astype(np.float16)
    bias = rng.uniform(-0.5, 0.5, size=(shape[1],)).astype(np.float16)
    expected = torch_npu.npu_group_norm_swish(
        torch.from_numpy(x).npu(),
        num_groups,
        torch.from_numpy(weight).npu(),
        torch.from_numpy(bias).npu(),
        swish_scale=swish_scale,
    )
    actual = npu_group_norm_swish(Tensor(x), num_groups, Tensor(weight), Tensor(bias), swish_scale=swish_scale)
    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item.asnumpy(), expected_item.cpu().numpy(), rtol=1e-3, atol=1e-3)
