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
KERNEL_SOURCE = Path(__file__).with_name("npu_group_norm_swish_grad.cc")
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_group_norm_swish_grad_test", [str(KERNEL_SOURCE)], backend="Ascend"
).load()


def npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale=1.0):
    return _custom_ops.npu_group_norm_swish_grad(
        grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("shape,num_groups,mask", [((1, 4, 2, 2), 2, [True, True, True]), ((2, 8, 2, 2), 4, [True, False, True])])
def test_npu_group_norm_swish_grad_matches_torch_npu(shape, num_groups, mask):
    rng = np.random.default_rng(42)
    x = rng.normal(size=shape).astype(np.float16)
    grad = rng.normal(size=shape).astype(np.float16)
    weight = rng.uniform(0.5, 1.5, size=(shape[1],)).astype(np.float16)
    bias = rng.uniform(-0.5, 0.5, size=(shape[1],)).astype(np.float16)
    x_t = torch.from_numpy(x).npu()
    weight_t = torch.from_numpy(weight).npu()
    bias_t = torch.from_numpy(bias).npu()
    _, mean_t, rstd_t = torch_npu.npu_group_norm_swish(x_t, num_groups, weight_t, bias_t)
    expected = torch_npu.npu_group_norm_swish_grad(
        torch.from_numpy(grad).npu(), x_t, num_groups, weight_t, bias_t, mean_t, rstd_t, mask
    )
    actual = npu_group_norm_swish_grad(
        Tensor(grad), Tensor(x), num_groups, Tensor(weight), Tensor(bias),
        Tensor(mean_t.cpu().numpy()), Tensor(rstd_t.cpu().numpy()), mask
    )
    for enabled, actual_item, expected_item in zip(mask, actual, expected):
        if not enabled:
            continue
        np.testing.assert_allclose(actual_item.asnumpy(), expected_item.cpu().numpy(), rtol=2e-3, atol=2e-3)
