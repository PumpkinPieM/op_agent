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

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_add_rms_norm_v2_test_v2",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_add_rms_norm_v2(x1, x2, gamma, epsilon):
    return _custom_ops.npu_add_rms_norm_v2(x1, x2, gamma, epsilon)


def _torch_tensor(array):
    return torch.from_numpy(np.array(array, copy=True)).npu()


def _ms_tensor(array):
    return Tensor(np.array(array, copy=True))


def _np_from_torch(value):
    return value.detach().cpu().numpy()


def _assert_exact(expected, actual):
    np.testing.assert_array_equal(expected, actual)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,gamma_shape,dtype,epsilon,value_mode",
    [
        ((1, 8), (8,), np.float16, 1e-6, "zeros"),
        ((2, 4, 8), (8,), np.float16, 1e-6, "normal"),
        ((2, 3, 4, 8), (4, 8), np.float16, 1e-4, "small"),
        ((2, 4, 8), (8,), np.float32, 1e-6, "normal"),
    ],
)
def test_npu_add_rms_norm_v2_matches_torch_npu(shape, gamma_shape, dtype, epsilon, value_mode):
    if not hasattr(torch_npu, "npu_add_rms_norm_v2"):
        pytest.skip("torch_npu.npu_add_rms_norm_v2 is not available in this environment")

    rng = np.random.default_rng(21)
    if value_mode == "zeros":
        x1_np = np.zeros(shape, dtype=dtype)
        x2_np = np.zeros(shape, dtype=dtype)
    elif value_mode == "small":
        x1_np = rng.uniform(-0.01, 0.01, size=shape).astype(dtype)
        x2_np = rng.uniform(-0.01, 0.01, size=shape).astype(dtype)
    else:
        x1_np = rng.normal(size=shape).astype(dtype)
        x2_np = rng.normal(size=shape).astype(dtype)
    gamma_np = rng.uniform(0.5, 1.5, size=gamma_shape).astype(dtype)

    torch_x1 = _torch_tensor(x1_np)
    torch_x2 = _torch_tensor(x2_np)
    torch_gamma = _torch_tensor(gamma_np)
    expected_rstd = torch_npu.npu_add_rms_norm_v2(torch_x1, torch_x2, torch_gamma, epsilon)

    ms_x1 = _ms_tensor(x1_np)
    ms_x2 = _ms_tensor(x2_np)
    actual_rstd = npu_add_rms_norm_v2(ms_x1, ms_x2, _ms_tensor(gamma_np), epsilon)

    _assert_exact(_np_from_torch(expected_rstd), actual_rstd.asnumpy())
    _assert_exact(_np_from_torch(torch_x1), ms_x1.asnumpy())
    _assert_exact(_np_from_torch(torch_x2), ms_x2.asnumpy())
