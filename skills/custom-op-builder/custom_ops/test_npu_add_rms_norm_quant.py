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
KERNEL_SOURCE = Path(__file__).with_name("npu_add_rms_norm_quant.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_add_rms_norm_quant_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_add_rms_norm_quant(
    x1,
    x2,
    gamma,
    scales1,
    zero_points1,
    beta=None,
    scales2=None,
    zero_points2=None,
    axis=-1,
    epsilon=1e-6,
    div_mode=True,
    dst_type=None,
):
    return _custom_ops.npu_add_rms_norm_quant(
        x1, x2, gamma, scales1, zero_points1, beta, scales2, zero_points2, axis, epsilon, div_mode, dst_type
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _make_inputs(shape, value_mode):
    rng = np.random.default_rng(19)
    if value_mode == "zeros":
        x1 = np.zeros(shape, dtype=np.float16)
        x2 = np.zeros(shape, dtype=np.float16)
    elif value_mode == "small":
        x1 = rng.uniform(-0.03, 0.03, size=shape).astype(np.float16)
        x2 = rng.uniform(-0.03, 0.03, size=shape).astype(np.float16)
    else:
        x1 = rng.normal(0.0, 0.7, size=shape).astype(np.float16)
        x2 = rng.normal(0.0, 0.7, size=shape).astype(np.float16)
    gamma = rng.uniform(0.5, 1.5, size=(shape[-1],)).astype(np.float16)
    beta = rng.uniform(-0.2, 0.2, size=(shape[-1],)).astype(np.float16)
    return x1, x2, gamma, beta


def _to_torch(array, dtype=None):
    if array is None:
        return None
    tensor = torch.from_numpy(array).npu()
    return tensor if dtype is None else tensor.to(dtype)


def _to_ms(array):
    return None if array is None else Tensor(array)


def _np_from_torch(tensor):
    return tensor.cpu().numpy()


def _np_from_ms(tensor):
    return tensor.asnumpy()


def _assert_outputs_match(actual, expected, compare_y2=True):
    np.testing.assert_array_equal(_np_from_ms(actual[0]), _np_from_torch(expected[0]))
    if compare_y2:
        np.testing.assert_array_equal(_np_from_ms(actual[1]), _np_from_torch(expected[1]))
    np.testing.assert_allclose(_np_from_ms(actual[2]), _np_from_torch(expected[2]), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape,value_mode,with_beta,with_second_quant,div_mode,epsilon",
    [
        ((2, 32), "zeros", False, False, True, 1e-6),
        ((2, 32), "normal", True, True, True, 1e-6),
        ((2, 3, 64), "small", True, True, False, 1e-4),
    ],
)
def test_npu_add_rms_norm_quant_matches_torch_npu(
    shape, value_mode, with_beta, with_second_quant, div_mode, epsilon
):
    x1, x2, gamma, beta_data = _make_inputs(shape, value_mode)
    scales1 = np.full((shape[-1],), 0.25 if div_mode else 0.5, dtype=np.float32)
    zero_points1 = np.arange(shape[-1], dtype=np.int32) % 3
    beta = beta_data if with_beta else None
    scales2 = np.full((shape[-1],), 0.5 if div_mode else 0.25, dtype=np.float32) if with_second_quant else None
    zero_points2 = (np.arange(shape[-1], dtype=np.int32) % 5) - 2 if with_second_quant else None

    expected = torch_npu.npu_add_rms_norm_quant(
        _to_torch(x1),
        _to_torch(x2),
        _to_torch(gamma),
        _to_torch(scales1),
        _to_torch(zero_points1),
        beta=_to_torch(beta),
        scales2=_to_torch(scales2),
        zero_points2=_to_torch(zero_points2),
        axis=-1,
        epsilon=epsilon,
        div_mode=div_mode,
    )
    actual = npu_add_rms_norm_quant(
        _to_ms(x1),
        _to_ms(x2),
        _to_ms(gamma),
        _to_ms(scales1),
        _to_ms(zero_points1),
        _to_ms(beta),
        _to_ms(scales2),
        _to_ms(zero_points2),
        -1,
        epsilon,
        div_mode,
        None,
    )

    _assert_outputs_match(actual, expected, compare_y2=with_second_quant)
