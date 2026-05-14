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
KERNEL_SOURCE = Path(__file__).with_name("npu_add_rms_norm_dynamic_quant.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_add_rms_norm_dynamic_quant_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_add_rms_norm_dynamic_quant(
    x1,
    x2,
    gamma,
    smooth_scale1=None,
    smooth_scale2=None,
    beta=None,
    epsilon=1e-6,
    output_mask=None,
    y_dtype=None,
):
    return _custom_ops.npu_add_rms_norm_dynamic_quant(
        x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, output_mask, y_dtype
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _make_inputs(shape, value_mode):
    rng = np.random.default_rng(23)
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
    smooth1 = rng.uniform(0.8, 1.2, size=(shape[-1],)).astype(np.float16)
    smooth2 = rng.uniform(0.8, 1.2, size=(shape[-1],)).astype(np.float16)
    return x1, x2, gamma, beta, smooth1, smooth2


def _to_torch(array):
    return None if array is None else torch.from_numpy(array).npu()


def _to_ms(array):
    return None if array is None else Tensor(array)


def _np_from_torch(tensor):
    return tensor.cpu().numpy()


def _np_from_ms(tensor):
    return tensor.asnumpy()


def _assert_outputs_match(actual, expected, output_mask):
    enabled = [bool(output_mask[0]), bool(output_mask[1])]
    if enabled[0]:
        np.testing.assert_array_equal(_np_from_ms(actual[0]), _np_from_torch(expected[0]))
        np.testing.assert_allclose(_np_from_ms(actual[3]), _np_from_torch(expected[3]), rtol=1e-3, atol=1e-3)
    if enabled[1]:
        np.testing.assert_array_equal(_np_from_ms(actual[1]), _np_from_torch(expected[1]))
        np.testing.assert_allclose(_np_from_ms(actual[4]), _np_from_torch(expected[4]), rtol=1e-3, atol=1e-3)
    else:
        assert _np_from_ms(actual[1]).shape == _np_from_torch(expected[1]).shape
        assert _np_from_ms(actual[4]).shape == _np_from_torch(expected[4]).shape
    np.testing.assert_allclose(_np_from_ms(actual[2]), _np_from_torch(expected[2]), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape,value_mode,with_smooth,with_beta,output_mask,epsilon",
    [
        ((2, 32), "zeros", False, False, [1, 1], 1e-6),
        ((2, 32), "normal", True, True, [1, 1], 1e-6),
        ((2, 3, 64), "small", True, True, [1, 0], 1e-4),
    ],
)
def test_npu_add_rms_norm_dynamic_quant_matches_torch_npu(
    shape, value_mode, with_smooth, with_beta, output_mask, epsilon
):
    x1, x2, gamma, beta_data, smooth1_data, smooth2_data = _make_inputs(shape, value_mode)
    smooth1 = smooth1_data if with_smooth else None
    smooth2 = smooth2_data if with_smooth and output_mask[1] else None
    beta = beta_data if with_beta else None

    expected = torch_npu.npu_add_rms_norm_dynamic_quant(
        _to_torch(x1),
        _to_torch(x2),
        _to_torch(gamma),
        smooth_scale1=_to_torch(smooth1),
        smooth_scale2=_to_torch(smooth2),
        beta=_to_torch(beta),
        epsilon=epsilon,
        output_mask=[bool(v) for v in output_mask],
    )
    actual = npu_add_rms_norm_dynamic_quant(
        _to_ms(x1),
        _to_ms(x2),
        _to_ms(gamma),
        _to_ms(smooth1),
        _to_ms(smooth2),
        _to_ms(beta),
        epsilon,
        output_mask,
        None,
    )

    _assert_outputs_match(actual, expected, output_mask)
