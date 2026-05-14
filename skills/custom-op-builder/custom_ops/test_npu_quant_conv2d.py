import gc
import os
from pathlib import Path

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_quant_conv2d.cc")

context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_quant_conv2d_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_quant_conv2d(*args):
    return _custom_ops.npu_quant_conv2d(*args)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _ms_tensor(array):
    return Tensor(array)


def _to_numpy(value):
    if value.dtype == ms.bfloat16:
        return value.astype(ms.float32).asnumpy()
    return value.asnumpy()


def _conv3d_quant_reference(input_array, weight_array, scale, bias, strides, pads, dilations):
    n, _, in_d, in_h, in_w = input_array.shape
    cout, cin, kd, kh, kw = weight_array.shape
    sd, sh, sw = strides
    pd, ph, pw = pads
    dd, dh, dw = dilations
    out_d = (in_d + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    out_h = (in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    padded = np.pad(
        input_array.astype(np.int32),
        ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
        mode="constant",
    )
    out = np.zeros((n, cout, out_d, out_h, out_w), dtype=np.float32)
    for batch in range(n):
        for co in range(cout):
            for od in range(out_d):
                for oh in range(out_h):
                    for ow in range(out_w):
                        acc = 0
                        for ci in range(cin):
                            for kdi in range(kd):
                                for khi in range(kh):
                                    for kwi in range(kw):
                                        id_pos = od * sd + kdi * dd
                                        ih_pos = oh * sh + khi * dh
                                        iw_pos = ow * sw + kwi * dw
                                        acc += int(padded[batch, ci, id_pos, ih_pos, iw_pos]) * int(
                                            weight_array[co, ci, kdi, khi, kwi]
                                        )
                        out[batch, co, od, oh, ow] = acc * scale[co] + bias[co]
    return out.astype(np.float16)


def _run_case(seed, input_shape, weight_shape, strides, pads, dilations):
    rng = np.random.default_rng(seed)
    input_array = rng.integers(-2, 3, size=input_shape, dtype=np.int8)
    weight_array = rng.integers(-2, 3, size=weight_shape, dtype=np.int8)
    scale = rng.uniform(0.25, 1.0, size=(weight_shape[0],)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, size=(weight_shape[0],)).astype(np.float32)
    expected = _conv3d_quant_reference(input_array, weight_array, scale, bias, strides, pads, dilations)

    actual = npu_quant_conv2d(
        _ms_tensor(input_array),
        _ms_tensor(weight_array),
        _ms_tensor(scale),
        strides,
        pads,
        dilations,
        1,
        0,
        "rint",
        5,
        _ms_tensor(bias),
        None,
        None,
        None,
    )
    actual_np = _to_numpy(actual)
    assert actual.dtype == ms.float16
    assert actual_np.shape == expected.shape
    np.testing.assert_allclose(actual_np, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shape,weight_shape,strides,pads,dilations",
    [
        ((1, 1, 1, 4, 4), (1, 1, 1, 3, 3), [1, 1, 1], [0, 0, 0], [1, 1, 1]),
        ((1, 2, 2, 5, 6), (3, 2, 1, 3, 2), [1, 1, 2], [0, 1, 0], [1, 1, 1]),
    ],
)
def test_npu_quant_conv2d_5d_aclnn(input_shape, weight_shape, strides, pads, dilations):
    _run_case(3, input_shape, weight_shape, strides, pads, dilations)
