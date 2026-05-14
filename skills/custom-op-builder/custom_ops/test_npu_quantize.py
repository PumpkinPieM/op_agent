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
KERNEL_SOURCE = Path(__file__).with_name("npu_quantize.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_quantize_test", [str(KERNEL_SOURCE)], backend="Ascend").load()


TORCH_DTYPE_IDS = {
    torch.qint8: 12,
    torch.quint8: 13,
    torch.qint32: 14,
    torch.quint4x2: 18,
}


def npu_quantize(input_x, scales, zero_points, dtype, axis=1, div_mode=True):
    return _custom_ops.npu_quantize(input_x, scales, zero_points, TORCH_DTYPE_IDS[dtype], axis, div_mode)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(array, dtype=torch.float16):
    return torch.from_numpy(np.array(array, copy=True)).to(dtype).npu()


def _ms_tensor(array, dtype=ms.float16):
    return Tensor(np.array(array, copy=True)).astype(dtype)


def _to_numpy(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.cpu().numpy()


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "div_true_per_tensor_qint8",
            "x": np.array([[-2.0, -0.4, 0.4, 2.0], [3.0, -3.0, 0.0, 1.2]], np.float32),
            "scale": np.array([0.5], np.float32),
            "zero": np.array([1], np.int8),
            "dtype": torch.qint8,
            "axis": 1,
            "div_mode": True,
        },
        {
            "id": "div_true_per_channel_quint8",
            "x": np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], np.float32),
            "scale": np.array([1.0, 2.0, 1.0, 0.5], np.float32),
            "zero": np.array([0, 1, 2, 3], np.uint8),
            "dtype": torch.quint8,
            "axis": 1,
            "div_mode": True,
        },
        {
            "id": "div_false_qint8_axis_last",
            "x": np.array([[[0.25, -0.5, 1.0, -1.25], [2.0, -2.0, 0.0, 0.5]]], np.float16),
            "scale": np.array([2.0, 3.0, 1.0, 0.5], np.float16),
            "zero": None,
            "dtype": torch.qint8,
            "axis": -1,
            "div_mode": False,
        },
        {
            "id": "div_false_quint4x2_packed",
            "x": np.linspace(-1.0, 1.0, 16, dtype=np.float16).reshape(1, 16),
            "scale": np.ones((16,), np.float16),
            "zero": None,
            "dtype": torch.quint4x2,
            "axis": -1,
            "div_mode": False,
        },
    ],
    ids=lambda c: c["id"],
)
def test_npu_quantize_matches_torch_npu(case):
    torch_x = _torch_tensor(case["x"], torch.float16 if case["x"].dtype == np.float16 else torch.float32)
    torch_scales = _torch_tensor(case["scale"], torch.float16 if case["scale"].dtype == np.float16 else torch.float32)
    torch_zero = None
    if case["zero"] is not None:
        torch_zero = torch.from_numpy(np.array(case["zero"], copy=True)).npu()

    ms_x = _ms_tensor(case["x"], ms.float16 if case["x"].dtype == np.float16 else ms.float32)
    ms_scales = _ms_tensor(case["scale"], ms.float16 if case["scale"].dtype == np.float16 else ms.float32)
    ms_zero = None if case["zero"] is None else Tensor(np.array(case["zero"], copy=True))

    expected = torch_npu.npu_quantize(
        torch_x, torch_scales, torch_zero, case["dtype"], case["axis"], case["div_mode"]
    )
    actual = npu_quantize(ms_x, ms_scales, ms_zero, case["dtype"], case["axis"], case["div_mode"])

    expected_np = _to_numpy(expected)
    actual_np = actual.asnumpy()
    assert expected_np.shape == actual_np.shape
    assert expected_np.dtype == actual_np.dtype
    np.testing.assert_array_equal(expected_np, actual_np)
