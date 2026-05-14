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
KERNEL_SOURCE = Path(__file__).with_name("npu_quant_matmul.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_quant_matmul_test_v2", [str(KERNEL_SOURCE)], backend="Ascend"
).load()


def npu_quant_matmul(
    x1,
    x2,
    scale,
    *,
    offset=None,
    pertoken_scale=None,
    bias=None,
    output_dtype=None,
    x1_dtype=None,
    x2_dtype=None,
    pertoken_scale_dtype=None,
    scale_dtype=None,
    group_sizes=None,
    y_scale=None,
):
    return _custom_ops.npu_quant_matmul(
        x1,
        x2,
        scale,
        offset,
        pertoken_scale,
        bias,
        output_dtype,
        x1_dtype,
        x2_dtype,
        pertoken_scale_dtype,
        scale_dtype,
        group_sizes,
        y_scale,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(value):
    if value is None:
        return None
    return torch.from_numpy(np.array(value, copy=True)).npu()


def _ms_tensor(value):
    if value is None:
        return None
    return Tensor(np.array(value, copy=True))


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_close(expected, actual, rtol=1e-2, atol=1e-2):
    expected_np = _np_from_torch(expected)
    actual_np = _np_from_ms(actual)
    assert expected_np.shape == actual_np.shape
    if expected_np.dtype.kind in "iu":
        np.testing.assert_array_equal(expected_np, actual_np)
    else:
        np.testing.assert_allclose(expected_np, actual_np, rtol=rtol, atol=atol, equal_nan=True)


DTYPE_MAP = {
    1: torch.int8,
    3: torch.int32,
    5: torch.float16,
    6: torch.float32,
    27: torch.bfloat16,
}


CASES = [
    {
        "id": "int8_output_with_offset_and_int32_bias",
        "x1": np.array([[1, -2, 3, 0], [-1, 2, 0, 1], [3, 1, -1, 2]], dtype=np.int8),
        "x2": np.array([[1, 0, -1, 2, 1], [0, 2, 1, -1, 0], [2, -1, 0, 1, -2], [1, 1, 2, 0, 1]], dtype=np.int8),
        "scale": np.full((5,), 0.125, dtype=np.float32),
        "offset": np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
        "pertoken_scale": None,
        "bias": np.array([1, -2, 0, 3, -1], dtype=np.int32),
        "output_dtype": None,
        "rtol": 0,
        "atol": 0,
    },
    {
        "id": "float16_output_with_pertoken_scale",
        "x1": np.array([[1, -2, 3, 0], [-1, 2, 0, 1], [3, 1, -1, 2]], dtype=np.int8),
        "x2": np.array([[1, 0, -1, 2, 1], [0, 2, 1, -1, 0], [2, -1, 0, 1, -2], [1, 1, 2, 0, 1]], dtype=np.int8),
        "scale": np.array([0.25, 0.5, 0.125, 0.25, 0.5], dtype=np.float32),
        "offset": None,
        "pertoken_scale": np.array([1.0, 0.5, 0.25], dtype=np.float32),
        "bias": np.array([0.25, -0.5, 0.0, 0.5, -0.25], dtype=np.float16),
        "output_dtype": 5,
        "rtol": 1e-2,
        "atol": 1e-2,
    },
]


def _torch_reference(case):
    kwargs = {
        "offset": _torch_tensor(case["offset"]),
        "pertoken_scale": _torch_tensor(case["pertoken_scale"]),
        "bias": _torch_tensor(case["bias"]),
    }
    if case["output_dtype"] is not None:
        kwargs["output_dtype"] = DTYPE_MAP[case["output_dtype"]]
    return torch_npu.npu_quant_matmul(
        _torch_tensor(case["x1"]),
        _torch_tensor(case["x2"]),
        _torch_tensor(case["scale"]),
        **{key: value for key, value in kwargs.items() if value is not None},
    )


def _custom_result(case):
    return npu_quant_matmul(
        _ms_tensor(case["x1"]),
        _ms_tensor(case["x2"]),
        _ms_tensor(case["scale"]),
        offset=_ms_tensor(case["offset"]),
        pertoken_scale=_ms_tensor(case["pertoken_scale"]),
        bias=_ms_tensor(case["bias"]),
        output_dtype=case["output_dtype"],
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_npu_quant_matmul_against_torch_npu_benchmark(case):
    assert hasattr(torch_npu, "npu_quant_matmul")
    expected = _torch_reference(case)
    actual = _custom_result(case)
    _assert_close(expected, actual, case["rtol"], case["atol"])
