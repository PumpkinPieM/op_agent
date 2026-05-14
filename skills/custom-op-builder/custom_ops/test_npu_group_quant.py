import gc
import os
from pathlib import Path

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_group_quant.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=True)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_group_quant_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def npu_group_quant(x, scale, group_index, offset=None, dst_dtype=None):
    return _ops().npu_group_quant(x, scale, group_index, offset, dst_dtype)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _reference_group_quant(x, scale, group_index, offset=None):
    x = x.astype(np.float32)
    scale = scale.astype(np.float32)
    offset = np.array(0, dtype=np.float32) if offset is None else offset.astype(np.float32)
    rows = []
    start = 0
    for scale_row, end in enumerate(group_index.astype(np.int64)):
        if start < end:
            rows.append(x[start:end] * scale[scale_row] + offset)
        start = end
    return np.clip(np.rint(np.concatenate(rows, axis=0)), -128, 127).astype(np.int8)


@pytest.mark.parametrize("group_index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("use_offset", [False, True])
def test_npu_group_quant_matches_cpu_reference(group_index_dtype, use_offset):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, (16, 32)).astype(np.float16)
    scale = rng.uniform(0.5, 1.5, (4, 32)).astype(np.float32)
    group_index = np.array([3, 7, 11, 16], dtype=group_index_dtype)
    offset = rng.uniform(-0.5, 0.5, (1,)).astype(np.float32) if use_offset else None

    actual = npu_group_quant(
        Tensor(x),
        Tensor(scale),
        Tensor(group_index),
        None if offset is None else Tensor(offset),
        2,
    ).asnumpy()
    expected = _reference_group_quant(x, scale, group_index, offset)

    assert actual.dtype == np.int8
    np.testing.assert_array_equal(actual, expected)


def test_npu_group_quant_default_dst_type():
    x = np.ones((4, 8), np.float32)
    scale = np.ones((1, 8), np.float32)
    group_index = np.array([4], np.int32)
    out = npu_group_quant(Tensor(x), Tensor(scale), Tensor(group_index))
    assert out.asnumpy().shape == x.shape
    assert out.asnumpy().dtype == np.int8
