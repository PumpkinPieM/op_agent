import gc
import math
import os
from pathlib import Path

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_fused_floyd_attention.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=True)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_fused_floyd_attention_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def npu_fused_floyd_attention(query_ik, key_ij, value_ij, key_jk, value_jk, *, atten_mask=None, scale_value=1.0):
    if atten_mask is None:
        query_shape = query_ik.shape
        key_shape = key_ij.shape
        atten_mask = Tensor(np.zeros((query_shape[0], 1, query_shape[2], 1, key_shape[3]), dtype=np.uint8))
    return _ops().npu_fused_floyd_attention(
        query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask, scale_value
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _make_inputs(with_mask):
    rng = np.random.default_rng(11)
    b, h, n, m, k, d = 1, 1, 16, 128, 128, 32
    arrays = [
        rng.normal(0, 0.2, (b, h, n, m, d)).astype(np.float16),
        rng.normal(0, 0.2, (b, h, n, k, d)).astype(np.float16),
        rng.normal(0, 0.2, (b, h, n, k, d)).astype(np.float16),
        rng.normal(0, 0.2, (b, h, k, m, d)).astype(np.float16),
        rng.normal(0, 0.2, (b, h, k, m, d)).astype(np.float16),
    ]
    mask = None
    if with_mask:
        mask = np.zeros((b, 1, n, 1, k), dtype=np.bool_)
        mask[..., ::17] = True
    return arrays, mask, 1.0 / math.sqrt(d)


def _ms_tensor(array):
    return Tensor(array)


@pytest.mark.parametrize("with_mask", [False, True])
def test_npu_fused_floyd_attention_metadata_and_launch(with_mask):
    arrays, mask, scale = _make_inputs(with_mask)

    ms_inputs = [_ms_tensor(array) for array in arrays]
    ms_mask = None if mask is None else Tensor(mask)

    actual = npu_fused_floyd_attention(*ms_inputs, atten_mask=ms_mask, scale_value=scale)

    assert len(actual) == 3
    assert actual[0].asnumpy().shape == arrays[0].shape[:-1] + (8,)
    assert actual[1].asnumpy().shape == arrays[0].shape[:-1] + (8,)
    assert actual[2].asnumpy().shape == arrays[0].shape
    assert actual[0].asnumpy().dtype == np.float32
    assert actual[1].asnumpy().dtype == np.float32
    assert actual[2].asnumpy().dtype == np.float16
