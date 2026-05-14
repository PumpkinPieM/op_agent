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
KERNEL_SOURCE = Path(__file__).with_name("npu_gather_pa_kv_cache.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops_npu_gather_pa_kv_cache_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
    return _CUSTOM_OPS


def _torch_tensor(arr, dtype=None):
    t = torch.from_numpy(np.array(arr, copy=True)).npu()
    if dtype is not None:
        t = t.to(dtype)
    return t


def _ms_tensor(arr, dtype=None):
    t = Tensor(np.array(arr, copy=True))
    if dtype is not None:
        t = t.astype(dtype)
    return t


def _np_from_torch(value):
    if value is None:
        return None
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value is None:
        return None
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _assert_close(expected, actual, rtol=1e-3, atol=1e-3):
    expected = _as_tuple(expected)
    actual = _as_tuple(actual)
    assert len(expected) == len(actual)
    for exp, act in zip(expected, actual):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if exp_np.dtype.kind in "iu" or act_np.dtype.kind in "iu" or exp_np.dtype == np.bool_:
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()




def _torch_npu_gather_pa_kv_cache_case():
    key = _torch_tensor(np.zeros((4, 2, 8), np.float16))
    value = _torch_tensor(np.zeros((4, 2, 8), np.float16))
    torch_npu.npu_gather_pa_kv_cache(
        _torch_tensor(np.ones((2, 4, 2, 8), np.float16)),
        _torch_tensor(np.ones((2, 4, 2, 8), np.float16)),
        _torch_tensor(np.array([[0, 1], [1, 0]], np.int32)),
        _torch_tensor(np.array([2, 2], np.int32)),
        key,
        value,
        seq_offset=_torch_tensor(np.zeros((2,), np.int32)),
        is_seq_lens_cumsum=True,
    )
    return key, value

def npu_gather_pa_kv_cache(key_cache, value_cache, block_tables, seq_lens, key, value, *, seq_offset=None, is_seq_lens_cumsum=False):
    return _ops().npu_gather_pa_kv_cache(key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset, is_seq_lens_cumsum)


CASES = [
    {"id":"valid_norm_cumsum", "torch": lambda: _torch_npu_gather_pa_kv_cache_case(), "ms": lambda: npu_gather_pa_kv_cache(_ms_tensor(np.ones((2, 4, 2, 8), np.float16)), _ms_tensor(np.ones((2, 4, 2, 8), np.float16)), _ms_tensor(np.array([[0, 1], [1, 0]], np.int32)), _ms_tensor(np.array([2, 2], np.int32)), _ms_tensor(np.zeros((4, 2, 8), np.float16)), _ms_tensor(np.zeros((4, 2, 8), np.float16)), seq_offset=_ms_tensor(np.zeros((2,), np.int32)), is_seq_lens_cumsum=True)},
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_npu_gather_pa_kv_cache_matches_torch_npu(case):
    expected = case["torch"]()
    actual = case["ms"]()
    _assert_close(expected, actual, case.get("rtol", 1e-3), case.get("atol", 1e-3))
