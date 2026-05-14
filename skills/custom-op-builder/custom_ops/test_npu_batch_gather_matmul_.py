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
KERNEL_SOURCE = Path(__file__).with_name("npu_batch_gather_matmul_.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_batch_gather_matmul__test_v5",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def _torch_from_np(arr):
    if arr is None:
        return None
    return torch.from_numpy(np.array(arr, copy=True)).npu()


def _ms_from_np(arr):
    if arr is None:
        return None
    return Tensor(np.array(arr, copy=True))


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_close(expected, actual, rtol=1e-3, atol=1e-3):
    exp_np = _np_from_torch(expected)
    act_np = _np_from_ms(actual)
    assert exp_np.shape == act_np.shape
    np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


def _runtime_skip(exc):
    msg = str(exc).lower()
    skip_keys = (
        "not support",
        "tiling",
        "hccl",
        "workspace",
        "not implemented",
        "has no attribute",
        "not initialized",
        "hcom",
        "not in libopapi.so",
        "does not support optype",
        "aclinit failed",
    )
    if any(key in msg for key in skip_keys):
        pytest.skip(f"benchmark/runtime constraint on this host: {exc}")
    raise exc


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def npu_batch_gather_matmul_(self, x, weight_b, indices, weight_a=None, layer_idx=0, scale=1e-3, y_offset=0,
                             y_slice_size=-1):
    return _ops().npu_batch_gather_matmul_(self, x, weight_b, indices, weight_a, layer_idx, scale, y_offset,
                                           y_slice_size)


def _run_case(case):
    if not hasattr(torch_npu, "npu_batch_gather_matmul_"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    expected = torch_npu.npu_batch_gather_matmul_(
        _torch_from_np(case["self"]),
        _torch_from_np(case["x"]),
        _torch_from_np(case["weight_b"]),
        _torch_from_np(case["indices"]),
        _torch_from_np(case["weight_a"]),
        case["layer_idx"],
        case["scale"],
        case["y_offset"],
        case["y_slice_size"],
    )
    actual = npu_batch_gather_matmul_(
        _ms_from_np(case["self"]),
        _ms_from_np(case["x"]),
        _ms_from_np(case["weight_b"]),
        _ms_from_np(case["indices"]),
        _ms_from_np(case["weight_a"]),
        case["layer_idx"],
        case["scale"],
        case["y_offset"],
        case["y_slice_size"],
    )
    _assert_close(expected, actual)
def _case_weight_a_absent_full_slice():
    return {
        "self": np.zeros((2, 16), dtype=np.float16),
        "x": np.linspace(-1.0, 1.0, num=2 * 16, dtype=np.float16).reshape(2, 16),
        "weight_b": np.ones((1, 1, 16, 16), dtype=np.float16),
        "indices": np.zeros((2,), dtype=np.int32),
        "weight_a": None,
        "layer_idx": 0,
        "scale": 1e-3,
        "y_offset": 0,
        "y_slice_size": -1,
    }


def _case_weight_a_present():
    rng = np.random.default_rng(12)
    return {
        "self": rng.normal(size=(2, 128)).astype(np.float16),
        "x": rng.normal(size=(2, 16)).astype(np.float16),
        "weight_b": rng.normal(size=(2, 1, 128, 16)).astype(np.float16),
        "indices": np.array([0, 1], dtype=np.int32),
        "weight_a": rng.normal(size=(2, 1, 16, 16)).astype(np.float16),
        "layer_idx": 0,
        "scale": 2.0,
        "y_offset": 0,
        "y_slice_size": 128,
    }


def _case_offset_slice():
    return {
        "self": np.zeros((2, 32), dtype=np.float16),
        "x": np.full((2, 16), 0.25, dtype=np.float16),
        "weight_b": np.ones((1, 1, 16, 16), dtype=np.float16),
        "indices": np.zeros((2,), dtype=np.int32),
        "weight_a": None,
        "layer_idx": 0,
        "scale": 0.5,
        "y_offset": 16,
        "y_slice_size": 16,
    }


@pytest.mark.parametrize("case_fn", [_case_weight_a_absent_full_slice, _case_weight_a_present, _case_offset_slice])
def test_npu_batch_gather_matmul__matches_torch_npu(case_fn):
    _run_case(case_fn())
