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
KERNEL_SOURCE = Path(__file__).with_name("npu_transpose_batchmatmul.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_transpose_batchmatmul_test_v3", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def _torch_tensor(arr, dtype):
    return torch.from_numpy(np.array(arr, copy=True)).to(dtype).npu()


def _ms_tensor(arr, dtype):
    return Tensor(np.array(arr, copy=True)).astype(dtype)


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
    assert exp_np.dtype == act_np.dtype or exp_np.dtype == np.float32
    np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol)


def npu_transpose_batchmatmul(x, weight, bias=None, scale=None, perm_x1=None, perm_x2=None, perm_y=None,
                              batch_split_factor=1):
    return _ops().npu_transpose_batchmatmul(
        x, weight, bias, scale, perm_x1, perm_x2, perm_y, batch_split_factor
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "default_perms",
            "x_shape": (16, 32, 512),
            "w_shape": (16, 512, 128),
            "perm_x1": [0, 1, 2],
            "perm_x2": [0, 1, 2],
            "perm_y": [1, 0, 2],
            "batch_split_factor": 1,
        },
        {
            "id": "transpose_x1",
            "x_shape": (32, 16, 512),
            "w_shape": (16, 512, 128),
            "perm_x1": [1, 0, 2],
            "perm_x2": [0, 1, 2],
            "perm_y": [1, 0, 2],
            "batch_split_factor": 1,
        },
        {
            "id": "batch_split",
            "x_shape": (32, 16, 512),
            "w_shape": (16, 512, 128),
            "perm_x1": [1, 0, 2],
            "perm_x2": [0, 1, 2],
            "perm_y": [1, 0, 2],
            "batch_split_factor": 2,
        },
    ],
    ids=lambda c: c["id"],
)
def test_npu_transpose_batchmatmul_matches_torch_npu(case):
    rng = np.random.default_rng(10)
    x_np = rng.normal(size=case["x_shape"]).astype(np.float16)
    w_np = rng.normal(size=case["w_shape"]).astype(np.float16)
    expected = torch_npu.npu_transpose_batchmatmul(
        _torch_tensor(x_np, torch.float16),
        _torch_tensor(w_np, torch.float16),
        scale=None,
        perm_x1=case["perm_x1"],
        perm_x2=case["perm_x2"],
        perm_y=case["perm_y"],
        batch_split_factor=case["batch_split_factor"],
    )
    actual = npu_transpose_batchmatmul(
        _ms_tensor(x_np, ms.float16),
        _ms_tensor(w_np, ms.float16),
        None,
        None,
        case["perm_x1"],
        case["perm_x2"],
        case["perm_y"],
        case["batch_split_factor"],
    )
    _assert_close(expected, actual, rtol=1e-3, atol=1e-3)
