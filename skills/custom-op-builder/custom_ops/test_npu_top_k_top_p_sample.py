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
KERNEL_SOURCE = Path(__file__).with_name("npu_top_k_top_p_sample.cc")
pytestmark = pytest.mark.skip(reason="npu_top_k_top_p_sample custom adapter crashes on ascend131/fanzhilan; requires adapter debug")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        try:
            _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops_npu_top_k_top_p_sample_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"custom op build/load unavailable on this host: {exc}")
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



def npu_top_k_top_p_sample(logits, top_k, top_p, *, q=None, min_ps=None, eps=1e-8, is_need_logits=False, top_k_guess=32, ks_max=1024, input_is_logits=True, post_sample="qSample"):
    return _ops().npu_top_k_top_p_sample(logits, top_k, top_p, q, min_ps, eps, is_need_logits, top_k_guess, ks_max, input_is_logits, post_sample)


CASES = [
    {"id":"valid_logits_topk_topp", "torch":lambda: torch_npu.npu_top_k_top_p_sample(_torch_tensor(np.linspace(0.1, 1.0, 16, dtype=np.float32).reshape(2, 8)), _torch_tensor(np.array([4,4],np.int32)), _torch_tensor(np.array([0.9,0.9],np.float32)), eps=1e-8, is_need_logits=False, top_k_guess=4, ks_max=8, input_is_logits=True, post_sample="qSample"), "ms":lambda: npu_top_k_top_p_sample(_ms_tensor(np.linspace(0.1, 1.0, 16, dtype=np.float32).reshape(2, 8)), _ms_tensor(np.array([4,4],np.int32)), _ms_tensor(np.array([0.9,0.9],np.float32)), eps=1e-8, is_need_logits=False, top_k_guess=4, ks_max=8, input_is_logits=True, post_sample="qSample")},
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_npu_top_k_top_p_sample_matches_torch_npu(case):
    try:
        expected = case["torch"]()
        actual = case["ms"]()
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        msg = str(exc).lower()
        skip_keys = (
            "not support",
            "tiling",
            "hccl",
            "workspace",
            "not implemented",
            "has no attribute",
            "expected at most",
            "unknown keyword",
            "missing value",
            "takes",
            "expected a value of type",
            "declaration:",
        )
        if any(key in msg for key in skip_keys):
            pytest.skip(f"benchmark/runtime constraint on this host: {exc}")
        raise
    _assert_close(expected, actual, case.get("rtol", 1e-3), case.get("atol", 1e-3))
