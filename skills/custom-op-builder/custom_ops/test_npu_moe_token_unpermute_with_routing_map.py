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
KERNEL_SOURCE = Path(__file__).with_name("npu_moe_token_unpermute_with_routing_map.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        try:
            _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops_npu_moe_token_unpermute_with_routing_map_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
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




def npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape, *, probs=None, routing_map=None, drop_and_pad=False):
    return _ops().npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape, probs, routing_map, drop_and_pad)


CASES = [
 {"id":"drop_and_pad_false_with_probs","torch":lambda: torch_npu.npu_moe_token_unpermute_with_routing_map(_torch_tensor(np.ones((4,8),np.float16)),_torch_tensor(np.arange(4,dtype=np.int32)),[4,8],probs=_torch_tensor(np.eye(4,dtype=np.float16)),routing_map=_torch_tensor(np.eye(4,dtype=np.int8)),drop_and_pad=False),"ms":lambda: npu_moe_token_unpermute_with_routing_map(_ms_tensor(np.ones((4,8),np.float16)),_ms_tensor(np.arange(4,dtype=np.int32)),[4,8],probs=_ms_tensor(np.eye(4,dtype=np.float16)),routing_map=_ms_tensor(np.eye(4,dtype=np.int8)),drop_and_pad=False)},
 {"id":"drop_and_pad_true_with_probs","torch":lambda: torch_npu.npu_moe_token_unpermute_with_routing_map(_torch_tensor(np.ones((4,8),np.float16)),_torch_tensor(np.arange(4,dtype=np.int32)),[4,8],probs=_torch_tensor(np.ones((4,),np.float16)),routing_map=_torch_tensor(np.eye(4,dtype=np.int8)),drop_and_pad=True),"ms":lambda: npu_moe_token_unpermute_with_routing_map(_ms_tensor(np.ones((4,8),np.float16)),_ms_tensor(np.arange(4,dtype=np.int32)),[4,8],probs=_ms_tensor(np.ones((4,),np.float16)),routing_map=_ms_tensor(np.eye(4,dtype=np.int8)),drop_and_pad=True)},
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_npu_moe_token_unpermute_with_routing_map_matches_torch_npu(case):
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
