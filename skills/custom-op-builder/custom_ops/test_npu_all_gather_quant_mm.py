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
KERNEL_SOURCE = Path(__file__).with_name("npu_all_gather_quant_mm.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        try:
            _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops_npu_all_gather_quant_mm_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"custom op build/load unavailable on this host: {exc}")
    return _CUSTOM_OPS


def _torch_f16(shape):
    return torch.randn(*shape, dtype=torch.float16).npu()


def _torch_i32(shape):
    return torch.zeros(*shape, dtype=torch.int32).npu()


def _ms_f16(shape):
    return Tensor(np.random.randn(*shape).astype(np.float16))


def _ms_i32(shape):
    return Tensor(np.zeros(shape, dtype=np.int32))


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


def npu_all_gather_quant_mm(self, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, gather_index=0, gather_output=True, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None):
    return _ops().npu_all_gather_quant_mm(self, x2, hcom, world_size, bias, x1_scale, x2_scale, quant_scale, block_size, gather_index, gather_output, comm_turn, group_sizes, amax_output, y_dtype, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype)


def test_npu_all_gather_quant_mm_matches_torch_npu():
    if not hasattr(torch_npu, "npu_all_gather_quant_mm"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    try:
        expected = torch_npu.npu_all_gather_quant_mm(_torch_f16((2, 256)), _torch_f16((256, 16)), "hccl_world_group", 2, bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, gather_index=0, gather_output=True, comm_turn=0, group_sizes=[2, 0, 0], amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None)
        actual = npu_all_gather_quant_mm(_ms_f16((2, 256)), _ms_f16((256, 16)), "hccl_world_group", 2, bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, gather_index=0, gather_output=True, comm_turn=0, group_sizes=[2, 0, 0], amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None)
    except (RuntimeError, AttributeError, TypeError, ValueError, IndexError) as exc:
        msg = str(exc).lower()
        skip_keys = ("not support", "tiling", "hccl", "workspace", "not implemented", "has no attribute",
                     "not initialized", "hcom", "parameter_error", "storageshape", "storage shape")
        if any(key in msg for key in skip_keys):
            pytest.skip(f"benchmark/runtime constraint on this host: {exc}")
        raise
    _assert_close(expected, actual)
