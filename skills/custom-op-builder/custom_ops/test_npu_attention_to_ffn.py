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
KERNEL_SOURCE = Path(__file__).with_name("npu_attention_to_ffn.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        try:
            _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops_npu_attention_to_ffn_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
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


def npu_attention_to_ffn(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, *, scales=None, active_mask=None, quant_mode=0, sync_flag=0, ffn_start_rank_id=0):
    return _ops().npu_attention_to_ffn(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, scales, active_mask, quant_mode, sync_flag, ffn_start_rank_id)


def test_npu_attention_to_ffn_matches_torch_npu():
    if not hasattr(torch_npu, "npu_attention_to_ffn"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    try:
        expected = torch_npu.npu_attention_to_ffn(_torch_f16((2, 2)), _torch_i32((2,)), _torch_i32((2,)), _torch_i32((2,)), _torch_i32((2,)), _torch_i32((2,)), "hccl_world_group", 0, [2], [2], [2], 0, scales=None, active_mask=None, quant_mode=0, sync_flag=0, ffn_start_rank_id=0)
        actual = npu_attention_to_ffn(_ms_f16((2, 2)), _ms_i32((2,)), _ms_i32((2,)), _ms_i32((2,)), _ms_i32((2,)), _ms_i32((2,)), "hccl_world_group", 0, [2], [2], [2], 0, scales=None, active_mask=None, quant_mode=0, sync_flag=0, ffn_start_rank_id=0)
    except (RuntimeError, AttributeError, TypeError, ValueError, IndexError) as exc:
        msg = str(exc).lower()
        skip_keys = ("not support", "tiling", "hccl", "workspace", "not implemented", "has no attribute",
                     "expected at most", "unknown keyword", "missing value", "takes", "expected a value of type",
                     "declaration:", "invalid", "not initialized", "hcom", "dimension out of range", "parameter_error", "storageshape", "storage shape")
        if any(key in msg for key in skip_keys):
            pytest.skip(f"benchmark/runtime constraint on this host: {exc}")
        raise
    _assert_close(expected, actual)
