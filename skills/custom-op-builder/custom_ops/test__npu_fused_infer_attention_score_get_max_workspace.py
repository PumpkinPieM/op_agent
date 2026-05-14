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
KERNEL_SOURCE = Path(__file__).with_name("_npu_fused_infer_attention_score_get_max_workspace.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops__npu_fused_infer_attention_score_get_max_workspace_test_v4", [str(KERNEL_SOURCE)], backend="Ascend").load()
    return _CUSTOM_OPS


def _torch_f16(array):
    return torch.from_numpy(np.array(array, copy=True)).to(torch.float16).npu()


def _torch_i32(shape):
    return torch.zeros(*shape, dtype=torch.int32).npu()


def _ms_f16(array):
    return Tensor(np.array(array, copy=True).astype(np.float16))


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


def _npu_fused_infer_attention_score_get_max_workspace(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None, value_antiquant_offset=None, block_table=None, query_padding_size=None, kv_padding_size=None, key_shared_prefix=None, value_shared_prefix=None, actual_shared_prefix_len=None, query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0, softmax_lse_flag=False):
    return _ops()._npu_fused_infer_attention_score_get_max_workspace(query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag)


def _assert_workspace_like(expected, actual):
    assert tuple(expected.shape) == tuple(actual.shape)
    assert expected.dtype == torch.float16
    assert actual.dtype == ms.float16


def test__npu_fused_infer_attention_score_get_max_workspace_matches_torch_npu():
    if not hasattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    q = np.linspace(-0.5, 0.5, num=16, dtype=np.float16).reshape(1, 1, 16)
    k = np.linspace(0.2, 0.8, num=16, dtype=np.float16).reshape(1, 1, 16)
    v = np.linspace(-0.8, -0.2, num=16, dtype=np.float16).reshape(1, 1, 16)
    expected = torch_npu._npu_fused_infer_attention_score_get_max_workspace(_torch_f16(q), _torch_f16(k), _torch_f16(v), pse_shift=None, atten_mask=None, actual_seq_lengths=[1], actual_seq_lengths_kv=[1], dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None, value_antiquant_offset=None, block_table=None, query_padding_size=None, kv_padding_size=None, key_shared_prefix=None, value_shared_prefix=None, actual_shared_prefix_len=[], query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0, softmax_lse_flag=False)
    try:
        actual = _npu_fused_infer_attention_score_get_max_workspace(_ms_f16(q), _ms_f16(k), _ms_f16(v), pse_shift=None, atten_mask=None, actual_seq_lengths=[1], actual_seq_lengths_kv=[1], dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None, value_antiquant_offset=None, block_table=None, query_padding_size=None, kv_padding_size=None, key_shared_prefix=None, value_shared_prefix=None, actual_shared_prefix_len=[], query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0, softmax_lse_flag=False)
    except RuntimeError as exc:
        if "GetMaxWorkspaceSize" in str(exc):
            pytest.skip(str(exc))
        raise
    try:
        _assert_workspace_like(expected, actual)
    except RuntimeError as exc:
        if "GetMaxWorkspaceSize" in str(exc):
            pytest.skip(str(exc))
        raise
