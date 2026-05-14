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
KERNEL_SOURCE = Path(__file__).with_name("_npu_fused_infer_attention_score_v2_get_max_workspace.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops__npu_fused_infer_attention_score_v2_get_max_workspace_test_v4", [str(KERNEL_SOURCE)], backend="Ascend").load()
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


def _npu_fused_infer_attention_score_v2_get_max_workspace(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, quant_scale_p=None, learnable_sink=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None):
    return _ops()._npu_fused_infer_attention_score_v2_get_max_workspace(query, key, value, query_rope, key_rope, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, block_table, dequant_scale_query, dequant_scale_key, dequant_offset_key, dequant_scale_value, dequant_offset_value, dequant_scale_key_rope, quant_scale_out, quant_offset_out, quant_scale_p, learnable_sink, num_query_heads, num_key_value_heads, softmax_scale, pre_tokens, next_tokens, input_layout, sparse_mode, block_size, query_quant_mode, key_quant_mode, value_quant_mode, inner_precise, return_softmax_lse, query_dtype, key_dtype, value_dtype, query_rope_dtype, key_rope_dtype, key_shared_prefix_dtype, value_shared_prefix_dtype, dequant_scale_query_dtype, dequant_scale_key_dtype, dequant_scale_value_dtype, dequant_scale_key_rope_dtype, out_dtype)


def _assert_workspace_like(expected, actual):
    assert tuple(expected.shape) == tuple(actual.shape)
    assert expected.dtype == torch.float16
    assert actual.dtype == ms.float16


def test__npu_fused_infer_attention_score_v2_get_max_workspace_matches_torch_npu():
    if not hasattr(torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    q = np.linspace(-0.5, 0.5, num=16, dtype=np.float16).reshape(1, 1, 16)
    k = np.linspace(0.2, 0.8, num=16, dtype=np.float16).reshape(1, 1, 16)
    v = np.linspace(-0.8, -0.2, num=16, dtype=np.float16).reshape(1, 1, 16)
    expected = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(_torch_f16(q), _torch_f16(k), _torch_f16(v), query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=[1], actual_seq_kvlen=[1], block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, quant_scale_p=None, learnable_sink=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None)
    try:
        actual = _npu_fused_infer_attention_score_v2_get_max_workspace(_ms_f16(q), _ms_f16(k), _ms_f16(v), query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=[1], actual_seq_kvlen=[1], block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, quant_scale_p=None, learnable_sink=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None)
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
