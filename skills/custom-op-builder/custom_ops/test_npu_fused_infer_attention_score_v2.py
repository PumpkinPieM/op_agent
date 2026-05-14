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
KERNEL_SOURCE = Path(__file__).with_name("npu_fused_infer_attention_score_v2.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_fused_infer_attention_score_v2_test_v8", [str(KERNEL_SOURCE)], backend="Ascend"
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


def _assert_outputs_close(expected, actual, rtol=1e-3, atol=1e-3):
    assert len(expected) == len(actual)
    for exp, act in zip(expected, actual):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


def npu_fused_infer_attention_score_v2(query, key, value, query_rope=None, key_rope=None, pse_shift=None,
                                       atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None,
                                       block_table=None, dequant_scale_query=None, dequant_scale_key=None,
                                       dequant_offset_key=None, dequant_scale_value=None,
                                       dequant_offset_value=None, dequant_scale_key_rope=None,
                                       quant_scale_out=None, quant_offset_out=None, quant_scale_p=None,
                                       learnable_sink=None, num_query_heads=1, num_key_value_heads=0,
                                       softmax_scale=1.0, pre_tokens=2147483647,
                                       next_tokens=2147483647, input_layout="BSH", sparse_mode=0,
                                       block_size=0, query_quant_mode=0, key_quant_mode=0,
                                       value_quant_mode=0, inner_precise=0, return_softmax_lse=False,
                                       query_dtype=None, key_dtype=None, value_dtype=None,
                                       query_rope_dtype=None, key_rope_dtype=None,
                                       key_shared_prefix_dtype=None, value_shared_prefix_dtype=None,
                                       dequant_scale_query_dtype=None, dequant_scale_key_dtype=None,
                                       dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None,
                                       out_dtype=None):
    return _ops().npu_fused_infer_attention_score_v2(
        query, key, value, query_rope, key_rope, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen,
        block_table, dequant_scale_query, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, dequant_scale_key_rope, quant_scale_out, quant_offset_out, quant_scale_p,
        learnable_sink, num_query_heads, num_key_value_heads, softmax_scale, pre_tokens, next_tokens,
        input_layout, sparse_mode, block_size, query_quant_mode, key_quant_mode, value_quant_mode,
        inner_precise, return_softmax_lse, query_dtype, key_dtype, value_dtype, query_rope_dtype,
        key_rope_dtype, key_shared_prefix_dtype, value_shared_prefix_dtype, dequant_scale_query_dtype,
        dequant_scale_key_dtype, dequant_scale_value_dtype, dequant_scale_key_rope_dtype, out_dtype
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "input_layout, shape",
    [
        ("BNSD", (1, 2, 16, 64)),
        ("BSH", (1, 16, 128)),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_npu_fused_infer_attention_score_v2_matches_torch_npu(input_layout, shape, return_lse):
    if not hasattr(torch_npu, "npu_fused_infer_attention_score_v2"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    rng = np.random.default_rng(30 + int(return_lse) + len(input_layout))
    query_np = rng.normal(size=shape).astype(np.float16)
    key_np = rng.normal(size=shape).astype(np.float16)
    value_np = rng.normal(size=shape).astype(np.float16)
    kwargs = dict(
        num_query_heads=2,
        num_key_value_heads=2,
        softmax_scale=1.0 / np.sqrt(64.0),
        pre_tokens=65535,
        next_tokens=65535,
        input_layout=input_layout,
        sparse_mode=0,
        block_size=0,
        query_quant_mode=0,
        key_quant_mode=0,
        value_quant_mode=0,
        inner_precise=0,
        return_softmax_lse=return_lse,
        actual_seq_qlen=[16],
        actual_seq_kvlen=[16],
    )
    expected = torch_npu.npu_fused_infer_attention_score_v2(
        _torch_tensor(query_np, torch.float16),
        _torch_tensor(key_np, torch.float16),
        _torch_tensor(value_np, torch.float16),
        **kwargs,
    )
    actual = npu_fused_infer_attention_score_v2(
        _ms_tensor(query_np, ms.float16),
        _ms_tensor(key_np, ms.float16),
        _ms_tensor(value_np, ms.float16),
        **kwargs,
    )
    _assert_outputs_close(expected, actual)
