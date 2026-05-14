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
KERNEL_SOURCE = Path(__file__).with_name("npu_fused_infer_attention_score.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_fused_infer_attention_score_test_v3", [str(KERNEL_SOURCE)], backend="Ascend"
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


def npu_fused_infer_attention_score(query, key, value, pse_shift=None, atten_mask=None, actual_seq_lengths=None,
                                    actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None,
                                    dequant_scale2=None, quant_scale2=None, quant_offset2=None,
                                    antiquant_scale=None, antiquant_offset=None, key_antiquant_scale=None,
                                    key_antiquant_offset=None, value_antiquant_scale=None,
                                    value_antiquant_offset=None, block_table=None, query_padding_size=None,
                                    kv_padding_size=None, key_shared_prefix=None, value_shared_prefix=None,
                                    actual_shared_prefix_len=None, query_rope=None, key_rope=None,
                                    key_rope_antiquant_scale=None, num_heads=1, scale=1.0,
                                    pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH",
                                    num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0,
                                    antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0,
                                    softmax_lse_flag=False):
    return _ops().npu_fused_infer_attention_score(
        query, key, value, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1,
        quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset,
        key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, block_table,
        query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len,
        query_rope, key_rope, key_rope_antiquant_scale, num_heads, scale, pre_tokens, next_tokens, input_layout,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode,
        value_antiquant_mode, softmax_lse_flag
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("return_lse", [False, True])
def test_npu_fused_infer_attention_score_bnsd_matches_torch_npu(return_lse):
    if not hasattr(torch_npu, "npu_fused_infer_attention_score"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    rng = np.random.default_rng(20 + int(return_lse))
    query_np = rng.normal(size=(1, 2, 16, 64)).astype(np.float16)
    key_np = rng.normal(size=(1, 2, 16, 64)).astype(np.float16)
    value_np = rng.normal(size=(1, 2, 16, 64)).astype(np.float16)
    kwargs = dict(
        num_heads=2,
        scale=1.0 / np.sqrt(64.0),
        pre_tokens=65535,
        next_tokens=65535,
        input_layout="BNSD",
        num_key_value_heads=2,
        sparse_mode=0,
        inner_precise=0,
        block_size=0,
        antiquant_mode=0,
        key_antiquant_mode=0,
        value_antiquant_mode=0,
        softmax_lse_flag=return_lse,
        actual_seq_lengths=[16],
        actual_seq_lengths_kv=[16],
    )
    expected = torch_npu.npu_fused_infer_attention_score(
        _torch_tensor(query_np, torch.float16),
        _torch_tensor(key_np, torch.float16),
        _torch_tensor(value_np, torch.float16),
        **kwargs,
    )
    actual = npu_fused_infer_attention_score(
        _ms_tensor(query_np, ms.float16),
        _ms_tensor(key_np, ms.float16),
        _ms_tensor(value_np, ms.float16),
        **kwargs,
    )
    _assert_outputs_close(expected, actual)
