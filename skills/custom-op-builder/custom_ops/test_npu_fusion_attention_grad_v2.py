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
KERNEL_SOURCE = Path(__file__).with_name("npu_fusion_attention_grad_v2.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_fusion_attention_grad_v2_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
    return _CUSTOM_OPS


def _pair(shape):
    data = np.random.randn(*shape).astype(np.float16)
    return torch.from_numpy(data).npu(), Tensor(data)


def _ms_from_torch(value):
    if value is None:
        return None
    if value.dtype == torch.bfloat16:
        value = value.float()
    return Tensor(value.detach().cpu().numpy())


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


def _assert_close(expected, actual, rtol=2e-2, atol=2e-2):
    assert len(expected) >= len(actual)
    for exp, act in zip(expected, actual):
        if exp is None:
            continue
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def npu_fusion_attention_grad_v2(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None,
                                 atten_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None,
                                 attention_in=None, query_rope=None, key_rope=None, scale_value=1.,
                                 keep_prob=1., pre_tokens=2147483647, next_tokens=2147483647,
                                 inner_precise=0, seed=0, offset=0, numels=0, prefix=None,
                                 actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                 gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None,
                                 kv_start_idx=None, softmax_layout="", sink=None):
    return tuple(_ops().npu_fusion_attention_grad_v2(
        query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum,
        softmax_in, attention_in, query_rope, key_rope, scale_value, keep_prob, pre_tokens, next_tokens,
        inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode,
        gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx, softmax_layout, sink))


@pytest.mark.parametrize(
    "shape,head_num,input_layout",
    [
        ((1, 4, 16), 2, "BSH"),
        ((1, 4, 2, 8), 2, "BSND"),
        ((1, 2, 4, 8), 2, "BNSD"),
    ],
)
def test_npu_fusion_attention_grad_v2_matches_torch_npu(shape, head_num, input_layout):
    if not hasattr(torch_npu, "npu_fusion_attention_grad_v2"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    torch_q, ms_q = _pair(shape)
    torch_k, ms_k = _pair(shape)
    torch_v, ms_v = _pair(shape)
    torch_dy, ms_dy = _pair(shape)
    attention, softmax_max, softmax_sum, softmax_out = torch_npu.npu_fusion_attention_v2(
        torch_q, torch_k, torch_v, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None,
        query_rope=None, key_rope=None, scale=1., keep_prob=1., pre_tokens=2147483647,
        next_tokens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None,
        sparse_mode=0, gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
        softmax_layout="", sink=None, dropout_mask=None, seed=0, offset=0)[:4]
    expected = torch_npu.npu_fusion_attention_grad_v2(
        torch_q, torch_k, torch_v, torch_dy, head_num, input_layout, pse=None, padding_mask=None,
        atten_mask=None, softmax_max=softmax_max, softmax_sum=softmax_sum, softmax_in=None,
        attention_in=attention, query_rope=None, key_rope=None, scale_value=1., keep_prob=1.,
        pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, seed=0, offset=0, numels=0,
        prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True,
        sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None, softmax_layout="", sink=None)
    actual = npu_fusion_attention_grad_v2(
        ms_q, ms_k, ms_v, ms_dy, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None,
        softmax_max=_ms_from_torch(softmax_max), softmax_sum=_ms_from_torch(softmax_sum), softmax_in=None,
        attention_in=_ms_from_torch(attention), query_rope=None, key_rope=None, scale_value=1., keep_prob=1.,
        pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, seed=0, offset=0, numels=0,
        prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True,
        sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None, softmax_layout="", sink=None)
    _assert_close(expected[:3], actual[:3])
