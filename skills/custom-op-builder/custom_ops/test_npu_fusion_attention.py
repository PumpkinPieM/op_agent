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
KERNEL_SOURCE = Path(__file__).with_name("npu_fusion_attention.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder("custom_ops_npu_fusion_attention_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
    return _CUSTOM_OPS


def _pair(shape, dtype=np.float16):
    data = np.random.randn(*shape).astype(dtype)
    return torch.from_numpy(data).npu(), Tensor(data)


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
    assert len(expected) >= len(actual)
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


def npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout="", sink=None, dropout_mask=None, seed=0, offset=0):
    outputs = _ops().npu_fusion_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, softmax_layout, sink, dropout_mask, seed, offset)
    return tuple(outputs)


@pytest.mark.parametrize(
    "shape,head_num,input_layout",
    [
        ((1, 4, 16), 2, "BSH"),
        ((1, 4, 2, 8), 2, "BSND"),
        ((1, 2, 4, 8), 2, "BNSD"),
    ],
)
def test_npu_fusion_attention_matches_torch_npu(shape, head_num, input_layout):
    if not hasattr(torch_npu, "npu_fusion_attention"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    torch_q, ms_q = _pair(shape)
    torch_k, ms_k = _pair(shape)
    torch_v, ms_v = _pair(shape)
    expected = torch_npu.npu_fusion_attention(
        torch_q, torch_k, torch_v, head_num, input_layout, None, None, None,
        1., 1., 2147483647, 2147483647, 0, None, None, None, 0, True, False, "", None, None, 0, 0)
    actual = npu_fusion_attention(
        ms_q, ms_k, ms_v, head_num, input_layout, None, None, None,
        1., 1., 2147483647, 2147483647, 0, None, None, None, 0, True, False, "", None, None, 0, 0)
    _assert_close(expected[:4], actual[:4])
