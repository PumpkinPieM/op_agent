import gc
import os
from pathlib import Path

import mindspore as ms
import numpy as np
import pytest
import torch
import torch_npu
from mindspore import Tensor, context

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_nsa_compress_attention.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=True)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_nsa_compress_attention_test", [str(KERNEL_SOURCE)], backend="Ascend"
).load()


def npu_nsa_compress_attention(*args):
    return _custom_ops.npu_nsa_compress_attention(*args)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()


def _np_from_torch(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy()
    return tensor.detach().cpu().numpy()


def _np_from_ms(tensor):
    if tensor.dtype == ms.bfloat16:
        return tensor.astype(ms.float32).asnumpy()
    return tensor.asnumpy()


def _to_ms(tensor):
    return Tensor(_np_from_torch(tensor))


def _assert_outputs_close(expected, actual):
    assert len(expected) == len(actual)
    for index, (exp, got) in enumerate(zip(expected, actual)):
        exp_np = _np_from_torch(exp)
        got_np = _np_from_ms(got)
        assert exp_np.shape == got_np.shape
        if index == 1:
            np.testing.assert_array_equal(exp_np, got_np)
        else:
            np.testing.assert_allclose(exp_np, got_np, rtol=1e-2, atol=1e-2, equal_nan=True)


def _case(with_masks):
    torch.manual_seed(4)
    query = torch.randn((16, 2, 16), dtype=torch.float16, device="npu")
    key = torch.randn((16, 2, 16), dtype=torch.float16, device="npu")
    value = torch.randn((16, 2, 16), dtype=torch.float16, device="npu")
    scale_value = 0.25
    head_num = 2
    compress_block_size = 16
    compress_stride = 16
    select_block_size = 16
    select_block_count = 1
    actual_seq_qlen = [16]
    actual_cmp_seq_kvlen = [16]
    actual_sel_seq_kvlen = [1]

    if with_masks:
        atten_mask = torch.triu(torch.ones((16, 16), dtype=torch.bool, device="npu"), diagonal=1)
        topk_mask = torch.zeros((16, 1), dtype=torch.bool, device="npu")
    else:
        atten_mask = None
        topk_mask = None

    torch_args = (
        query,
        key,
        value,
        scale_value,
        head_num,
        compress_block_size,
        compress_stride,
        select_block_size,
        select_block_count,
    )
    kwargs = {
        "topk_mask": topk_mask,
        "atten_mask": atten_mask,
        "actual_seq_qlen": actual_seq_qlen,
        "actual_cmp_seq_kvlen": actual_cmp_seq_kvlen,
        "actual_sel_seq_kvlen": actual_sel_seq_kvlen,
    }
    ms_args = (
        _to_ms(query),
        _to_ms(key),
        _to_ms(value),
        scale_value,
        head_num,
        compress_block_size,
        compress_stride,
        select_block_size,
        select_block_count,
        None if topk_mask is None else Tensor(_np_from_torch(topk_mask)),
        None if atten_mask is None else Tensor(_np_from_torch(atten_mask)),
        actual_seq_qlen,
        actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen,
    )
    return torch_args, kwargs, ms_args


@pytest.mark.parametrize("with_masks", [False, True])
def test_npu_nsa_compress_attention_against_torch_npu(with_masks):
    torch_args, kwargs, ms_args = _case(with_masks)
    expected = torch_npu.npu_nsa_compress_attention(*torch_args, **kwargs)
    actual = npu_nsa_compress_attention(*ms_args)
    _assert_outputs_close(expected, actual)
