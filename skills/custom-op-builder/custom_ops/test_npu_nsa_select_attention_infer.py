import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_nsa_select_attention_infer.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_nsa_select_attention_infer_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_nsa_select_attention_infer(*args):
    return _custom_ops.npu_nsa_select_attention_infer(*args)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(array):
    return torch.from_numpy(array).npu()


def _ms_tensor(array):
    return Tensor(array)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16:
            return value.float().cpu().numpy()
        return value.cpu().numpy()
    if value.dtype == ms.bfloat16:
        return value.astype(ms.float32).asnumpy()
    return value.asnumpy()


def _assert_close(expected, actual, rtol=1e-2, atol=1e-2):
    expected_np = _to_numpy(expected)
    actual_np = _to_numpy(actual)
    assert actual_np.shape == expected_np.shape
    np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol, equal_nan=True)


def _bsnd_case(dtype=np.float16, batch=1, query_seq=1, kv_seq=48):
    rng = np.random.default_rng(7)
    head_num = 1
    key_value_head_num = 1
    query_head_dim = 192
    value_head_dim = 128
    page_block_size = 64
    select_block_size = 64
    select_block_count = 1
    block_count = 1

    query = rng.normal(0, 0.1, (batch, query_seq, head_num, query_head_dim)).astype(dtype)
    key = rng.normal(0, 0.1, (block_count, page_block_size, key_value_head_num, query_head_dim)).astype(dtype)
    value = rng.normal(0, 0.1, (block_count, page_block_size, key_value_head_num, value_head_dim)).astype(dtype)
    topk_indices = np.zeros((batch, query_seq, key_value_head_num, select_block_count), dtype=np.int32)
    block_table = np.zeros((batch, block_count), dtype=np.int32)
    actual_seq_qlen = [query_seq] * batch
    actual_seq_kvlen = [kv_seq] * batch
    scale_value = 1.0 / np.sqrt(query_head_dim)
    layout = "BSND"

    torch_args = (
        _torch_tensor(query),
        _torch_tensor(key),
        _torch_tensor(value),
        _torch_tensor(topk_indices),
        scale_value,
        head_num,
        key_value_head_num,
        select_block_size,
        select_block_count,
        page_block_size,
    )
    torch_kwargs = {
        "layout": layout,
        "atten_mask": None,
        "block_table": _torch_tensor(block_table),
        "actual_seq_qlen": actual_seq_qlen,
        "actual_seq_kvlen": actual_seq_kvlen,
    }
    ms_args = (
        _ms_tensor(query),
        _ms_tensor(key),
        _ms_tensor(value),
        _ms_tensor(topk_indices),
        scale_value,
        head_num,
        key_value_head_num,
        select_block_size,
        select_block_count,
        page_block_size,
        layout,
        None,
        _ms_tensor(block_table),
        actual_seq_qlen,
        actual_seq_kvlen,
    )
    return torch_args, torch_kwargs, ms_args


def _run_case(torch_args, torch_kwargs, ms_args):
    try:
        expected = torch_npu.npu_nsa_select_attention_infer(*torch_args, **torch_kwargs)
        actual = npu_nsa_select_attention_infer(*ms_args)
        _assert_close(expected, actual)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "not support" in message or "does not has any binary" in message or "not in" in message:
            pytest.skip(f"nsa selected attention infer is not supported by this runtime: {exc}")
        raise


def test_npu_nsa_select_attention_infer_bsnd_against_torch_npu():
    assert hasattr(torch_npu, "npu_nsa_select_attention_infer")
    _run_case(*_bsnd_case())


def test_npu_nsa_select_attention_infer_repeated_sequence_attrs():
    assert hasattr(torch_npu, "npu_nsa_select_attention_infer")
    _run_case(*_bsnd_case(batch=1, query_seq=1, kv_seq=32))
    _run_case(*_bsnd_case(batch=1, query_seq=1, kv_seq=48))
