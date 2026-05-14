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
KERNEL_SOURCE = Path(__file__).with_name("npu_block_sparse_attention.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_block_sparse_attention")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        if not HAS_TORCH_NPU_INTERFACE:
            pytest.skip("torch_npu on this host does not expose npu_block_sparse_attention")
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_block_sparse_attention_test",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def npu_block_sparse_attention(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=1,
    scale_value=0.0,
    inner_precise=1,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    softmax_lse_flag=0,
):
    return _ops().npu_block_sparse_attention(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        q_input_layout,
        kv_input_layout,
        num_key_value_heads,
        scale_value,
        inner_precise,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        softmax_lse_flag,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _unsupported_host(exc):
    text = str(exc).lower()
    markers = ("does not has any binary", "parse dynamic kernel", "not support", "unsupported", "not in")
    return any(marker in text for marker in markers)


def _to_torch(x_np, use_bf16=False):
    tensor = torch.from_numpy(x_np).npu()
    if use_bf16:
        tensor = tensor.to(torch.bfloat16)
    return tensor


def _to_ms(x_np, use_bf16=False):
    tensor = Tensor(x_np)
    if use_bf16:
        tensor = tensor.astype(ms.bfloat16)
    return tensor


def _torch_np(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _ms_np(tensor):
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.astype(ms.float32)
    return tensor.asnumpy()


def _compare_outputs(expected, actual, rtol=3e-2, atol=3e-2):
    assert len(expected) == len(actual) == 2
    for exp, act in zip(expected, actual):
        exp_np = _torch_np(exp)
        act_np = _ms_np(act)
        assert exp_np.shape == act_np.shape
        np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


def _run_reference_case(
    query_np,
    key_np,
    value_np,
    block_sparse_mask_np,
    block_shape,
    layout,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_key_value_heads,
    inner_precise,
    use_bf16=False,
):
    head_dim = query_np.shape[-1]
    scale_value = 1.0 / np.sqrt(head_dim)
    try:
        expected = torch_npu.npu_block_sparse_attention(
            _to_torch(query_np, use_bf16),
            _to_torch(key_np, use_bf16),
            _to_torch(value_np, use_bf16),
            torch.from_numpy(block_sparse_mask_np).npu(),
            block_shape,
            q_input_layout=layout,
            kv_input_layout=layout,
            num_key_value_heads=num_key_value_heads,
            scale_value=scale_value,
            inner_precise=inner_precise,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            softmax_lse_flag=1,
        )
        actual = npu_block_sparse_attention(
            _to_ms(query_np, use_bf16),
            _to_ms(key_np, use_bf16),
            _to_ms(value_np, use_bf16),
            Tensor(block_sparse_mask_np),
            block_shape,
            layout,
            layout,
            num_key_value_heads,
            scale_value,
            inner_precise,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            1,
        )
        actual_np = [_ms_np(out) for out in actual]
    except RuntimeError as exc:
        if _unsupported_host(exc):
            pytest.skip(f"aclnnBlockSparseAttention is not supported on this host: {exc}")
        raise

    assert actual[0].dtype == (ms.bfloat16 if use_bf16 else ms.float16)
    assert actual[1].dtype == ms.float32
    _compare_outputs(expected, actual)
    assert actual_np[1].shape[-1] == 1


def test_npu_block_sparse_attention_bnsd_fp16_matches_torch_npu():
    rng = np.random.default_rng(1)
    batch, num_heads, seq_len, head_dim = 1, 2, 128, 64
    block_shape = [128, 128]
    mask = np.ones((batch, num_heads, 1, 1), dtype=np.int8)
    _run_reference_case(
        rng.normal(size=(batch, num_heads, seq_len, head_dim)).astype(np.float16),
        rng.normal(size=(batch, num_heads, seq_len, head_dim)).astype(np.float16),
        rng.normal(size=(batch, num_heads, seq_len, head_dim)).astype(np.float16),
        mask,
        block_shape,
        layout="BNSD",
        actual_seq_lengths=[seq_len],
        actual_seq_lengths_kv=[seq_len],
        num_key_value_heads=num_heads,
        inner_precise=1,
    )


def test_npu_block_sparse_attention_tnd_fp16_gqa_sparse_mask_matches_torch_npu():
    rng = np.random.default_rng(2)
    tokens, num_heads, num_kv_heads, head_dim = 128, 4, 2, 64
    block_shape = [128, 128]
    mask = np.ones((1, num_heads, 1, 1), dtype=np.int8)
    _run_reference_case(
        rng.normal(size=(tokens, num_heads, head_dim)).astype(np.float16),
        rng.normal(size=(tokens, num_kv_heads, head_dim)).astype(np.float16),
        rng.normal(size=(tokens, num_kv_heads, head_dim)).astype(np.float16),
        mask,
        block_shape,
        layout="TND",
        actual_seq_lengths=[tokens],
        actual_seq_lengths_kv=[tokens],
        num_key_value_heads=num_kv_heads,
        inner_precise=1,
    )


def test_npu_block_sparse_attention_bnsd_bfloat16_inner_precise0_matches_torch_npu():
    rng = np.random.default_rng(3)
    batch, num_heads, seq_len, head_dim = 1, 2, 128, 64
    block_shape = [128, 128]
    mask = np.ones((batch, num_heads, 1, 1), dtype=np.int8)
    _run_reference_case(
        rng.normal(size=(batch, num_heads, seq_len, head_dim)).astype(np.float16),
        rng.normal(size=(batch, num_heads, seq_len, head_dim)).astype(np.float16),
        rng.normal(size=(batch, num_heads, seq_len, head_dim)).astype(np.float16),
        mask,
        block_shape,
        layout="BNSD",
        actual_seq_lengths=[seq_len],
        actual_seq_lengths_kv=[seq_len],
        num_key_value_heads=num_heads,
        inner_precise=0,
        use_bf16=True,
    )
