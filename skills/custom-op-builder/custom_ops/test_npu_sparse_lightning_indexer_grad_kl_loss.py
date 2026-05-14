import gc
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_sparse_lightning_indexer_grad_kl_loss.cc")


torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_sparse_lightning_indexer_grad_kl_loss_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_sparse_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    softmax_max,
    softmax_sum,
    scale_value,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
):
    return _custom_ops.npu_sparse_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _make_case(layout):
    rng = np.random.default_rng(45)
    b, s1, s2, n1, nidx1, n2, nidx2, d, didx, dr, topk = 1, 128, 128, 64, 64, 1, 1, 512, 128, 64, 2048
    query = rng.normal(size=(b, s1, n1, d)).astype(np.float16)
    key = rng.normal(size=(b, s2, n2, d)).astype(np.float16)
    query_index = rng.normal(size=(b, s1, nidx1, didx)).astype(np.float16)
    key_index = rng.normal(size=(b, s2, nidx2, didx)).astype(np.float16)
    weights = rng.uniform(-0.05, 0.05, size=(b, s1, nidx1)).astype(np.float16)
    sparse_indices = np.full((b, s1, nidx2, topk), -1, dtype=np.int32)
    for i in range(s1):
        valid = min(s2 - s1 + i + 1, topk)
        if valid <= 0:
            valid = min(s2, topk)
        sparse_indices[:, i, :, :valid] = np.arange(valid, dtype=np.int32)
    softmax_max = rng.uniform(-0.05, 0.05, size=(b, n2, s1, n1)).astype(np.float32)
    softmax_sum = rng.uniform(1.0, 2.0, size=(b, n2, s1, n1)).astype(np.float32)
    query_rope = rng.normal(size=(b, s1, n1, dr)).astype(np.float16)
    key_rope = rng.normal(size=(b, s2, n2, dr)).astype(np.float16)
    actual_seq_qlen = None
    actual_seq_klen = None
    if layout == "TND":
        query = query.reshape(s1, n1, d)
        key = key.reshape(s2, n2, d)
        query_index = query_index.reshape(s1, nidx1, didx)
        key_index = key_index.reshape(s2, nidx2, didx)
        weights = weights.reshape(s1, nidx1)
        sparse_indices = sparse_indices.reshape(s1, nidx2, topk)
        softmax_max = softmax_max.reshape(n2, s1, n1)
        softmax_sum = softmax_sum.reshape(n2, s1, n1)
        query_rope = query_rope.reshape(s1, n1, dr)
        key_rope = key_rope.reshape(s2, n2, dr)
        actual_seq_qlen = [s1]
        actual_seq_klen = [s2]
    return {
        "query": query,
        "key": key,
        "query_index": query_index,
        "key_index": key_index,
        "weights": weights,
        "sparse_indices": sparse_indices,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "actual_seq_qlen": actual_seq_qlen,
        "actual_seq_klen": actual_seq_klen,
        "layout": layout,
        "scale_value": 1.0 / math.sqrt(d),
    }


def _torch_tensor(arr):
    return torch.from_numpy(np.array(arr, copy=True)).npu()


def _ms_tensor(arr):
    return Tensor(np.array(arr, copy=True))


def _assert_outputs_close(expected, actual):
    for exp, act in zip(expected, actual):
        exp_np = exp.float().detach().cpu().numpy() if exp.dtype == torch.bfloat16 else exp.detach().cpu().numpy()
        act_np = act.astype(ms.float32).asnumpy() if act.dtype == ms.bfloat16 else act.asnumpy()
        assert exp_np.shape == act_np.shape
        np.testing.assert_allclose(exp_np, act_np, rtol=2e-2, atol=2e-2, equal_nan=True)


@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_npu_sparse_lightning_indexer_grad_kl_loss_matches_torch_npu(layout):
    case = _make_case(layout)
    expected = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        _torch_tensor(case["query"]),
        _torch_tensor(case["key"]),
        _torch_tensor(case["query_index"]),
        _torch_tensor(case["key_index"]),
        _torch_tensor(case["weights"]),
        _torch_tensor(case["sparse_indices"]),
        _torch_tensor(case["softmax_max"]),
        _torch_tensor(case["softmax_sum"]),
        case["scale_value"],
        query_rope=_torch_tensor(case["query_rope"]),
        key_rope=_torch_tensor(case["key_rope"]),
        actual_seq_qlen=case["actual_seq_qlen"],
        actual_seq_klen=case["actual_seq_klen"],
        layout=layout,
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    )
    actual = npu_sparse_lightning_indexer_grad_kl_loss(
        _ms_tensor(case["query"]),
        _ms_tensor(case["key"]),
        _ms_tensor(case["query_index"]),
        _ms_tensor(case["key_index"]),
        _ms_tensor(case["weights"]),
        _ms_tensor(case["sparse_indices"]),
        _ms_tensor(case["softmax_max"]),
        _ms_tensor(case["softmax_sum"]),
        case["scale_value"],
        query_rope=_ms_tensor(case["query_rope"]),
        key_rope=_ms_tensor(case["key_rope"]),
        actual_seq_qlen=case["actual_seq_qlen"],
        actual_seq_klen=case["actual_seq_klen"],
        layout=layout,
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    )
    _assert_outputs_close(expected, actual)
