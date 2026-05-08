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
KERNEL_SOURCE = Path(__file__).with_name("npu_dense_lightning_indexer_grad_kl_loss.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_dense_lightning_indexer_grad_kl_loss")

if HAS_TORCH_NPU_INTERFACE:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    context.set_context(device_target="Ascend", device_id=DEVICE_ID)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    _custom_ops = ms.ops.CustomOpBuilder(
        "custom_ops_npu_dense_lightning_indexer_grad_kl_loss_test", [str(KERNEL_SOURCE)], backend="Ascend"
    ).load()
else:
    _custom_ops = None


def npu_dense_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    softmax_max,
    softmax_sum,
    softmax_max_index,
    softmax_sum_index,
    scale_value,
    *,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
):
    return _custom_ops.npu_dense_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        softmax_max,
        softmax_sum,
        softmax_max_index,
        softmax_sum_index,
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


def _case():
    rng = np.random.default_rng(44)
    query = rng.normal(size=(1, 4, 2, 16)).astype(np.float16)
    key = rng.normal(size=(1, 5, 1, 16)).astype(np.float16)
    query_index = rng.normal(size=(1, 4, 2, 16)).astype(np.float16)
    key_index = rng.normal(size=(1, 5, 1, 16)).astype(np.float16)
    weights = rng.normal(size=(1, 4, 2)).astype(np.float16)
    softmax_max = rng.uniform(-0.05, 0.05, size=(1, 1, 4, 2)).astype(np.float32)
    softmax_sum = rng.uniform(1.0, 2.0, size=(1, 1, 4, 2)).astype(np.float32)
    softmax_max_index = rng.normal(size=(1, 1, 4)).astype(np.float32)
    softmax_sum_index = rng.normal(size=(1, 1, 4)).astype(np.float32)
    return query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index


def test_npu_dense_lightning_indexer_grad_kl_loss_metadata():
    if not HAS_TORCH_NPU_INTERFACE:
        pytest.skip("torch_npu on this host does not expose npu_dense_lightning_indexer_grad_kl_loss")
    arrays = _case()
    actual = npu_dense_lightning_indexer_grad_kl_loss(*(Tensor(x) for x in arrays), 1.0, layout="BSND")
    expected_shapes = [arrays[2].shape, arrays[3].shape, arrays[4].shape, (1,)]
    expected_dtypes = [np.float16, np.float16, np.float16, np.float32]
    for item, shape, dtype in zip(actual, expected_shapes, expected_dtypes):
        arr = item.asnumpy()
        assert arr.shape == shape
        assert arr.dtype == dtype
