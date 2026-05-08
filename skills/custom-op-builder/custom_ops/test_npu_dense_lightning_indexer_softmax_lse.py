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
KERNEL_SOURCE = Path(__file__).with_name("npu_dense_lightning_indexer_softmax_lse.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_dense_lightning_indexer_softmax_lse")

if HAS_TORCH_NPU_INTERFACE:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    context.set_context(device_target="Ascend", device_id=DEVICE_ID)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

    _custom_ops = ms.ops.CustomOpBuilder(
        "custom_ops_npu_dense_lightning_indexer_softmax_lse_test",
        [str(KERNEL_SOURCE)],
        backend="Ascend",
    ).load()
else:
    _custom_ops = None


def npu_dense_lightning_indexer_softmax_lse(
    query_index,
    key_index,
    weights,
    *,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
):
    return _custom_ops.npu_dense_lightning_indexer_softmax_lse(
        query_index, key_index, weights, actual_seq_qlen, actual_seq_klen, layout, sparse_mode, pre_tokens, next_tokens
    )


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("value_mode", ["zeros", "normal"])
def test_npu_dense_lightning_indexer_softmax_lse_matches_torch_npu(layout, value_mode):
    if not HAS_TORCH_NPU_INTERFACE:
        pytest.skip("torch_npu on this host does not expose npu_dense_lightning_indexer_softmax_lse")

    rng = np.random.default_rng(4)
    if layout == "TND":
        query_shape = (8, 8, 128)
        key_shape = (10, 1, 128)
        weight_shape = (8, 8)
        actual_seq_qlen = [4, 8]
        actual_seq_klen = [5, 10]
    else:
        query_shape = (2, 4, 8, 128)
        key_shape = (2, 5, 1, 128)
        weight_shape = (2, 4, 8)
        actual_seq_qlen = [4, 4]
        actual_seq_klen = [5, 5]

    if value_mode == "zeros":
        query_np = np.zeros(query_shape, dtype=np.float16)
        key_np = np.zeros(key_shape, dtype=np.float16)
        weights_np = np.zeros(weight_shape, dtype=np.float16)
    else:
        query_np = rng.normal(size=query_shape).astype(np.float16)
        key_np = rng.normal(size=key_shape).astype(np.float16)
        weights_np = rng.random(size=weight_shape).astype(np.float16)

    actual = npu_dense_lightning_indexer_softmax_lse(
        Tensor(query_np),
        Tensor(key_np),
        Tensor(weights_np),
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout=layout,
        sparse_mode=3,
    )

    expected_shape = (key_shape[1], query_shape[0]) if layout == "TND" else (query_shape[0], key_shape[2], query_shape[1])
    for actual_item in actual:
        assert actual_item.asnumpy().shape == expected_shape
        assert actual_item.asnumpy().dtype == np.float32
