import gc
import os
from pathlib import Path

import mindspore as ms
import numpy as np
import pytest
from mindspore import Tensor, context

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_nsa_compress_infer.cc")

context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=True)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_nsa_compress_infer_test", [str(KERNEL_SOURCE)], backend="Ascend"
).load()


def npu_nsa_compress_infer(*args):
    return _custom_ops.npu_nsa_compress_infer(*args)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()


def _expected(input_np, weight_np, slot_mapping, cache_np, actual_seq_len, compress_block_size, compress_stride):
    expected = cache_np.copy()
    for batch_idx, seq_len in enumerate(actual_seq_len):
        if seq_len < compress_block_size or (seq_len - compress_block_size) % compress_stride != 0:
            continue
        if input_np.ndim == 4:
            block = input_np[0, seq_len - compress_block_size : seq_len]
        else:
            block = input_np[seq_len - compress_block_size : seq_len]
        expected[slot_mapping[batch_idx]] = (block.astype(np.float32) * weight_np[:, :, None].astype(np.float32)).sum(axis=0)
    return expected.astype(cache_np.dtype)


def _case(with_block_table):
    rng = np.random.default_rng(2)
    if with_block_table:
        input_np = rng.normal(size=(1, 64, 1, 64)).astype(np.float16)
        block_table = Tensor(np.zeros((1, 1), dtype=np.int32))
    else:
        input_np = rng.normal(size=(64, 1, 64)).astype(np.float16)
        block_table = None
    weight_np = rng.normal(size=(16, 1)).astype(np.float16)
    slot_mapping_np = np.zeros((1,), dtype=np.int32)
    cache_np = rng.normal(size=(1, 1, 64)).astype(np.float16)
    compress_block_size = 16
    compress_stride = 16
    page_block_size = 64
    actual_seq_len = [16]
    expected = _expected(input_np, weight_np, slot_mapping_np, cache_np, actual_seq_len, compress_block_size, compress_stride)

    ms_args = (
        Tensor(input_np),
        Tensor(weight_np),
        Tensor(slot_mapping_np),
        compress_block_size,
        compress_stride,
        page_block_size,
        block_table,
        actual_seq_len,
        Tensor(cache_np),
    )
    return expected, ms_args


@pytest.mark.parametrize("with_block_table", [False, True])
def test_npu_nsa_compress_infer_formula(with_block_table):
    expected, ms_args = _case(with_block_table)
    actual = npu_nsa_compress_infer(*ms_args)
    np.testing.assert_allclose(expected, actual.asnumpy(), rtol=1e-2, atol=1e-2)
