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
KERNEL_SOURCE = Path(__file__).with_name("npu_advance_step_flashattn.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_advance_step_flashattn_test",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def _torch_tensor(array):
    return torch.from_numpy(np.array(array, copy=True)).npu()


def _ms_tensor(array):
    return Tensor(np.array(array, copy=True))


def npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping,
                               block_tables, num_seqs, num_queries, block_size, spec_token=None, accepted_num=None):
    return _ops().npu_advance_step_flashattn(
        input_tokens,
        sampled_token_ids,
        input_positions,
        seq_lens,
        slot_mapping,
        block_tables,
        num_seqs,
        num_queries,
        block_size,
        spec_token,
        accepted_num,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def test_npu_advance_step_flashattn_mutates_decode_state_like_torch_npu():
    num_seqs = 4
    num_queries = 2
    block_size = 4
    input_tokens = np.array([10, 11, 12, 13], dtype=np.int64)
    sampled_token_ids = np.array([[101], [102]], dtype=np.int64)
    input_positions = np.array([0, 3, 0, 0], dtype=np.int64)
    seq_lens = np.array([1, 4, 0, 0], dtype=np.int64)
    slot_mapping = np.array([0, 0, 0, 0], dtype=np.int64)
    block_tables = np.array([[0, 5], [1, 7], [2, 8], [3, 9]], dtype=np.int64)

    torch_inputs = [
        _torch_tensor(input_tokens),
        _torch_tensor(sampled_token_ids),
        _torch_tensor(input_positions),
        _torch_tensor(seq_lens),
        _torch_tensor(slot_mapping),
        _torch_tensor(block_tables),
    ]
    ms_inputs = [
        _ms_tensor(input_tokens),
        _ms_tensor(sampled_token_ids),
        _ms_tensor(input_positions),
        _ms_tensor(seq_lens),
        _ms_tensor(slot_mapping),
        _ms_tensor(block_tables),
    ]

    torch_npu.npu_advance_step_flashattn(*torch_inputs, num_seqs, num_queries, block_size)
    actual = npu_advance_step_flashattn(*ms_inputs, num_seqs, num_queries, block_size)

    assert actual is None
    for torch_value, ms_value in zip((torch_inputs[0], torch_inputs[2], torch_inputs[3], torch_inputs[4]),
                                     (ms_inputs[0], ms_inputs[2], ms_inputs[3], ms_inputs[4])):
        np.testing.assert_array_equal(ms_value.asnumpy(), torch_value.cpu().numpy())


def test_npu_advance_step_flashattn_v2_mutates_speculative_state_like_torch_npu():
    num_seqs = 2
    spec_num = 3
    token_each_req = spec_num + 1
    block_size = 8
    input_tokens = np.arange(num_seqs * token_each_req, dtype=np.int64)
    sampled_token_ids = np.array([[11, 12, -1, -1], [21, 22, 23, -1]], dtype=np.int64)
    input_positions = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    seq_lens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    slot_mapping = np.zeros(num_seqs * token_each_req, dtype=np.int64)
    block_tables = np.arange(num_seqs * 16, dtype=np.int64).reshape(num_seqs, 16)
    spec_token = np.array([[31, 32, 33], [41, 42, 43]], dtype=np.int64)
    accepted_num = np.array([1, 2], dtype=np.int64)

    torch_inputs = [
        _torch_tensor(input_tokens),
        _torch_tensor(sampled_token_ids),
        _torch_tensor(input_positions),
        _torch_tensor(seq_lens),
        _torch_tensor(slot_mapping),
        _torch_tensor(block_tables),
        _torch_tensor(spec_token),
        _torch_tensor(accepted_num),
    ]
    ms_inputs = [
        _ms_tensor(input_tokens),
        _ms_tensor(sampled_token_ids),
        _ms_tensor(input_positions),
        _ms_tensor(seq_lens),
        _ms_tensor(slot_mapping),
        _ms_tensor(block_tables),
        _ms_tensor(spec_token),
        _ms_tensor(accepted_num),
    ]

    try:
        torch_npu.npu_advance_step_flashattn(
            torch_inputs[0],
            torch_inputs[1],
            torch_inputs[2],
            torch_inputs[3],
            torch_inputs[4],
            torch_inputs[5],
            num_seqs,
            num_seqs,
            block_size,
            spec_token=torch_inputs[6],
            accepted_num=torch_inputs[7],
        )
    except RuntimeError as exc:
        pytest.skip(f"torch_npu AdvanceStepV2 is unavailable on this host: {exc}")

    npu_advance_step_flashattn(
        ms_inputs[0],
        ms_inputs[1],
        ms_inputs[2],
        ms_inputs[3],
        ms_inputs[4],
        ms_inputs[5],
        num_seqs,
        num_seqs,
        block_size,
        ms_inputs[6],
        ms_inputs[7],
    )
    for torch_value, ms_value in zip((torch_inputs[0], torch_inputs[2], torch_inputs[3], torch_inputs[4]),
                                     (ms_inputs[0], ms_inputs[2], ms_inputs[3], ms_inputs[4])):
        np.testing.assert_array_equal(ms_value.asnumpy(), torch_value.cpu().numpy())
