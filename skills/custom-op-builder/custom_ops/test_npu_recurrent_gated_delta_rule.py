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
KERNEL_SOURCE = Path(__file__).with_name("npu_recurrent_gated_delta_rule.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_recurrent_gated_delta_rule_test", [str(KERNEL_SOURCE)], backend="Ascend"
        ).load()
    return _CUSTOM_OPS


def npu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    return _ops().npu_recurrent_gated_delta_rule(
        query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g, gk
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _make_case(seed, batch, tokens_per_batch, nk, nv, dk, dv, with_gk, with_num_accepted):
    rng = np.random.default_rng(seed)
    actual_seq_lengths = np.full((batch,), tokens_per_batch, dtype=np.int32)
    total_tokens = int(actual_seq_lengths.sum())
    query = rng.normal(size=(total_tokens, nk, dk)).astype(np.float32)
    key = rng.normal(size=(total_tokens, nk, dk)).astype(np.float32)
    value = rng.normal(size=(total_tokens, nv, dv)).astype(np.float32)
    state = rng.normal(size=(total_tokens, nv, dv, dk)).astype(np.float32)
    beta = rng.random(size=(total_tokens, nv)).astype(np.float32)
    g = -rng.random(size=(total_tokens, nv)).astype(np.float32)
    gk = -rng.random(size=(total_tokens, nv, dk)).astype(np.float32) if with_gk else None
    ssm_state_indices = np.arange(total_tokens, dtype=np.int32)
    num_accepted_tokens = (
        rng.integers(1, tokens_per_batch + 1, size=(batch,), dtype=np.int32) if with_num_accepted else None
    )
    return {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "scale": float(dk ** -0.5),
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
    }


def _torch_bf16(array):
    return torch.from_numpy(np.array(array, copy=True)).npu().to(torch.bfloat16)


def _torch_f32(array):
    return torch.from_numpy(np.array(array, copy=True)).npu().to(torch.float32)


def _torch_i32(array):
    return torch.from_numpy(np.array(array, copy=True)).npu().to(torch.int32)


def _ms_bf16(array):
    return Tensor(np.array(array, copy=True)).astype(ms.bfloat16)


def _ms_f32(array):
    return Tensor(np.array(array, copy=True)).astype(ms.float32)


def _ms_i32(array):
    return Tensor(np.array(array, copy=True)).astype(ms.int32)


def _to_numpy(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


@pytest.mark.parametrize(
    "case",
    [
        _make_case(0, batch=1, tokens_per_batch=1, nk=1, nv=1, dk=16, dv=16, with_gk=False, with_num_accepted=False),
        _make_case(1, batch=2, tokens_per_batch=2, nk=2, nv=2, dk=32, dv=32, with_gk=False, with_num_accepted=True),
    ],
    ids=["minimal_no_gk", "multi_batch_with_num_accepted"],
)
def test_npu_recurrent_gated_delta_rule_matches_torch_npu(case):
    torch_state = _torch_bf16(case["state"])
    expected = torch_npu.npu_recurrent_gated_delta_rule(
        _torch_bf16(case["query"]),
        _torch_bf16(case["key"]),
        _torch_bf16(case["value"]),
        torch_state,
        beta=_torch_bf16(case["beta"]),
        scale=case["scale"],
        actual_seq_lengths=_torch_i32(case["actual_seq_lengths"]),
        ssm_state_indices=_torch_i32(case["ssm_state_indices"]),
        num_accepted_tokens=None if case["num_accepted_tokens"] is None else _torch_i32(case["num_accepted_tokens"]),
        g=_torch_f32(case["g"]),
        gk=None if case["gk"] is None else _torch_f32(case["gk"]),
    )
    actual = npu_recurrent_gated_delta_rule(
        _ms_bf16(case["query"]),
        _ms_bf16(case["key"]),
        _ms_bf16(case["value"]),
        _ms_bf16(case["state"]),
        _ms_bf16(case["beta"]),
        case["scale"],
        _ms_i32(case["actual_seq_lengths"]),
        _ms_i32(case["ssm_state_indices"]),
        None if case["num_accepted_tokens"] is None else _ms_i32(case["num_accepted_tokens"]),
        _ms_f32(case["g"]),
        None if case["gk"] is None else _ms_f32(case["gk"]),
    )

    actual_np = actual.astype(ms.float32).asnumpy()
    expected_np = _to_numpy(expected)
    assert expected_np.shape == actual_np.shape
    np.testing.assert_allclose(expected_np, actual_np, rtol=3e-2, atol=3e-2)
