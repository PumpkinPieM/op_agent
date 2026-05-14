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
KERNEL_SOURCE = Path(__file__).with_name("_npu_moe_token_unpermute_with_routing_map.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops__npu_moe_token_unpermute_with_routing_map_test_v5",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def _npu_moe_token_unpermute_with_routing_map(
    permuted_tokens,
    sorted_indices,
    restore_shape,
    probs,
    routing_map,
    drop_and_pad,
):
    return _ops()._npu_moe_token_unpermute_with_routing_map(
        permuted_tokens,
        sorted_indices,
        restore_shape,
        probs,
        routing_map,
        drop_and_pad,
    )


def _torch_tensor(array):
    return torch.from_numpy(np.array(array, copy=True)).npu()


def _ms_tensor(array):
    return Tensor(np.array(array, copy=True))


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


def _assert_selected_outputs_equal(expected, actual, indices):
    assert len(expected) == len(actual)
    for index in indices:
        exp = expected[index]
        act = actual[index]
        if exp is None:
            continue
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        np.testing.assert_array_equal(exp_np, act_np)


def _routing_map(dtype):
    data = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    if dtype == "bool":
        return data.astype(np.bool_)
    return data


def _sorted_indices(routing_map, drop_and_pad):
    routing_map_t = torch.from_numpy(routing_map.astype(np.bool_)).T.contiguous()
    token_num, expert_num = routing_map.shape
    if drop_and_pad:
        capacity = token_num
        return (
            routing_map_t.to(torch.int8)
            .argsort(dim=-1, descending=True, stable=True)[:, :capacity]
            .to(torch.int32)
            .contiguous()
            .view(-1)
            .numpy()
        )
    token_indices = torch.arange(token_num).unsqueeze(0).expand(expert_num, -1)
    sorted_indices_tmp = token_indices.masked_select(routing_map_t)
    return torch.sort(sorted_indices_tmp.float(), stable=True)[1].to(torch.int32).numpy()


def _case_inputs(drop_and_pad, with_probs, routing_map_dtype, token_dtype):
    routing_map = _routing_map(routing_map_dtype)
    sorted_indices = _sorted_indices(routing_map, drop_and_pad)
    token_num, expert_num = routing_map.shape
    hidden_size = 5
    top_k = int(routing_map.sum(axis=1)[0])
    rows = expert_num * token_num if drop_and_pad else token_num * top_k
    values = np.arange(rows * hidden_size, dtype=np.float32).reshape(rows, hidden_size) / 10.0
    permuted_tokens = values.astype(token_dtype)
    probs = None
    if with_probs:
        probs = (
            np.linspace(0.2, 1.4, token_num * expert_num, dtype=np.float32)
            .reshape(token_num, expert_num)
            .astype(token_dtype)
        )
    return permuted_tokens, sorted_indices, [token_num, hidden_size], probs, routing_map


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize("drop_and_pad", [False, True])
@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("routing_map_dtype", ["int8", "bool"])
@pytest.mark.parametrize("token_dtype", [np.float16, np.float32])
def test__npu_moe_token_unpermute_with_routing_map_matches_torch_npu(
    drop_and_pad,
    with_probs,
    routing_map_dtype,
    token_dtype,
):
    if not hasattr(torch_npu, "_npu_moe_token_unpermute_with_routing_map"):
        pytest.skip("torch_npu._npu_moe_token_unpermute_with_routing_map is not available in this environment")

    permuted_tokens, sorted_indices, restore_shape, probs, routing_map = _case_inputs(
        drop_and_pad,
        with_probs,
        routing_map_dtype,
        token_dtype,
    )
    torch_probs = _torch_tensor(probs) if probs is not None else None
    ms_probs = _ms_tensor(probs) if probs is not None else None

    expected = torch_npu._npu_moe_token_unpermute_with_routing_map(
        _torch_tensor(permuted_tokens),
        _torch_tensor(sorted_indices),
        restore_shape,
        probs=torch_probs,
        routing_map=_torch_tensor(routing_map),
        drop_and_pad=drop_and_pad,
    )
    actual = _npu_moe_token_unpermute_with_routing_map(
        _ms_tensor(permuted_tokens),
        _ms_tensor(sorted_indices),
        restore_shape,
        ms_probs,
        _ms_tensor(routing_map),
        drop_and_pad,
    )
    if drop_and_pad:
        output_indices = range(4)
    else:
        output_indices = [0, 3] if with_probs else [0]
    _assert_selected_outputs_equal(expected, actual, output_indices)


def test__npu_moe_token_unpermute_with_routing_map_allows_absent_optional_tensors():
    permuted_tokens, sorted_indices, restore_shape, _, _ = _case_inputs(
        False,
        False,
        "int8",
        np.float16,
    )
    expected = torch_npu._npu_moe_token_unpermute_with_routing_map(
        _torch_tensor(permuted_tokens),
        _torch_tensor(sorted_indices),
        restore_shape,
        probs=None,
        routing_map=None,
        drop_and_pad=False,
    )
    actual = _npu_moe_token_unpermute_with_routing_map(
        _ms_tensor(permuted_tokens),
        _ms_tensor(sorted_indices),
        restore_shape,
        None,
        None,
        False,
    )
    _assert_selected_outputs_equal(expected, actual, [0])
