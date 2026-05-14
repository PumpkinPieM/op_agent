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
KERNEL_SOURCE = Path(__file__).with_name("npu_mla_prolog_v2.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = True
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_mla_prolog_v2_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_mla_prolog_v2(*args):
    return _custom_ops.npu_mla_prolog_v2(*args)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_to_ms(tensor):
    if tensor.dtype == torch.bfloat16:
        return Tensor(tensor.float().cpu().numpy(), ms.bfloat16)
    return Tensor(tensor.cpu().numpy())


def _ms_nz(tensor):
    return ms.ops.auto_generate.format_cast(_torch_to_ms(tensor), 29)


def _to_numpy(value):
    if isinstance(value, (tuple, list)):
        return [_to_numpy(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.float().cpu().numpy() if value.dtype == torch.bfloat16 else value.cpu().numpy()
    if value.dtype == ms.bfloat16:
        return value.astype(ms.float32).asnumpy()
    return value.asnumpy()


def _assert_close(expected, actual, rtol=4e-2, atol=4e-2):
    expected_np = _to_numpy(expected)
    actual_np = _to_numpy(actual)
    assert len(expected_np) == len(actual_np)
    for expected_item, actual_item in zip(expected_np, actual_np):
        assert expected_item.shape == actual_item.shape
        np.testing.assert_allclose(expected_item, actual_item, rtol=rtol, atol=atol, equal_nan=True)


def _assert_v2_outputs_close(expected, actual, rtol=4e-2, atol=4e-2):
    expected_np = _to_numpy(expected)
    actual_np = _to_numpy(actual)
    assert len(expected_np) == len(actual_np) == 5
    for expected_item, actual_item in zip(expected_np[:4], actual_np[:4]):
        assert expected_item.shape == actual_item.shape
        np.testing.assert_allclose(expected_item, actual_item, rtol=rtol, atol=atol, equal_nan=True)
    assert expected_np[4].shape == actual_np[4].shape
    assert actual_np[4].dtype == np.float32


def _case(bs_merged):
    torch.manual_seed(1)
    b, s, he, hcq, hckv, n, d, dr = 1, 1, 1024, 1536, 512, 1, 128, 64
    block_size, block_num, nkv = 16, 1, 1
    token_shape = (b * s, he) if bs_merged else (b, s, he)
    rope_shape = (b * s, dr) if bs_merged else (b, s, dr)
    cache_index_shape = (b * s,) if bs_merged else (b, s)

    def rand(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="npu")

    token_x = rand(token_shape)
    weight_dq = rand((he, hcq))
    weight_uq_qr = rand((hcq, n * (d + dr)))
    weight_uk = rand((n, d, hckv))
    weight_dkv_kr = rand((he, hckv + dr))
    rmsnorm_gamma_cq = rand((hcq,))
    rmsnorm_gamma_ckv = rand((hckv,))
    rope_sin = rand(rope_shape)
    rope_cos = rand(rope_shape)
    cache_index = torch.zeros(cache_index_shape, dtype=torch.int64, device="npu")
    kv_cache = rand((block_num, block_size, nkv, hckv))
    kr_cache = rand((block_num, block_size, nkv, dr))

    torch_args = (
        token_x,
        torch_npu.npu_format_cast(weight_dq.contiguous(), 29),
        torch_npu.npu_format_cast(weight_uq_qr.contiguous(), 29),
        weight_uk,
        torch_npu.npu_format_cast(weight_dkv_kr.contiguous(), 29),
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        cache_index,
        kv_cache.clone(),
        kr_cache.clone(),
    )
    ms_args = (
        _torch_to_ms(token_x),
        _ms_nz(weight_dq),
        _ms_nz(weight_uq_qr),
        _torch_to_ms(weight_uk),
        _ms_nz(weight_dkv_kr),
        _torch_to_ms(rmsnorm_gamma_cq),
        _torch_to_ms(rmsnorm_gamma_ckv),
        _torch_to_ms(rope_sin),
        _torch_to_ms(rope_cos),
        _torch_to_ms(cache_index),
        _torch_to_ms(kv_cache),
        _torch_to_ms(kr_cache),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1e-5,
        1e-5,
        "PA_BSND",
    )
    return torch_args, ms_args


@pytest.mark.parametrize("bs_merged", [False, True])
def test_npu_mla_prolog_v2_against_torch_npu_benchmark(bs_merged):
    if not hasattr(torch_npu, "npu_mla_prolog_v2"):
        pytest.skip("torch_npu.npu_mla_prolog_v2 is not available")
    torch_args, ms_args = _case(bs_merged)
    expected = torch_npu.npu_mla_prolog_v2(*torch_args, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5)
    actual = npu_mla_prolog_v2(*ms_args)
    _assert_v2_outputs_close(expected, actual)
