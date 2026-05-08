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
KERNEL_SOURCE = Path(__file__).with_name("npu_mla_prolog_v3_functional.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
_custom_ops = ms.ops.CustomOpBuilder("custom_ops_npu_mla_prolog_v3_functional_test", [str(KERNEL_SOURCE)], backend="Ascend").load()
def npu_mla_prolog_v3_functional(*args):
    return _custom_ops.npu_mla_prolog_v3_functional(*args)
@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
def _pta(x):
    if x is None:
        return None
    t = torch.from_numpy(x).npu()
    return t

def _ms(x):
    if x is None:
        return None
    return Tensor(x)

def _np(x):
    if isinstance(x, (tuple, list)):
        return [_np(v) for v in x]
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return x.float().cpu().numpy()
        return x.cpu().numpy()
    if hasattr(x, "asnumpy"):
        if x.dtype == ms.bfloat16:
            return x.astype(ms.float32).asnumpy()
        return x.asnumpy()
    return np.asarray(x)

def _assert_close(expected, actual, rtol=1e-2, atol=1e-2):
    e=_np(expected); a=_np(actual)
    if not isinstance(e, list): e=[e]
    if not isinstance(a, list): a=[a]
    assert len(e)==len(a)
    for ev,av in zip(e,a):
        assert ev.shape == av.shape
        np.testing.assert_allclose(ev, av, rtol=rtol, atol=atol, equal_nan=True)
def _case():
    weight_uq_qr = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    weight_dkv_kr = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    rope_sin = np.ones((4,), dtype=np.float16)
    token_x = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float16)
    rope_cos = np.ones((4,), dtype=np.float16)
    kv_cache = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    rmsnorm_gamma_ckv = np.ones((4,), dtype=np.float16)
    rmsnorm_gamma_cq = np.ones((4,), dtype=np.float16)
    kr_cache = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    weight_dq = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    weight_uk = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float16)
    cache_index_opt = None
    dequant_scale_x_opt = None
    dequant_scale_w_dq_opt = None
    dequant_scale_w_uq_qr_opt = None
    dequant_scale_w_dkv_kr_opt = None
    quant_scale_ckv_opt = None
    quant_scale_ckr_opt = None
    smooth_scales_cq_opt = None
    actual_seq_len_opt = None
    k_nope_clip_alpha_opt = None
    rmsnorm_epsilon_cq = 1.0 if "scale" in "rmsnorm_epsilon_cq" else 1e-5
    rmsnorm_epsilon_ckv = 1.0 if "scale" in "rmsnorm_epsilon_ckv" else 1e-5
    cache_mode_opt = None
    query_norm_flag = False
    weight_quant_mode = 0
    kv_cache_quant_mode = 0
    query_quant_mode = 0
    ckvkr_repo_mode = 0
    quant_scale_repo_mode = 0
    tile_size = 1
    qc_qr_scale = 1.0 if "scale" in "qc_qr_scale" else 1e-5
    kc_scale = 1.0 if "scale" in "kc_scale" else 1e-5
    token_x_dtype_opt = None
    weight_dq_dtype_opt = None
    weight_uq_qr_dtype_opt = None
    weight_dkv_kr_dtype_opt = None
    kv_cache_dtype_opt = None
    return (_pta(token_x), _pta(weight_dq), _pta(weight_uq_qr), _pta(weight_uk), _pta(weight_dkv_kr), _pta(rmsnorm_gamma_cq), _pta(rmsnorm_gamma_ckv), _pta(rope_sin), _pta(rope_cos), _pta(kv_cache), _pta(kr_cache), _pta(cache_index_opt), _pta(dequant_scale_x_opt), _pta(dequant_scale_w_dq_opt), _pta(dequant_scale_w_uq_qr_opt), _pta(dequant_scale_w_dkv_kr_opt), _pta(quant_scale_ckv_opt), _pta(quant_scale_ckr_opt), _pta(smooth_scales_cq_opt), _pta(actual_seq_len_opt), _pta(k_nope_clip_alpha_opt), rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_opt, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, token_x_dtype_opt, weight_dq_dtype_opt, weight_uq_qr_dtype_opt, weight_dkv_kr_dtype_opt, kv_cache_dtype_opt), (_ms(token_x), _ms(weight_dq), _ms(weight_uq_qr), _ms(weight_uk), _ms(weight_dkv_kr), _ms(rmsnorm_gamma_cq), _ms(rmsnorm_gamma_ckv), _ms(rope_sin), _ms(rope_cos), _ms(kv_cache), _ms(kr_cache), _ms(cache_index_opt), _ms(dequant_scale_x_opt), _ms(dequant_scale_w_dq_opt), _ms(dequant_scale_w_uq_qr_opt), _ms(dequant_scale_w_dkv_kr_opt), _ms(quant_scale_ckv_opt), _ms(quant_scale_ckr_opt), _ms(smooth_scales_cq_opt), _ms(actual_seq_len_opt), _ms(k_nope_clip_alpha_opt), rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_opt, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, token_x_dtype_opt, weight_dq_dtype_opt, weight_uq_qr_dtype_opt, weight_dkv_kr_dtype_opt, kv_cache_dtype_opt)
def _torch_reference(torch_args):
    required_count = 11
    keyword_names = ['cache_index', 'dequant_scale_x', 'dequant_scale_w_dq', 'dequant_scale_w_uq_qr', 'dequant_scale_w_dkv_kr', 'quant_scale_ckv', 'quant_scale_ckr', 'smooth_scales_cq', 'actual_seq_len', 'k_nope_clip_alpha', 'rmsnorm_epsilon_cq', 'rmsnorm_epsilon_ckv', 'cache_mode', 'query_norm_flag', 'weight_quant_mode', 'kv_cache_quant_mode', 'query_quant_mode', 'ckvkr_repo_mode', 'quant_scale_repo_mode', 'tile_size', 'qc_qr_scale', 'kc_scale', 'token_x_dtype', 'weight_dq_dtype', 'weight_uq_qr_dtype', 'weight_dkv_kr_dtype', 'kv_cache_dtype']
    kwargs = {name: value for name, value in zip(keyword_names, torch_args[required_count:]) if value is not None}

    dtype_map = {5: torch.float16, 6: torch.float32, 27: torch.bfloat16}
    for key, value in list(kwargs.items()):
        if key.endswith("dtype") and isinstance(value, int) and value in dtype_map:
            kwargs[key] = dtype_map[value]
    return torch_npu.npu_mla_prolog_v3_functional(*torch_args[:required_count], **kwargs)

def test_npu_mla_prolog_v3_functional_against_torch_npu_benchmark():
    assert hasattr(torch_npu, "npu_mla_prolog_v3_functional")
    torch_args, ms_args = _case()
    expected = _torch_reference(torch_args)
    actual = npu_mla_prolog_v3_functional(*ms_args)
    _assert_close(expected, actual)
