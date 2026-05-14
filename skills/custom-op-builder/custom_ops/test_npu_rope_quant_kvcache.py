import gc
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_rope_quant_kvcache.cc")


torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_rope_quant_kvcache_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_rope_quant_kvcache(
    x,
    cos,
    sin,
    k_cache,
    v_cache,
    indices,
    scale_k,
    scale_v,
    size_splits,
    offset_k=None,
    offset_v=None,
    quant_mode=0,
    input_layout="BSND",
    kv_output=False,
    cache_mode="contiguous",
):
    return _custom_ops.npu_rope_quant_kvcache(
        x,
        cos,
        sin,
        k_cache,
        v_cache,
        indices,
        scale_k,
        scale_v,
        size_splits,
        offset_k,
        offset_v,
        quant_mode,
        input_layout,
        kv_output,
        cache_mode,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_torch(arr, dtype=None):
    tensor = torch.from_numpy(np.array(arr, copy=True)).npu()
    return tensor.to(dtype) if dtype is not None else tensor


def _to_ms(arr, dtype=None):
    tensor = Tensor(np.array(arr, copy=True))
    return tensor.astype(dtype) if dtype is not None else tensor


def _np_from_torch(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _np_from_ms(tensor):
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.astype(ms.float32)
    return tensor.asnumpy()


def _make_case(dtype, q_heads, kv_output):
    rng = np.random.default_rng(31 + q_heads + int(kv_output))
    batch, seq, kv_heads, head_dim = 1, 1, 1, 128
    size_splits = [q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim]
    hidden = sum(size_splits)
    x = rng.normal(size=(batch, seq, hidden)).astype(np.float32)
    cos = rng.normal(size=(batch, seq, 1, head_dim)).astype(np.float32)
    sin = rng.normal(size=(batch, seq, 1, head_dim)).astype(np.float32)
    k_cache = rng.integers(0, 2, size=(batch, 2, kv_heads, head_dim), dtype=np.int8)
    v_cache = rng.integers(0, 2, size=(batch, 2, kv_heads, head_dim), dtype=np.int8)
    indices = np.array([0], dtype=np.int32)
    scale_k = rng.uniform(0.1, 0.4, size=(head_dim,)).astype(np.float32)
    scale_v = rng.uniform(0.1, 0.4, size=(head_dim,)).astype(np.float32)
    offset_k = rng.uniform(-0.2, 0.2, size=(head_dim,)).astype(np.float32)
    offset_v = rng.uniform(-0.2, 0.2, size=(head_dim,)).astype(np.float32)
    return {
        "x": x.astype(np.float16 if dtype == "float16" else np.float32),
        "cos": cos.astype(np.float16 if dtype == "float16" else np.float32),
        "sin": sin.astype(np.float16 if dtype == "float16" else np.float32),
        "k_cache": k_cache,
        "v_cache": v_cache,
        "indices": indices,
        "scale_k": scale_k,
        "scale_v": scale_v,
        "offset_k": offset_k,
        "offset_v": offset_v,
        "size_splits": size_splits,
        "kv_output": kv_output,
    }


@pytest.mark.parametrize("dtype,torch_dtype,ms_dtype", [("float16", torch.float16, ms.float16), ("bfloat16", torch.bfloat16, ms.bfloat16)])
@pytest.mark.parametrize("q_heads,kv_output", [(1, False), (2, True)])
def test_npu_rope_quant_kvcache_matches_torch_npu(dtype, torch_dtype, ms_dtype, q_heads, kv_output):
    case = _make_case(dtype, q_heads, kv_output)
    expected = torch_npu.npu_rope_quant_kvcache(
        _to_torch(case["x"], torch_dtype),
        _to_torch(case["cos"], torch_dtype),
        _to_torch(case["sin"], torch_dtype),
        _to_torch(case["k_cache"]),
        _to_torch(case["v_cache"]),
        _to_torch(case["indices"]),
        _to_torch(case["scale_k"]),
        _to_torch(case["scale_v"]),
        case["size_splits"],
        offset_k=_to_torch(case["offset_k"]),
        offset_v=_to_torch(case["offset_v"]),
        kv_output=kv_output,
    )
    actual = npu_rope_quant_kvcache(
        _to_ms(case["x"], ms_dtype),
        _to_ms(case["cos"], ms_dtype),
        _to_ms(case["sin"], ms_dtype),
        _to_ms(case["k_cache"]),
        _to_ms(case["v_cache"]),
        _to_ms(case["indices"]),
        _to_ms(case["scale_k"]),
        _to_ms(case["scale_v"]),
        case["size_splits"],
        _to_ms(case["offset_k"]),
        _to_ms(case["offset_v"]),
        0,
        "BSND",
        kv_output,
        "contiguous",
    )
    assert len(expected) == len(actual) == 5
    for index, (exp, act) in enumerate(zip(expected, actual)):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if index >= 3:
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=2e-2, atol=2e-2, equal_nan=True)
