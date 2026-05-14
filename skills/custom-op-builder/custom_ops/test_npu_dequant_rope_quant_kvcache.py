import gc
import os
from pathlib import Path

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
KERNEL_SOURCE = Path(__file__).with_name("npu_dequant_rope_quant_kvcache.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_dequant_rope_quant_kvcache_test_{os.getpid()}",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_torch(arr, dtype=None):
    t = torch.from_numpy(np.array(arr, copy=True)).npu()
    return t.to(dtype) if dtype is not None else t


def _to_ms(arr, dtype=None):
    t = Tensor(np.array(arr, copy=True))
    return t.astype(dtype) if dtype is not None else t


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_outputs(expected, actual, kv_output):
    assert len(actual) == 5
    assert len(expected) == 5
    for index, (exp, act) in enumerate(zip(expected, actual)):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if index in (1, 2) and not kv_output:
            assert exp_np.size == 0
            assert act_np.size == 0
        elif exp_np.dtype.kind in "iu":
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=1e-3, atol=1e-3)


def _make_case(rank3, kv_output):
    rng = np.random.default_rng(1717 + int(rank3) * 10 + int(kv_output))
    b, s, q_heads, kv_heads, d = 2, 2, 1, 1, 64
    x_shape = (b, s, (q_heads + 2 * kv_heads) * d) if rank3 else (b, (q_heads + 2 * kv_heads) * d)
    cos_shape = (b, s, 1, d) if rank3 else (b, d)
    indices_shape = (b,)
    x = rng.integers(-8, 8, size=x_shape, dtype=np.int32)
    cos = rng.normal(0.0, 0.1, size=cos_shape).astype(np.float16)
    sin = rng.normal(0.0, 0.1, size=cos_shape).astype(np.float16)
    k_cache = np.zeros((b, 4, kv_heads, d), dtype=np.int8)
    v_cache = np.zeros((b, 4, kv_heads, d), dtype=np.int8)
    indices = np.arange(b, dtype=np.int32).reshape(indices_shape)
    scale_k = np.ones((kv_heads, d), dtype=np.float32)
    scale_v = np.ones((kv_heads, d), dtype=np.float32)
    weight_scale = np.ones((x_shape[-1],), dtype=np.float32)
    activation_scale = np.ones((b * s if rank3 else b,), dtype=np.float32)
    bias = np.zeros((x_shape[-1],), dtype=np.float32)
    return {
        "x": x,
        "cos": cos,
        "sin": sin,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "indices": indices,
        "scale_k": scale_k,
        "scale_v": scale_v,
        "size_splits": [q_heads * d, kv_heads * d, kv_heads * d],
        "weight_scale": weight_scale,
        "activation_scale": activation_scale,
        "bias": bias,
        "kv_output": kv_output,
    }


def npu_dequant_rope_quant_kvcache(case):
    return _ops().npu_dequant_rope_quant_kvcache(
        _to_ms(case["x"]),
        _to_ms(case["cos"]),
        _to_ms(case["sin"]),
        _to_ms(case["k_cache"]),
        _to_ms(case["v_cache"]),
        _to_ms(case["indices"]),
        _to_ms(case["scale_k"]),
        _to_ms(case["scale_v"]),
        case["size_splits"],
        None,
        None,
        _to_ms(case["weight_scale"]),
        _to_ms(case["activation_scale"]),
        _to_ms(case["bias"]),
        0,
        "BSND",
        case["kv_output"],
        "contiguous",
    )


@pytest.mark.parametrize("rank3,kv_output", [(False, False), (False, True), (True, True)])
def test_npu_dequant_rope_quant_kvcache_matches_torch_npu(rank3, kv_output):
    if not hasattr(torch_npu, "npu_dequant_rope_quant_kvcache"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    case = _make_case(rank3, kv_output)
    try:
        expected = torch_npu.npu_dequant_rope_quant_kvcache(
            _to_torch(case["x"]),
            _to_torch(case["cos"]),
            _to_torch(case["sin"]),
            _to_torch(case["k_cache"]),
            _to_torch(case["v_cache"]),
            _to_torch(case["indices"]),
            _to_torch(case["scale_k"]),
            _to_torch(case["scale_v"]),
            case["size_splits"],
            weight_scale=_to_torch(case["weight_scale"]),
            activation_scale=_to_torch(case["activation_scale"]),
            bias=_to_torch(case["bias"]),
            quant_mode=0,
            input_layout="BSND",
            kv_output=kv_output,
            cache_mode="contiguous",
        )
        actual = npu_dequant_rope_quant_kvcache(case)
        _assert_outputs(expected, actual, kv_output)
    except RuntimeError as exc:
        text = str(exc).lower()
        if "dequantropequantkvcache" in text and (
            "not in" in text or "not support" in text or "does not has any binary" in text
        ):
            pytest.skip(f"aclnnDequantRopeQuantKvcache is unavailable on this host: {exc}")
        raise
