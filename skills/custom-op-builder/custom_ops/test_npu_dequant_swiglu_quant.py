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
KERNEL_SOURCE = Path(__file__).with_name("npu_dequant_swiglu_quant.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_dequant_swiglu_quant_test_{os.getpid()}",
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


def _assert_outputs(expected, actual, compare_scale=True):
    assert len(actual) == 2
    assert len(expected) == 2
    pairs = zip(expected, actual) if compare_scale else zip(expected[:1], actual[:1])
    for exp, act in pairs:
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if exp_np.dtype.kind in "iu":
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=1e-3, atol=1e-3)


def _case(shape, quant_mode, activate_dim, swiglu_mode):
    rng = np.random.default_rng(9000 + len(shape) + shape[-1] + quant_mode + swiglu_mode)
    x = rng.integers(-32, 32, size=shape, dtype=np.int32)
    reduce_shape = shape[:-1]
    half = shape[activate_dim if activate_dim >= 0 else activate_dim + len(shape)] // 2
    q_shape = (half,) if quant_mode == 1 else (1,)
    if quant_mode == 1 and activate_dim == len(shape) - 1:
        q_shape = (shape[-1] // 2,)
    return {
        "x": x,
        "weight_scale": np.ones((shape[-1],), dtype=np.float32),
        "activation_scale": np.ones(reduce_shape, dtype=np.float32),
        "bias": np.zeros((shape[-1],), dtype=np.float32),
        "quant_scale": np.ones(q_shape, dtype=np.float32),
        "quant_offset": np.zeros(q_shape, dtype=np.float32),
        "group_index": None,
        "activate_left": False,
        "quant_mode": quant_mode,
        "dst_type": 2,
        "round_mode": 0,
        "activate_dim": activate_dim,
        "swiglu_mode": swiglu_mode,
        "clamp_limit": 7.0,
        "glu_alpha": 1.702,
        "glu_bias": 1.0,
    }


def npu_dequant_swiglu_quant(case):
    return _ops().npu_dequant_swiglu_quant(
        _to_ms(case["x"]),
        _to_ms(case["weight_scale"]),
        _to_ms(case["activation_scale"]),
        _to_ms(case["bias"]),
        _to_ms(case["quant_scale"]),
        _to_ms(case["quant_offset"]),
        None if case["group_index"] is None else _to_ms(case["group_index"]),
        case["activate_left"],
        case["quant_mode"],
        case["dst_type"],
        case["round_mode"],
        case["activate_dim"],
        case["swiglu_mode"],
        case["clamp_limit"],
        case["glu_alpha"],
        case["glu_bias"],
    )


@pytest.mark.parametrize(
    "shape,quant_mode,activate_dim,swiglu_mode",
    [
        ((2, 64), 0, -1, 0),
        ((2, 64), 1, -1, 0),
        ((2, 2, 64), 1, -1, 1),
    ],
)
def test_npu_dequant_swiglu_quant_matches_torch_npu(shape, quant_mode, activate_dim, swiglu_mode):
    if not hasattr(torch_npu, "npu_dequant_swiglu_quant"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    case = _case(shape, quant_mode, activate_dim, swiglu_mode)
    try:
        expected = torch_npu.npu_dequant_swiglu_quant(
            _to_torch(case["x"]),
            weight_scale=_to_torch(case["weight_scale"]),
            activation_scale=_to_torch(case["activation_scale"]),
            bias=_to_torch(case["bias"]),
            quant_scale=_to_torch(case["quant_scale"]),
            quant_offset=_to_torch(case["quant_offset"]),
            group_index=None,
            activate_left=case["activate_left"],
            quant_mode=case["quant_mode"],
            dst_type=case["dst_type"],
            round_mode=case["round_mode"],
            activate_dim=case["activate_dim"],
            swiglu_mode=case["swiglu_mode"],
            clamp_limit=case["clamp_limit"],
            glu_alpha=case["glu_alpha"],
            glu_bias=case["glu_bias"],
        )
        actual = npu_dequant_swiglu_quant(case)
        _assert_outputs(expected, actual, compare_scale=case["quant_mode"] == 1)
    except RuntimeError as exc:
        text = str(exc).lower()
        if "dequantswigluquant" in text and (
            "not in" in text or "not support" in text or "does not has any binary" in text
        ):
            pytest.skip(f"aclnnDequantSwigluQuant is unavailable on this host: {exc}")
        raise
