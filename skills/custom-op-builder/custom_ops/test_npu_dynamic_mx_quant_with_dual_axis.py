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
KERNEL_SOURCE = Path(__file__).with_name("npu_dynamic_mx_quant_with_dual_axis.cc")
HAS_TORCH_NPU_INTERFACE = hasattr(torch_npu, "npu_dynamic_mx_quant_with_dual_axis")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        if not HAS_TORCH_NPU_INTERFACE:
            pytest.skip("torch_npu on this host does not expose npu_dynamic_mx_quant_with_dual_axis")
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_dynamic_mx_quant_with_dual_axis_test",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def npu_dynamic_mx_quant_with_dual_axis(x, round_mode="rint", dst_type=296, scale_alg=0):
    return _ops().npu_dynamic_mx_quant_with_dual_axis(x, round_mode, dst_type, scale_alg)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _unsupported_host(exc):
    text = str(exc).lower()
    markers = ("not support", "unsupported", "not in", "current platform", "not be found")
    return any(marker in text for marker in markers)


def _to_torch_input(x_np, use_bf16=False):
    tensor = torch.from_numpy(x_np).npu()
    if use_bf16:
        tensor = tensor.to(torch.bfloat16)
    return tensor


def _to_ms_input(x_np, use_bf16=False):
    tensor = Tensor(x_np)
    if use_bf16:
        tensor = tensor.astype(ms.bfloat16)
    return tensor


def _torch_bytes(tensor):
    if tensor.dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return tensor.view(torch.uint8).cpu().numpy()
    return tensor.cpu().numpy()


def _ms_bytes(tensor):
    return tensor.asnumpy().view(np.uint8)


def _run_case(x_np, use_bf16, dst_type, expected_y_shape, expected_scale1_shape, expected_scale2_shape):
    try:
        expected = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            _to_torch_input(x_np, use_bf16),
            round_mode="rint",
            dst_type=dst_type,
            scale_alg=0,
        )
        actual = npu_dynamic_mx_quant_with_dual_axis(_to_ms_input(x_np, use_bf16), "rint", dst_type, 0)
        actual_np = [_ms_bytes(out) for out in actual]
        expected_np = [_torch_bytes(out) for out in expected]
    except RuntimeError as exc:
        if _unsupported_host(exc):
            pytest.skip(f"aclnnDynamicMxQuantWithDualAxis is not supported on this host: {exc}")
        raise

    assert len(actual) == 4
    assert tuple(actual[0].shape) == expected_y_shape
    assert tuple(actual[1].shape) == expected_scale1_shape
    assert tuple(actual[2].shape) == expected_y_shape
    assert tuple(actual[3].shape) == expected_scale2_shape
    for exp, act in zip(expected_np, actual_np):
        np.testing.assert_array_equal(exp, act)


@pytest.mark.parametrize("use_bf16", [False, True])
def test_npu_dynamic_mx_quant_with_dual_axis_fp4_matches_torch_npu(use_bf16):
    x = np.zeros((1, 2, 2), dtype=np.float16)
    _run_case(
        x,
        use_bf16=use_bf16,
        dst_type=296,
        expected_y_shape=(1, 2, 1),
        expected_scale1_shape=(1, 2, 1, 2),
        expected_scale2_shape=(1, 1, 2, 2),
    )


def test_npu_dynamic_mx_quant_with_dual_axis_fp8_shapes_and_values():
    x = np.arange(2 * 32 * 64, dtype=np.float16).reshape(2, 32, 64) / np.float16(128.0)
    _run_case(
        x,
        use_bf16=False,
        dst_type=23,
        expected_y_shape=(2, 32, 64),
        expected_scale1_shape=(2, 32, 1, 2),
        expected_scale2_shape=(2, 1, 64, 2),
    )
