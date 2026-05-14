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
KERNEL_SOURCE = Path(__file__).with_name("npu_grouped_matmul_finalize_routing.cc")
_CUSTOM_OPS = None

torch_npu.npu.config.allow_internal_format = True


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        module_name = f"custom_ops_npu_grouped_matmul_finalize_routing_test_{os.getpid()}"
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(module_name, [str(KERNEL_SOURCE)], backend="Ascend").load()
    return _CUSTOM_OPS


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_torch(array):
    return torch.from_numpy(array).npu()


def _to_ms(array):
    return Tensor(array)


def _to_ms_nz(array):
    return ms.ops.auto_generate.format_cast(_to_ms(array), 29)


def _make_case(m=16, k=2048, n=7168, expert=1, group_list_type=1):
    rng = np.random.default_rng(20260516 + m + expert + group_list_type)
    x = rng.integers(-5, 5, size=(m, k), dtype=np.int8)
    weight = rng.integers(-5, 5, size=(expert, k, n), dtype=np.int8)
    scale = rng.uniform(0.001, 0.01, size=(expert, n)).astype(np.float32)
    pertoken_scale = rng.uniform(0.001, 0.01, size=(m,)).astype(np.float32)
    logit = np.ones((m,), dtype=np.float32)
    row_index = np.arange(m, dtype=np.int64)
    counts = np.asarray([m // expert] * expert, dtype=np.int64)
    group_list = counts if group_list_type == 1 else np.cumsum(counts)
    return x, weight, group_list, scale, pertoken_scale, logit, row_index, group_list_type


def npu_grouped_matmul_finalize_routing(
    x,
    w,
    group_list,
    scale=None,
    bias=None,
    offset=None,
    pertoken_scale=None,
    shared_input=None,
    logit=None,
    row_index=None,
    dtype=None,
    shared_input_weight=1.0,
    shared_input_offset=0,
    output_bs=0,
    group_list_type=1,
    tuning_config=None,
    x_dtype=None,
    w_dtype=None,
    scale_dtype=None,
    pertoken_scale_dtype=None,
):
    return _ops().npu_grouped_matmul_finalize_routing(
        x,
        w,
        group_list,
        scale,
        bias,
        offset,
        pertoken_scale,
        shared_input,
        logit,
        row_index,
        dtype,
        shared_input_weight,
        shared_input_offset,
        output_bs,
        group_list_type,
        tuning_config,
        x_dtype,
        w_dtype,
        scale_dtype,
        pertoken_scale_dtype,
    )


def test_custom_op_builds():
    assert hasattr(_ops(), "npu_grouped_matmul_finalize_routing")


@pytest.mark.parametrize(
    "case",
    [
        _make_case(m=16, expert=1, group_list_type=1),
        _make_case(m=16, expert=1, group_list_type=0),
    ],
)
def test_npu_grouped_matmul_finalize_routing_matches_torch_npu(case):
    if not hasattr(torch_npu, "npu_grouped_matmul_finalize_routing"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    x, weight, group_list, scale, pertoken_scale, logit, row_index, group_list_type = case
    torch_weight = torch_npu.npu_format_cast(_to_torch(weight).contiguous(), 29)
    expected = torch_npu.npu_grouped_matmul_finalize_routing(
        _to_torch(x),
        torch_weight,
        _to_torch(group_list),
        scale=_to_torch(scale),
        pertoken_scale=_to_torch(pertoken_scale),
        logit=_to_torch(logit),
        row_index=_to_torch(row_index),
        output_bs=x.shape[0],
        group_list_type=group_list_type,
    )
    actual = npu_grouped_matmul_finalize_routing(
        _to_ms(x),
        _to_ms_nz(weight),
        _to_ms(group_list),
        scale=_to_ms(scale),
        pertoken_scale=_to_ms(pertoken_scale),
        logit=_to_ms(logit),
        row_index=_to_ms(row_index),
        output_bs=x.shape[0],
        group_list_type=group_list_type,
    )
    actual = actual[0] if isinstance(actual, (list, tuple)) else actual
    expected_np = expected.detach().cpu().numpy()
    actual_np = actual.asnumpy()

    assert actual_np.shape == expected_np.shape
    assert actual_np.dtype == np.float32
    np.testing.assert_allclose(actual_np, expected_np, rtol=1e-3, atol=1e-3)
