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
KERNEL_SOURCE = Path(__file__).with_name("npu_attention_update.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            f"custom_ops_npu_attention_update_test_{os.getpid()}",
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


def _to_torch(arr, dtype):
    t = torch.from_numpy(np.array(arr, copy=True)).npu()
    return t.to(dtype) if dtype is not None else t


def _to_ms(arr, dtype):
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


def _assert_close(expected, actual, rtol, atol, check_lse):
    assert len(actual) == 2
    np.testing.assert_allclose(_np_from_torch(expected[0]), _np_from_ms(actual[0]), rtol=rtol, atol=atol)
    if check_lse:
        np.testing.assert_allclose(_np_from_torch(expected[1]), _np_from_ms(actual[1]), rtol=1e-4, atol=1e-4)


def _case_inputs(n, head_dim, sp, dtype):
    rng = np.random.default_rng(20260517 + n + head_dim + sp)
    lse = [rng.normal(0.0, 0.5, size=(n,)).astype(np.float32) for _ in range(sp)]
    local_out = [rng.normal(0.0, 0.25, size=(n, head_dim)).astype(np.float32) for _ in range(sp)]
    torch_lse = [_to_torch(x, torch.float32) for x in lse]
    torch_local_out = [_to_torch(x, dtype) for x in local_out]
    ms_lse = [_to_ms(x, ms.float32) for x in lse]
    ms_dtype = {torch.float32: ms.float32, torch.float16: ms.float16, torch.bfloat16: ms.bfloat16}[dtype]
    ms_local_out = [_to_ms(x, ms_dtype) for x in local_out]
    return torch_lse, torch_local_out, ms_lse, ms_local_out


def npu_attention_update(lse, local_out, update_type):
    return _ops().npu_attention_update(lse, local_out, update_type)


@pytest.mark.parametrize(
    "n,head_dim,sp,update_type,dtype,rtol,atol",
    [
        (4, 8, 1, 0, torch.float32, 1e-4, 1e-4),
        (8, 32, 2, 1, torch.float32, 1e-4, 1e-4),
        (8, 64, 3, 1, torch.float16, 1e-3, 1e-3),
        (16, 128, 2, 0, torch.bfloat16, 4e-3, 4e-3),
    ],
)
def test_npu_attention_update_matches_torch_npu(n, head_dim, sp, update_type, dtype, rtol, atol):
    if not hasattr(torch_npu, "npu_attention_update"):
        pytest.skip("benchmark torch_npu API is unavailable on this host")
    torch_lse, torch_local_out, ms_lse, ms_local_out = _case_inputs(n, head_dim, sp, dtype)
    try:
        expected = torch_npu.npu_attention_update(torch_lse, torch_local_out, update_type)
        actual = npu_attention_update(ms_lse, ms_local_out, update_type)
        _assert_close(expected, actual, rtol, atol, update_type == 1)
    except RuntimeError as exc:
        if "aclnnattentionupdate" in str(exc).lower() and (
            "not in" in str(exc).lower() or "not support" in str(exc).lower() or "does not has any binary" in str(exc).lower()
        ):
            pytest.skip(f"aclnnAttentionUpdate is unavailable on this host: {exc}")
        raise
