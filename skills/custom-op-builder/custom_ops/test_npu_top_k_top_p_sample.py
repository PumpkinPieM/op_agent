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
KERNEL_SOURCE = Path(__file__).with_name("npu_top_k_top_p_sample.cc")


torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_top_k_top_p_sample_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_top_k_top_p_sample(
    logits,
    top_k,
    top_p,
    q=None,
    min_ps=None,
    eps=1e-8,
    is_need_logits=False,
    top_k_guess=32,
    ks_max=1024,
    input_is_logits=True,
    post_sample="qSample",
):
    return _custom_ops.npu_top_k_top_p_sample(
        logits, top_k, top_p, q, min_ps, eps, is_need_logits, top_k_guess, ks_max, input_is_logits, post_sample
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _torch_tensor(arr, dtype):
    return torch.from_numpy(np.array(arr, copy=True)).npu().to(dtype)


def _ms_tensor(arr, dtype):
    return Tensor(np.array(arr, copy=True)).astype(dtype)


def _np_from_torch(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().cpu().numpy()


def _np_from_ms(value):
    if value.dtype == ms.bfloat16:
        value = value.astype(ms.float32)
    return value.asnumpy()


def _assert_outputs_close(expected, actual, rtol=1e-3, atol=1e-3):
    assert len(expected) == len(actual)
    for exp, act in zip(expected, actual):
        exp_np = _np_from_torch(exp)
        act_np = _np_from_ms(act)
        assert exp_np.shape == act_np.shape
        if exp_np.dtype.kind in "iu":
            np.testing.assert_array_equal(exp_np, act_np)
        else:
            np.testing.assert_allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize(
    "dtype,np_dtype",
    [(torch.float32, np.float32), (torch.float16, np.float16), (torch.bfloat16, np.float32)],
)
@pytest.mark.parametrize("with_q", [False, True])
@pytest.mark.parametrize("with_min_ps", [False, True])
def test_npu_top_k_top_p_sample_matches_torch_npu(dtype, np_dtype, with_q, with_min_ps):
    logits = np.array([[0.1, 1.5, -0.2, 0.7, 0.3, 2.0, -1.0, 0.4], [1.0, 0.2, 0.8, -0.5, 1.4, 0.6, 0.1, 0.3]], np.float32)
    top_k = np.array([4, 3], np.int32)
    top_p = np.array([0.85, 0.9], np_dtype)
    q = np.linspace(0.2, 1.8, logits.size, dtype=np.float32).reshape(logits.shape) if with_q else None
    min_ps = np.array([0.05, 0.10], np_dtype) if with_min_ps else None

    expected = torch_npu.npu_top_k_top_p_sample(
        _torch_tensor(logits, dtype),
        _torch_tensor(top_k, torch.int32),
        _torch_tensor(top_p, dtype),
        q=None if q is None else _torch_tensor(q, torch.float32),
        eps=1e-8,
        is_need_logits=True,
        top_k_guess=4,
        min_ps=None if min_ps is None else _torch_tensor(min_ps, dtype),
        ks_max=8,
        input_is_logits=True,
        post_sample="qSample",
    )
    actual = npu_top_k_top_p_sample(
        _ms_tensor(logits, ms.bfloat16 if dtype == torch.bfloat16 else getattr(ms, str(dtype).split(".")[-1])),
        _ms_tensor(top_k, ms.int32),
        _ms_tensor(top_p, ms.bfloat16 if dtype == torch.bfloat16 else getattr(ms, str(dtype).split(".")[-1])),
        q=None if q is None else _ms_tensor(q, ms.float32),
        min_ps=None
        if min_ps is None
        else _ms_tensor(min_ps, ms.bfloat16 if dtype == torch.bfloat16 else getattr(ms, str(dtype).split(".")[-1])),
        is_need_logits=True,
        top_k_guess=4,
        ks_max=8,
        input_is_logits=True,
        post_sample="qSample",
    )
    _assert_outputs_close(expected, actual, rtol=3e-3, atol=3e-3)


def test_npu_top_k_top_p_sample_probability_input_matches_torch_npu():
    probs = np.array([[0.05, 0.25, 0.10, 0.20, 0.15, 0.25], [0.30, 0.05, 0.15, 0.10, 0.25, 0.15]], np.float32)
    top_k = np.array([0, 2], np.int32)
    top_p = np.array([1.0, 0.75], np.float32)

    expected = torch_npu.npu_top_k_top_p_sample(
        _torch_tensor(probs, torch.float32),
        _torch_tensor(top_k, torch.int32),
        _torch_tensor(top_p, torch.float32),
        is_need_logits=True,
        top_k_guess=3,
        ks_max=8,
        input_is_logits=False,
        post_sample="None",
    )
    actual = npu_top_k_top_p_sample(
        _ms_tensor(probs, ms.float32),
        _ms_tensor(top_k, ms.int32),
        _ms_tensor(top_p, ms.float32),
        is_need_logits=True,
        top_k_guess=3,
        ks_max=8,
        input_is_logits=False,
        post_sample="None",
    )
    _assert_outputs_close(expected, actual)
