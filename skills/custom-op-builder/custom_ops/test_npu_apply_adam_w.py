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
KERNEL_SOURCE = Path(__file__).with_name("npu_apply_adam_w.cc")
_CUSTOM_OPS = None


def _ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        torch.npu.set_device(DEVICE_ID)
        torch.npu.set_compile_mode(jit_compile=False)
        context.set_context(device_target="Ascend", device_id=DEVICE_ID)
        context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
        _CUSTOM_OPS = ms.ops.CustomOpBuilder(
            "custom_ops_npu_apply_adam_w_test",
            [str(KERNEL_SOURCE)],
            backend="Ascend",
        ).load()
    return _CUSTOM_OPS


def _ms_tensor(array):
    return Tensor(np.array(array, copy=True))


def _ms_scalar(value):
    return Tensor(np.array([value], dtype=np.float32))


def _torch_tensor(array):
    return torch.from_numpy(np.array(array, copy=True)).npu()


def _cpu_apply_adam_w(var, m, v, grad, max_grad_norm, beta1_power, beta2_power, lr, weight_decay,
                      beta1, beta2, epsilon, amsgrad, maximize):
    gt = -grad if maximize else grad
    m_out = m * beta1 - (beta1 - 1.0) * gt
    v_out = v * beta2 - (beta2 - 1.0) * gt * gt
    var_t = var * (1.0 - lr * weight_decay)
    beta1_power_out = beta1_power * beta1
    beta2_power_out = beta2_power * beta2
    denom_src = np.maximum(max_grad_norm, v_out) if amsgrad else v_out
    denom = np.sqrt(denom_src / (1.0 - beta2_power_out)) + epsilon
    var_out = var_t + (-lr * m_out / (1.0 - beta1_power_out)) / denom
    return var_out.astype(np.float32), m_out.astype(np.float32), v_out.astype(np.float32)


def npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad,
                     max_grad_norm, amsgrad, maximize, out):
    var, m, v = out
    return _ops().npu_apply_adam_w(
        var,
        m,
        v,
        _ms_scalar(beta1_power),
        _ms_scalar(beta2_power),
        _ms_scalar(lr),
        _ms_scalar(weight_decay),
        _ms_scalar(beta1),
        _ms_scalar(beta2),
        _ms_scalar(epsilon),
        grad,
        max_grad_norm,
        amsgrad,
        maximize,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,maximize",
    [
        ((1024,), False),
        ((17, 19), True),
    ],
)
def test_npu_apply_adam_w_out_matches_torch_npu_and_formula(shape, maximize):
    rng = np.random.default_rng(20260517 + int(maximize))
    var = rng.uniform(10.0, 20.0, shape).astype(np.float32)
    m = rng.uniform(5.0, 10.0, shape).astype(np.float32)
    v = rng.uniform(0.1, 5.0, shape).astype(np.float32)
    grad = rng.uniform(-5.0, 5.0, shape).astype(np.float32)

    beta1_power = 0.12
    beta2_power = 0.23
    lr = 0.003
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-5
    amsgrad = False
    max_grad_norm = None

    expected_cpu = _cpu_apply_adam_w(
        var, m, v, grad, max_grad_norm, beta1_power, beta2_power, lr, weight_decay,
        beta1, beta2, epsilon, amsgrad, maximize,
    )
    expected_torch = torch_npu.npu_apply_adam_w(
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        _torch_tensor(grad),
        None,
        amsgrad,
        maximize,
        out=(_torch_tensor(var), _torch_tensor(m), _torch_tensor(v)),
    )
    actual = npu_apply_adam_w(
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        _ms_tensor(grad),
        None,
        amsgrad,
        maximize,
        out=(_ms_tensor(var), _ms_tensor(m), _ms_tensor(v)),
    )

    for cpu_value, torch_value, ms_value in zip(expected_cpu, expected_torch, actual):
        np.testing.assert_allclose(torch_value.detach().cpu().numpy(), cpu_value, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(ms_value.asnumpy(), torch_value.detach().cpu().numpy(), rtol=2e-3, atol=2e-3)
