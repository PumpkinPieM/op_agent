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
KERNEL_SOURCE = Path(__file__).with_name("npu_apply_rotary_pos_emb.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_apply_rotary_pos_emb_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_apply_rotary_pos_emb(query, key, cos, sin, layout="BSH", rotary_mode="half"):
    return _custom_ops.npu_apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


@pytest.mark.parametrize(
    "shape,cos_shape,layout,value_mode",
    [
        ((1, 4, 2, 64), (1, 4, 1, 64), "BSND", "zeros"),
        ((2, 8, 4, 128), (2, 8, 1, 128), "BSND", "normal"),
    ],
)
@pytest.mark.parametrize("rotary_mode", ["half", "quarter", "interleave"])
def test_npu_apply_rotary_pos_emb_matches_torch_npu(shape, cos_shape, layout, value_mode, rotary_mode):
    rng = np.random.default_rng(3)
    if value_mode == "zeros":
        query_np = np.zeros(shape, dtype=np.float16)
        key_np = np.zeros(shape, dtype=np.float16)
        cos_np = np.ones(cos_shape, dtype=np.float16)
        sin_np = np.zeros(cos_shape, dtype=np.float16)
    else:
        query_np = rng.normal(size=shape).astype(np.float16)
        key_np = rng.normal(size=shape).astype(np.float16)
        cos_np = rng.normal(size=cos_shape).astype(np.float16)
        sin_np = rng.normal(size=cos_shape).astype(np.float16)

    expected = torch_npu.npu_apply_rotary_pos_emb(
        torch.from_numpy(query_np).npu(),
        torch.from_numpy(key_np).npu(),
        torch.from_numpy(cos_np).npu(),
        torch.from_numpy(sin_np).npu(),
        layout,
        rotary_mode,
    )
    actual = npu_apply_rotary_pos_emb(Tensor(query_np), Tensor(key_np), Tensor(cos_np), Tensor(sin_np), layout, rotary_mode)

    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item.asnumpy(), expected_item.cpu().numpy(), rtol=1e-3, atol=1e-3)
