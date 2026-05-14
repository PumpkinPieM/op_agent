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
KERNEL_SOURCE = Path(__file__).with_name("npu_scatter_pa_kv_cache.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)

_custom_ops = ms.ops.CustomOpBuilder(
    f"custom_ops_npu_scatter_pa_kv_cache_test_{os.getpid()}",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_scatter_pa_kv_cache(
    key, value, key_cache, value_cache, slot_mapping, compress_lens=None, compress_seq_offsets=None, seq_lens=None
):
    return _custom_ops.npu_scatter_pa_kv_cache(
        key, value, key_cache, value_cache, slot_mapping, compress_lens, compress_seq_offsets, seq_lens, None
    )


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_torch(arr):
    return torch.from_numpy(np.array(arr, copy=True)).npu()


def _to_ms(arr):
    return Tensor(np.array(arr, copy=True))


def _make_pa_nz_case(bs, num_heads, k_head_size, v_head_size, block_size, num_blocks):
    rng = np.random.default_rng(17 + bs + v_head_size + block_size)
    last_dim = 16
    key = rng.normal(size=(bs, num_heads, k_head_size)).astype(np.float16)
    value = rng.normal(size=(bs, num_heads, v_head_size)).astype(np.float16)
    key_cache = rng.normal(
        size=(num_blocks, num_heads * k_head_size // last_dim, block_size, last_dim)
    ).astype(np.float16)
    value_cache = rng.normal(
        size=(num_blocks, num_heads * v_head_size // last_dim, block_size, last_dim)
    ).astype(np.float16)
    slot_mapping = rng.choice(num_blocks * block_size, size=bs, replace=False).astype(np.int32)
    return key, value, key_cache, value_cache, slot_mapping


@pytest.mark.parametrize(
    "bs,num_heads,k_head_size,v_head_size,block_size,num_blocks",
    [(4, 2, 16, 16, 8, 2), (6, 2, 32, 64, 16, 2)],
)
def test_npu_scatter_pa_kv_cache_matches_torch_npu(bs, num_heads, k_head_size, v_head_size, block_size, num_blocks):
    key, value, key_cache, value_cache, slot_mapping = _make_pa_nz_case(
        bs, num_heads, k_head_size, v_head_size, block_size, num_blocks
    )

    torch_key_cache = _to_torch(key_cache)
    torch_value_cache = _to_torch(value_cache)
    torch_npu.npu_scatter_pa_kv_cache(
        _to_torch(key), _to_torch(value), torch_key_cache, torch_value_cache, _to_torch(slot_mapping)
    )

    ms_key_cache = _to_ms(key_cache)
    ms_value_cache = _to_ms(value_cache)
    actual_key_cache, actual_value_cache = npu_scatter_pa_kv_cache(
        _to_ms(key), _to_ms(value), ms_key_cache, ms_value_cache, _to_ms(slot_mapping)
    )

    np.testing.assert_allclose(torch_key_cache.cpu().numpy(), actual_key_cache.asnumpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(torch_value_cache.cpu().numpy(), actual_value_cache.asnumpy(), rtol=1e-3, atol=1e-3)
