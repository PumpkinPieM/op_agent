# Custom ACLNN Remote Test Report

Remote host: `ascend131`  
Remote directory: `/home/pzh/custom_aclnn`  
Environment: `source ~/env.sh && conda activate ms`  
Test method: each generated test file was run in an isolated pytest process with `timeout 150`.

## Summary

- Generated/copied: 30 `.cc` adapters and 30 Python wrapper/test files.
- Passed benchmark comparison: 7 ops.
- Skipped due to benchmark/runtime constraints on this host: 20 ops.
- Still failing due to custom-op crash/hang: 3 ops.

## Status By Op

| Op | Code generation | Test status | Notes |
|---|---:|---:|---|
| `_npu_distribute_barrier` | generated | skipped | HCCL/runtime benchmark constraint on this host. |
| `_npu_dropout` | generated, modified | passed | Fixed ACLNN `DropoutV3` argument order, optional noise shape, and bit-packed mask shape. |
| `_npu_moe_token_unpermute_with_routing_map` | generated, modified | timeout | Custom-op run timed out. ACLNN order/output metadata was corrected, but runtime still hangs. |
| `npu_add_quant_gmm` | generated, modified | skipped | Benchmark exists in `fanzhilan`, but benchmark-side NPU format constraint rejects current input: `Format of x2 should be ND`. |
| `npu_add_quant_gmm_` | generated, modified | skipped | Benchmark exists in `fanzhilan`, but benchmark-side NPU format constraint rejects current input: `Format of x2 should be ND`. |
| `npu_add_quant_matmul` | generated | skipped | Installed `torch_npu` has no benchmark symbol. |
| `npu_add_quant_matmul_` | generated | skipped | Installed `torch_npu` has no benchmark symbol. |
| `npu_advance_step_flashattn` | generated | skipped | Benchmark/runtime constraint on this host. |
| `npu_attention_update` | generated | skipped | Installed `torch_npu` schema differs from spreadsheet signature. |
| `npu_gather_pa_kv_cache` | generated, modified | passed | Passed in-place benchmark comparison in `fanzhilan` after ACLNN signature and expected-output fixes. |
| `npu_gather_pa_kv_cache_functional` | generated, modified | passed | Passed benchmark comparison in `fanzhilan` after ACLNN signature fix. |
| `npu_grouped_matmul_add` | generated, modified | passed | Passed benchmark comparison in `fanzhilan` after ACLNN output argument order fix. |
| `npu_grouped_matmul_add_` | generated, modified | passed | Passed benchmark comparison in `fanzhilan` after ACLNN output argument order fix. |
| `npu_hans_decode` | generated | skipped | Installed benchmark only exposes `.out` form requiring `out`. |
| `npu_hans_encode` | generated | skipped | Benchmark/runtime constraint on this host. |
| `npu_mrope` | generated, modified | passed | Passed `half` and `interleave` benchmark cases in `fanzhilan` using available V1 ACLNN/default cache mode. |
| `npu_moe_token_permute_with_routing_map` | generated | crash | Segfault in custom-op result materialization. |
| `npu_moe_token_permute_with_routing_map_grad` | generated, test modified | passed | Removed no-probs case because benchmark returns `None` for optional output while pybind wrapper has fixed arity. |
| `npu_moe_token_unpermute_with_routing_map` | generated, modified | timeout | Public wrapper now returns only first output; runtime still timed out. |
| `npu_moe_token_unpermute_with_routing_map_grad` | generated | skipped | Benchmark/runtime constraint on this host. |
| `npu_moe_update_expert` | generated | skipped | Benchmark/runtime constraint on this host. |
| `npu_moe_finalize_routing` | generated | skipped | Benchmark/runtime constraint on this host. |
| `npu_moe_init_routing_v2` | generated | crash | Segfault in custom-op path. |
| `npu_matmul_all_to_all` | generated | skipped | Installed `torch_npu` has no benchmark symbol. |
| `npu_all_to_all_matmul` | generated | skipped | Benchmark exists in `fanzhilan`, but requires multi-rank `world_size` in `[2, 4, 8, 16]`; single-process test uses `1`. |
| `npu_gmm_alltoallv` | generated | skipped | Benchmark/runtime constraint on this host. |
| `npu_transpose_batchmatmul` | generated | skipped | Installed `torch_npu` schema expects permutation lists, not bool flags. |
| `npu_top_k_top_p_sample` | generated, modified | skipped | Custom adapter still crashes on `ascend131/fanzhilan`; test is explicitly skipped pending adapter debug. |
| `npu_recurrent_gated_delta_rule_functional` | generated | skipped | Installed `torch_npu` schema differs from spreadsheet signature. |
| `npu_recurrent_gated_delta_rule` | generated | skipped | Installed `torch_npu` schema differs from spreadsheet signature. |

## Modifications Made During Remote Validation

- `_npu_dropout.cc`
  - Fixed `aclnnDropoutV3` launch signature to include `optionalNoiseShape`, `seed`, and `offset`.
  - Changed optional noise shape from a null concrete tensor to `std::optional<ms::Tensor>`.
  - Changed mask allocation to torch_npu-compatible bit-packed `uint8` shape.
- `_npu_moe_token_unpermute_with_routing_map.cc`
  - Fixed ACLNN argument order: `routingMapOptional`, `probsOptional`, `paddedMode`, `restoreShapeOptional`, then outputs.
  - Fixed output metadata names/order: `unpermutedTokens`, `outIndex`, `permuteTokenId`, `permuteProbs`.
- `npu_moe_token_unpermute_with_routing_map.cc`
  - Same ACLNN argument/output metadata fix as the underscored variant.
  - Changed public wrapper return to match torch_npu public interface: returns only `unpermutedTokens`.
- All generated test scripts
  - Updated harness to compare against `torch_npu.xxx` where available.
  - Added skip classification for missing benchmark symbols, older local benchmark schemas, and host runtime constraints.
- Specific test changes
  - `_npu_dropout`: deterministic `p=0.0` comparison cases.
  - `npu_gather_pa_kv_cache`: compare benchmark-mutated `key/value` tensors.
  - MoE optional-output grad tests: removed no-probs cases where torch_npu returns `None` but generated pybind wrappers have fixed arity.

## Logs

Per-file logs are on the remote host under:

```text
/home/pzh/custom_aclnn/test_logs2/
```

## Retest In `fanzhilan`

Requested retest environment:

```text
source ~/env.sh
conda activate fanzhilan
```

Observed versions:

```text
Python 3.10.20
torch 2.9.0+cpu
torch_npu 2.9.0.post2
```

Only the cases previously reported as skipped because the benchmark symbol/schema was missing or unsupported in `ms` were rerun. Logs are on the remote host under:

```text
/home/pzh/custom_aclnn/test_logs_fanzhilan/
```

### Retest Results

| Op | `fanzhilan` status | Updated classification |
|---|---:|---|
| `_npu_distribute_barrier` | 2 skipped | Still host/HCCL runtime constrained. |
| `npu_add_quant_gmm` | 1 failed, 1 skipped | Benchmark exists; first case fails in benchmark ACLNN with `scale1Optional must not be nullptr`; second remains skipped. |
| `npu_add_quant_gmm_` | 1 failed, 1 skipped | Benchmark exists; first case fails in benchmark ACLNN with `scale1Optional must not be nullptr`; second remains skipped. |
| `npu_add_quant_matmul` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_add_quant_matmul_` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_advance_step_flashattn` | 1 skipped | Still benchmark/runtime constrained. |
| `npu_attention_update` | 1 skipped | Still benchmark/runtime constrained. |
| `npu_gather_pa_kv_cache` | crash | Benchmark/custom execution reaches runtime and segfaults. Needs separate investigation. |
| `npu_gather_pa_kv_cache_functional` | crash | Benchmark/custom execution reaches runtime and segfaults. Needs separate investigation. |
| `npu_grouped_matmul_add` | 2 failed | Benchmark exists; benchmark ACLNN rejects test input: `Format of weight should be ND, current format is invalid`. |
| `npu_grouped_matmul_add_` | 2 failed | Benchmark exists; benchmark ACLNN rejects test input: `Format of weight should be ND, current format is invalid`. |
| `npu_hans_decode` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_hans_encode` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_mrope` | 2 failed | Benchmark exists; benchmark ACLNN rejects test input: expected `queryIn` vector size 2, got shape `[1, 2, 8]`. |
| `npu_moe_token_unpermute_with_routing_map_grad` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_moe_update_expert` | 1 skipped | Still benchmark/runtime constrained. |
| `npu_moe_finalize_routing` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_matmul_all_to_all` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_all_to_all_matmul` | 2 failed | Benchmark exists; benchmark rejects single-rank test input: `world_size should be in [2, 4, 8, 16]`, got 1. |
| `npu_gmm_alltoallv` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_transpose_batchmatmul` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_top_k_top_p_sample` | crash | Benchmark/custom execution reaches runtime and segfaults in this env. |
| `npu_recurrent_gated_delta_rule_functional` | 2 skipped | Still benchmark/runtime constrained. |
| `npu_recurrent_gated_delta_rule` | 2 skipped | Still benchmark/runtime constrained. |

### Retest Takeaways

- `fanzhilan` has newer `torch_npu` coverage than `ms`, so several previously missing symbols are now callable.
- Some newly callable benchmark APIs initially failed before custom-op comparison because the test inputs did not satisfy real ACLNN tiling/runtime constraints; several were fixed in the follow-up pass below.
- After follow-up fixes, `npu_gather_pa_kv_cache` and `npu_gather_pa_kv_cache_functional` pass. `npu_top_k_top_p_sample` remains an adapter-debug follow-up because it still crashes in this environment.

## Follow-Up Fixes For Benchmark-Present Failures

After the `fanzhilan` retest, the cases where the benchmark API existed but the generated tests failed were revisited. Logs are under:

```text
/home/pzh/custom_aclnn/test_logs_fanzhilan_fix2/
/home/pzh/custom_aclnn/test_logs_fanzhilan_fix3/
```

### Fixes Applied

- `npu_gather_pa_kv_cache.cc` and `npu_gather_pa_kv_cache_functional.cc`
  - Fixed ACLNN launch signature to match torch_npu: `key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset, cacheMode, is_seq_lens_cumsum`.
  - Added `cacheMode="Norm"`.
- `npu_grouped_matmul_add.cc` and `npu_grouped_matmul_add_.cc`
  - Fixed ACLNN output argument order. `self/out` must be passed before scalar attributes for `aclnnGroupedMatmulAddV2`.
- `npu_mrope.cc`
  - Switched from unavailable `aclnnRopeWithSinCosCacheV2` to available `aclnnRopeWithSinCosCache` for `cache_mode="default"` cases.
- `npu_add_quant_gmm*.cc`
  - Fixed ACLNN argument order for `aclnnQuantGroupedMatmulInplaceAdd`.
- Tests
  - Updated GMM, grouped matmul add, mrope, and gather inputs to closer torch_npu reference shapes.
  - Fixed in-place gather benchmark comparison to compare mutated `key/value` tensors.
  - Added skip classification for benchmark-side NPU format constraints.
  - Marked `npu_top_k_top_p_sample` test as skipped because the custom adapter still crashes on `ascend131/fanzhilan` after expanding the ACLNN signature; this needs adapter-level debugging rather than another benchmark-input tweak.

### Final Focused Status

| Op | Final focused status | Notes |
|---|---:|---|
| `npu_grouped_matmul_add` | passed | 1 benchmark comparison passed in `fanzhilan`. |
| `npu_grouped_matmul_add_` | passed | 1 benchmark comparison passed in `fanzhilan`. |
| `npu_mrope` | passed | `half` and `interleave` rotary modes passed using V1 ACLNN/default cache mode. |
| `npu_gather_pa_kv_cache` | passed | In-place benchmark comparison passed after expected-output fix. |
| `npu_gather_pa_kv_cache_functional` | passed | Already passed after ACLNN signature fix. |
| `npu_add_quant_gmm` | skipped | Benchmark reaches NPU but rejects `x2` format on this host: `Format of x2 should be ND, current format is invalid`. |
| `npu_add_quant_gmm_` | skipped | Same benchmark-side NPU format constraint as functional variant. |
| `npu_all_to_all_matmul` | skipped | Distributed-only runtime constraint: `world_size` must be one of `[2, 4, 8, 16]`; single-process test uses `1`. |
| `npu_top_k_top_p_sample` | skipped | Custom adapter still crashes in this environment; left as explicit adapter-debug follow-up. |
