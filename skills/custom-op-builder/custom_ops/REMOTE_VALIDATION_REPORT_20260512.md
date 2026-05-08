# Custom ACLNN Validation Report - 2026-05-12

Remote host: `ascend131`  
Environment: `source ~/env.sh && conda activate fanzhilan`  
Remote directory: `/home/pzh/custom_aclnn/custom_ops`

## Summary

- Generated/copied: 36 C++ adapters and 36 Python benchmark tests.
- Build status: all 36 generated C++ adapters compiled successfully through `CustomOpBuilder`.
- Latest full per-file rerun: 36 tests executed with benchmark comparison still present. No tests were skipped.
- Passing tests: 3.
- Failing tests: 33.

## Fixes Made In This Round

- `npu_quant_scatter.cc` and `npu_quant_scatter_.cc`: changed the ACLNN launch from the incorrect `aclnnInplaceQuantScatterV2` wrapper-style argument list to documented `aclnnInplaceQuantScatter(selfRef, indices, updates, quantScales, quantZeroPoints, axis, quantAxis, reduction)`.
- `test_npu_quant_scatter.py` and `test_npu_quant_scatter_.py`: restored benchmark-aligned parameters from torch_npu tests (`axis=-2`, `quant_axis=-1`, `reduce="update"`, int8 self, 3D scale/update tensors).
- `npu_weight_quant_batchmatmul.cc`: removed the torch_npu-only `weight_dtype` value from the `aclnnWeightQuantBatchMatmulV3` launch; ACLNN V3 docs do not include that argument.
- `test_npu_quant_conv2d.py`: aligned the test with both paths where the host exposes different ABI requirements: torch_npu fallback uses encoded uint64 scale/int32 bias, while MindSpore ACLNN requires float32 scale/float32 bias. The benchmark comparison remains.

## Current Per-Op Status

| Op | Code generation | Build status | Latest test status |
|---|---|---|---|
| npu_matmul_compress_dequant | generated | compiled | failed: torch_npu benchmark K-axis mismatch after fixing scale rank |
| npu_mla_prolog_v2 | generated | compiled | failed: benchmark requires 3D `weight_uk` |
| npu_mla_prolog_v3 | generated | compiled | failed: benchmark requires 3D `weight_uk` |
| npu_mla_prolog_v3_functional | generated | compiled | failed: benchmark requires 3D `weight_uk` |
| npu_mm_all_reduce_base | generated | compiled | failed: benchmark K mismatch |
| npu_mm_reduce_scatter_base | generated | compiled | failed: benchmark `world_size` must be one of supported distributed sizes |
| npu_moe_distribute_combine_add_rms_norm | generated | compiled | failed: benchmark requires 2D `x` and `expert_ids` |
| npu_moe_distribute_combine_setup | generated | compiled | failed: ACLNN symbol missing from remote `libopapi.so` |
| npu_moe_distribute_combine_teardown | generated | compiled | failed: benchmark requires 2D `expert_ids` |
| npu_moe_distribute_combine_v2 | generated | compiled | failed: benchmark requires 2D `x` and `expert_ids` |
| npu_moe_distribute_dispatch_setup | generated | compiled | failed: benchmark requires 2D `expert_ids` |
| npu_moe_distribute_dispatch_teardown | generated | compiled | failed: benchmark requires 2D `expert_ids` |
| npu_moe_distribute_dispatch_v2 | generated | compiled | failed: benchmark requires 2D `x` and `expert_ids` |
| npu_nsa_compress_attention | generated | compiled | failed: benchmark dimension error |
| npu_nsa_compress_attention_infer | generated | compiled | failed: benchmark dimension error |
| npu_nsa_compress_infer | generated | compiled | failed: benchmark requires `compressBlockSize` divisible by 16 |
| npu_nsa_select_attention_infer | generated | compiled | failed: benchmark dimension error |
| npu_qkv_rms_norm_rope_cache | generated | compiled | failed: ACLNN symbol missing from remote `libopapi.so` |
| npu_qkv_rms_norm_rope_cache_functional | generated | compiled | failed: ACLNN symbol missing from remote `libopapi.so` |
| npu_quant_all_reduce | generated | compiled | failed: benchmark `world_size` must be 2, 4, or 8 |
| npu_quant_conv2d | generated | compiled | failed: MindSpore ACLNN on this host requires 5D input, while torch_npu benchmark interface is 4D conv2d |
| npu_quant_fusion_attention | generated | compiled | failed: benchmark query rank must be 3D or 4D |
| npu_quant_grouped_matmul_dequant | generated | compiled | failed: benchmark dimension error |
| npu_quant_matmul | generated | compiled | failed: benchmark expects int8 quantized input, generated case still fp16 |
| npu_quant_matmul_all_to_all | generated | compiled | failed: benchmark `world_size` must be 2, 4, 8, or 16 |
| npu_quant_matmul_dequant | generated | compiled | failed: ACLNN reports Ascend910B support not implemented |
| npu_quant_matmul_gelu | generated | compiled | failed: benchmark requires A4W4 or A8W8 quantized inputs |
| npu_quant_matmul_reduce_sum | generated | compiled | failed: benchmark requires `x2` NZ format |
| npu_quant_mm_reduce_scatter | generated | compiled | failed: benchmark `world_size` must be supported distributed size |
| npu_quant_reduce_scatter | generated | compiled | failed: benchmark `world_size` must be 2, 4, or 8 |
| npu_quant_scatter | generated | compiled | passed |
| npu_quant_scatter_ | generated | compiled | passed |
| npu_rope_quant_kvcache | generated | compiled | failed: benchmark requires 4D cache tensor |
| npu_scatter_pa_kv_cache | generated | compiled | failed: benchmark requires 3D key tensor |
| npu_transpose_quant_batchmatmul | generated | compiled | failed: ACLNN reports only DAV_3510 supported |
| npu_weight_quant_batchmatmul | generated | compiled | passed |

## Raw Status Artifact

Remote latest status table: `/tmp/custom_aclnn_current_status.tsv` on `ascend131`.
Per-test logs: `/tmp/test_<op>.py.log` on `ascend131`.
