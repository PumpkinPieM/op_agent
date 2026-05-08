#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> _npu_fused_infer_attention_score_get_max_workspace(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &pse_shift_opt, const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<std::vector<int64_t>> &actual_seq_lengths_opt, const std::optional<std::vector<int64_t>> &actual_seq_lengths_kv_opt, const std::optional<ms::Tensor> &dequant_scale1_opt, const std::optional<ms::Tensor> &quant_scale1_opt, const std::optional<ms::Tensor> &dequant_scale2_opt, const std::optional<ms::Tensor> &quant_scale2_opt, const std::optional<ms::Tensor> &quant_offset2_opt, const std::optional<ms::Tensor> &antiquant_scale_opt, const std::optional<ms::Tensor> &antiquant_offset_opt, const std::optional<ms::Tensor> &key_antiquant_scale_opt, const std::optional<ms::Tensor> &key_antiquant_offset_opt, const std::optional<ms::Tensor> &value_antiquant_scale_opt, const std::optional<ms::Tensor> &value_antiquant_offset_opt, const std::optional<ms::Tensor> &block_table_opt, const std::optional<ms::Tensor> &query_padding_size_opt, const std::optional<ms::Tensor> &kv_padding_size_opt, const std::optional<ms::Tensor> &key_shared_prefix_opt, const std::optional<ms::Tensor> &value_shared_prefix_opt, const std::optional<std::vector<int64_t>> &actual_shared_prefix_len_opt, const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt, const std::optional<ms::Tensor> &key_rope_antiquant_scale_opt, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, const std::string &input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {
  auto pse_shift_value = pse_shift_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto actual_seq_lengths = actual_seq_lengths_opt.value_or(std::vector<int64_t>{});
  auto actual_seq_lengths_kv = actual_seq_lengths_kv_opt.value_or(std::vector<int64_t>{});
  auto dequant_scale1_value = dequant_scale1_opt.value_or(ms::Tensor());
  auto quant_scale1_value = quant_scale1_opt.value_or(ms::Tensor());
  auto dequant_scale2_value = dequant_scale2_opt.value_or(ms::Tensor());
  auto quant_scale2_value = quant_scale2_opt.value_or(ms::Tensor());
  auto quant_offset2_value = quant_offset2_opt.value_or(ms::Tensor());
  auto antiquant_scale_value = antiquant_scale_opt.value_or(ms::Tensor());
  auto antiquant_offset_value = antiquant_offset_opt.value_or(ms::Tensor());
  auto key_antiquant_scale_value = key_antiquant_scale_opt.value_or(ms::Tensor());
  auto key_antiquant_offset_value = key_antiquant_offset_opt.value_or(ms::Tensor());
  auto value_antiquant_scale_value = value_antiquant_scale_opt.value_or(ms::Tensor());
  auto value_antiquant_offset_value = value_antiquant_offset_opt.value_or(ms::Tensor());
  auto block_table_value = block_table_opt.value_or(ms::Tensor());
  auto query_padding_size_value = query_padding_size_opt.value_or(ms::Tensor());
  auto kv_padding_size_value = kv_padding_size_opt.value_or(ms::Tensor());
  auto key_shared_prefix_value = key_shared_prefix_opt.value_or(ms::Tensor());
  auto value_shared_prefix_value = value_shared_prefix_opt.value_or(ms::Tensor());
  auto actual_shared_prefix_len = actual_shared_prefix_len_opt.value_or(std::vector<int64_t>{});
  auto query_rope_value = query_rope_opt.value_or(ms::Tensor());
  auto key_rope_value = key_rope_opt.value_or(ms::Tensor());
  auto key_rope_antiquant_scale_value = key_rope_antiquant_scale_opt.value_or(ms::Tensor());
  auto base_shape = query.shape();
  auto base_dtype = query.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedInferAttentionScoreV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedInferAttentionScoreV3, query, key, value, pse_shift_opt, atten_mask_opt, actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1_opt, quant_scale1_opt, dequant_scale2_opt, quant_scale2_opt, quant_offset2_opt, antiquant_scale_opt, antiquant_offset_opt, key_antiquant_scale_opt, key_antiquant_offset_opt, value_antiquant_scale_opt, value_antiquant_offset_opt, block_table_opt, query_padding_size_opt, kv_padding_size_opt, key_shared_prefix_opt, value_shared_prefix_opt, actual_shared_prefix_len, query_rope_opt, key_rope_opt, key_rope_antiquant_scale_opt, num_heads, scale, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, key_antiquant_mode, value_antiquant_mode, softmax_lse_flag, out0));
  runner->Run({query, key, value, pse_shift_value, atten_mask_value, dequant_scale1_value, quant_scale1_value, dequant_scale2_value, quant_scale2_value, quant_offset2_value, antiquant_scale_value, antiquant_offset_value, key_antiquant_scale_value, key_antiquant_offset_value, value_antiquant_scale_value, value_antiquant_offset_value, block_table_value, query_padding_size_value, kv_padding_size_value, key_shared_prefix_value, value_shared_prefix_value, query_rope_value, key_rope_value, key_rope_antiquant_scale_value}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_fused_infer_attention_score_get_max_workspace", PYBOOST_CALLER(1, custom::_npu_fused_infer_attention_score_get_max_workspace));
}
}  // namespace custom
