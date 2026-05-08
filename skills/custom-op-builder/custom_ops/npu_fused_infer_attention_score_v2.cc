#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_fused_infer_attention_score_v2(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt, const std::optional<ms::Tensor> &pse_shift_opt, const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt, const std::optional<std::vector<int64_t>> &actual_seq_kvlen_opt, const std::optional<ms::Tensor> &block_table_opt, const std::optional<ms::Tensor> &dequant_scale_query_opt, const std::optional<ms::Tensor> &dequant_scale_key_opt, const std::optional<ms::Tensor> &dequant_offset_key_opt, const std::optional<ms::Tensor> &dequant_scale_value_opt, const std::optional<ms::Tensor> &dequant_offset_value_opt, const std::optional<ms::Tensor> &dequant_scale_key_rope_opt, const std::optional<ms::Tensor> &quant_scale_out_opt, const std::optional<ms::Tensor> &quant_offset_out_opt, const std::optional<ms::Tensor> &quant_scale_p_opt, const std::optional<ms::Tensor> &learnable_sink_opt, int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens, const std::string &input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse, const std::optional<int64_t> &query_dtype_opt, const std::optional<int64_t> &key_dtype_opt, const std::optional<int64_t> &value_dtype_opt, const std::optional<int64_t> &query_rope_dtype_opt, const std::optional<int64_t> &key_rope_dtype_opt, const std::optional<int64_t> &key_shared_prefix_dtype_opt, const std::optional<int64_t> &value_shared_prefix_dtype_opt, const std::optional<int64_t> &dequant_scale_query_dtype_opt, const std::optional<int64_t> &dequant_scale_key_dtype_opt, const std::optional<int64_t> &dequant_scale_value_dtype_opt, const std::optional<int64_t> &dequant_scale_key_rope_dtype_opt, const std::optional<int64_t> &out_dtype_opt) {
  auto query_rope_value = query_rope_opt.value_or(ms::Tensor());
  auto key_rope_value = key_rope_opt.value_or(ms::Tensor());
  auto pse_shift_value = pse_shift_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto actual_seq_qlen = actual_seq_qlen_opt.value_or(std::vector<int64_t>{});
  auto actual_seq_kvlen = actual_seq_kvlen_opt.value_or(std::vector<int64_t>{});
  auto block_table_value = block_table_opt.value_or(ms::Tensor());
  auto dequant_scale_query_value = dequant_scale_query_opt.value_or(ms::Tensor());
  auto dequant_scale_key_value = dequant_scale_key_opt.value_or(ms::Tensor());
  auto dequant_offset_key_value = dequant_offset_key_opt.value_or(ms::Tensor());
  auto dequant_scale_value_value = dequant_scale_value_opt.value_or(ms::Tensor());
  auto dequant_offset_value_value = dequant_offset_value_opt.value_or(ms::Tensor());
  auto dequant_scale_key_rope_value = dequant_scale_key_rope_opt.value_or(ms::Tensor());
  auto quant_scale_out_value = quant_scale_out_opt.value_or(ms::Tensor());
  auto quant_offset_out_value = quant_offset_out_opt.value_or(ms::Tensor());
  auto quant_scale_p_value = quant_scale_p_opt.value_or(ms::Tensor());
  auto learnable_sink_value = learnable_sink_opt.value_or(ms::Tensor());
  auto query_dtype = query_dtype_opt.value_or(0);
  auto key_dtype = key_dtype_opt.value_or(0);
  auto value_dtype = value_dtype_opt.value_or(0);
  auto query_rope_dtype = query_rope_dtype_opt.value_or(0);
  auto key_rope_dtype = key_rope_dtype_opt.value_or(0);
  auto key_shared_prefix_dtype = key_shared_prefix_dtype_opt.value_or(0);
  auto value_shared_prefix_dtype = value_shared_prefix_dtype_opt.value_or(0);
  auto dequant_scale_query_dtype = dequant_scale_query_dtype_opt.value_or(0);
  auto dequant_scale_key_dtype = dequant_scale_key_dtype_opt.value_or(0);
  auto dequant_scale_value_dtype = dequant_scale_value_dtype_opt.value_or(0);
  auto dequant_scale_key_rope_dtype = dequant_scale_key_rope_dtype_opt.value_or(0);
  auto out_dtype = out_dtype_opt.value_or(0);
  auto base_shape = query.shape();
  auto base_dtype = query.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedInferAttentionScoreV4");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedInferAttentionScoreV4, query, key, value, query_rope_opt, key_rope_opt, pse_shift_opt, atten_mask_opt, actual_seq_qlen, actual_seq_kvlen, block_table_opt, dequant_scale_query_opt, dequant_scale_key_opt, dequant_offset_key_opt, dequant_scale_value_opt, dequant_offset_value_opt, dequant_scale_key_rope_opt, quant_scale_out_opt, quant_offset_out_opt, quant_scale_p_opt, learnable_sink_opt, num_query_heads, num_key_value_heads, softmax_scale, pre_tokens, next_tokens, input_layout, sparse_mode, block_size, query_quant_mode, key_quant_mode, value_quant_mode, inner_precise, return_softmax_lse, query_dtype, key_dtype, value_dtype, query_rope_dtype, key_rope_dtype, key_shared_prefix_dtype, value_shared_prefix_dtype, dequant_scale_query_dtype, dequant_scale_key_dtype, dequant_scale_value_dtype, dequant_scale_key_rope_dtype, out_dtype, out0, out1));
  runner->Run({query, key, value, query_rope_value, key_rope_value, pse_shift_value, atten_mask_value, block_table_value, dequant_scale_query_value, dequant_scale_key_value, dequant_offset_key_value, dequant_scale_value_value, dequant_offset_value_value, dequant_scale_key_rope_value, quant_scale_out_value, quant_offset_out_value, quant_scale_p_value, learnable_sink_value}, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fused_infer_attention_score_v2", PYBOOST_CALLER(2, custom::npu_fused_infer_attention_score_v2));
}
}  // namespace custom
