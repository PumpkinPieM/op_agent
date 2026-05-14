#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> _npu_fused_infer_attention_score_v2_get_max_workspace(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt, const std::optional<ms::Tensor> &pse_shift_opt, const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt, const std::optional<std::vector<int64_t>> &actual_seq_kvlen_opt, const std::optional<ms::Tensor> &block_table_opt, const std::optional<ms::Tensor> &dequant_scale_query_opt, const std::optional<ms::Tensor> &dequant_scale_key_opt, const std::optional<ms::Tensor> &dequant_offset_key_opt, const std::optional<ms::Tensor> &dequant_scale_value_opt, const std::optional<ms::Tensor> &dequant_offset_value_opt, const std::optional<ms::Tensor> &dequant_scale_key_rope_opt, const std::optional<ms::Tensor> &quant_scale_out_opt, const std::optional<ms::Tensor> &quant_offset_out_opt, const std::optional<ms::Tensor> &quant_scale_p_opt, const std::optional<ms::Tensor> &learnable_sink_opt, int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens, const std::string &input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse, const std::optional<int64_t> &query_dtype_opt, const std::optional<int64_t> &key_dtype_opt, const std::optional<int64_t> &value_dtype_opt, const std::optional<int64_t> &query_rope_dtype_opt, const std::optional<int64_t> &key_rope_dtype_opt, const std::optional<int64_t> &key_shared_prefix_dtype_opt, const std::optional<int64_t> &value_shared_prefix_dtype_opt, const std::optional<int64_t> &dequant_scale_query_dtype_opt, const std::optional<int64_t> &dequant_scale_key_dtype_opt, const std::optional<int64_t> &dequant_scale_value_dtype_opt, const std::optional<int64_t> &dequant_scale_key_rope_dtype_opt, const std::optional<int64_t> &out_dtype_opt) {
  throw std::runtime_error("ACLNN GetMaxWorkspaceSize is not exposed safely through MindSpore CustomOpBuilder.");
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_fused_infer_attention_score_v2_get_max_workspace", PYBOOST_CALLER(1, custom::_npu_fused_infer_attention_score_v2_get_max_workspace));
}
}  // namespace custom
