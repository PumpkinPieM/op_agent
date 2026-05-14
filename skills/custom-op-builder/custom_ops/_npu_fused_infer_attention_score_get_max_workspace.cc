#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> _npu_fused_infer_attention_score_get_max_workspace(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &pse_shift_opt, const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<std::vector<int64_t>> &actual_seq_lengths_opt, const std::optional<std::vector<int64_t>> &actual_seq_lengths_kv_opt, const std::optional<ms::Tensor> &dequant_scale1_opt, const std::optional<ms::Tensor> &quant_scale1_opt, const std::optional<ms::Tensor> &dequant_scale2_opt, const std::optional<ms::Tensor> &quant_scale2_opt, const std::optional<ms::Tensor> &quant_offset2_opt, const std::optional<ms::Tensor> &antiquant_scale_opt, const std::optional<ms::Tensor> &antiquant_offset_opt, const std::optional<ms::Tensor> &key_antiquant_scale_opt, const std::optional<ms::Tensor> &key_antiquant_offset_opt, const std::optional<ms::Tensor> &value_antiquant_scale_opt, const std::optional<ms::Tensor> &value_antiquant_offset_opt, const std::optional<ms::Tensor> &block_table_opt, const std::optional<ms::Tensor> &query_padding_size_opt, const std::optional<ms::Tensor> &kv_padding_size_opt, const std::optional<ms::Tensor> &key_shared_prefix_opt, const std::optional<ms::Tensor> &value_shared_prefix_opt, const std::optional<std::vector<int64_t>> &actual_shared_prefix_len_opt, const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt, const std::optional<ms::Tensor> &key_rope_antiquant_scale_opt, int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens, const std::string &input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {
  throw std::runtime_error("ACLNN GetMaxWorkspaceSize is not exposed safely through MindSpore CustomOpBuilder.");
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_fused_infer_attention_score_get_max_workspace", PYBOOST_CALLER(1, custom::_npu_fused_infer_attention_score_get_max_workspace));
}
}  // namespace custom
