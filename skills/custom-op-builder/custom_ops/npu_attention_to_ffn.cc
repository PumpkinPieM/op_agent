#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_attention_to_ffn(const ms::Tensor &x, const ms::Tensor &session_id, const ms::Tensor &micro_batch_id, const ms::Tensor &layer_id, const ms::Tensor &expert_ids, const ms::Tensor &expert_rank_table, const std::string &group, int64_t world_size, const std::vector<int64_t> &ffn_token_info_table_shape, const std::vector<int64_t> &ffn_token_data_shape, const std::vector<int64_t> &attn_token_info_table_shape, int64_t moe_expert_num, const std::optional<ms::Tensor> &scales_opt, const std::optional<ms::Tensor> &active_mask_opt, int64_t quant_mode, int64_t sync_flag, int64_t ffn_start_rank_id) {
  auto scales_value = scales_opt.value_or(ms::Tensor());
  auto active_mask_value = active_mask_opt.value_or(ms::Tensor());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AttentionToFFN");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAttentionToFFN, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, scales_opt, active_mask_opt, quant_mode, sync_flag, ffn_start_rank_id));
  runner->Run({x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, scales_value, active_mask_value}, {});
  return {};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_attention_to_ffn", PYBOOST_CALLER(0, custom::npu_attention_to_ffn));
}
}  // namespace custom
