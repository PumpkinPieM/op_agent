#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_ffn_to_attention(const ms::Tensor &x, const ms::Tensor &session_ids, const ms::Tensor &micro_batch_ids, const ms::Tensor &token_ids, const ms::Tensor &expert_offsets, const ms::Tensor &actual_token_num, const std::string &group, int64_t world_size, const std::vector<int64_t> &token_info_table_shape, const std::vector<int64_t> &token_data_shape, const std::optional<ms::Tensor> &attn_rank_table_opt) {
  auto attn_rank_table_value = attn_rank_table_opt.value_or(ms::Tensor());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FFNToAttention");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFFNToAttention, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, group, world_size, token_info_table_shape, token_data_shape, attn_rank_table_opt));
  runner->Run({x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table_value}, {});
  return {};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_ffn_to_attention", PYBOOST_CALLER(0, custom::npu_ffn_to_attention));
}
}  // namespace custom
