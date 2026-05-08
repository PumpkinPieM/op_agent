#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_init_routing_v2(const ms::Tensor &x, const ms::Tensor &expert_idx, const std::optional<ms::Tensor> &scale_opt = std::nullopt, const std::optional<ms::Tensor> &expanded_x_out_opt = std::nullopt, const std::optional<int64_t> &active_num_opt = std::nullopt) {
  auto scale = scale_opt.value_or(ms::Tensor());
  auto expanded_x_out = expanded_x_out_opt.value_or(ms::Tensor());
  auto active_num = active_num_opt.value_or(x.shape()[0]);
  auto expanded_x = ms::Tensor(x.data_type(), x.shape());
  auto expanded_row_idx = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{active_num});
  auto expert_tokens_count = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{expert_idx.shape().empty() ? 1 : expert_idx.shape()[0]});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeInitRoutingV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeInitRoutingV2, x, expert_idx, scale_opt, active_num, expanded_x, expanded_row_idx, expert_tokens_count));
  runner->Run({x, expert_idx, scale, expanded_x_out}, {expanded_x, expanded_row_idx, expert_tokens_count});
  return {expanded_x, expanded_row_idx, expert_tokens_count};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_init_routing_v2", PYBOOST_CALLER(3, custom::npu_moe_init_routing_v2));
}
}  // namespace custom
