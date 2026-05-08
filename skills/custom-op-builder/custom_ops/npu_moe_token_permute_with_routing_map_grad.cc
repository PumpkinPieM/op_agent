#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_token_permute_with_routing_map_grad(const ms::Tensor &permuted_token_out_grad, const std::optional<ms::Tensor> &probs_grad_opt, const ms::Tensor &sorted_indices, const ms::Tensor &routing_map, int64_t experts_num, int64_t tokens_num, bool drop_and_pad) {
  auto probs_grad = probs_grad_opt.value_or(ms::Tensor());
  auto tokens_grad = ms::Tensor(permuted_token_out_grad.data_type(), std::vector<int64_t>{tokens_num, permuted_token_out_grad.shape()[1]});
  auto probs_out_grad = ms::Tensor(permuted_token_out_grad.data_type(), std::vector<int64_t>{tokens_num, experts_num});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeTokenPermuteWithRoutingMapGrad");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeTokenPermuteWithRoutingMapGrad, permuted_token_out_grad, probs_grad_opt, sorted_indices, routing_map, experts_num, tokens_num, drop_and_pad, tokens_grad, probs_out_grad));
  runner->Run({permuted_token_out_grad, probs_grad, sorted_indices, routing_map}, {tokens_grad, probs_out_grad});
  return {tokens_grad, probs_out_grad};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_token_permute_with_routing_map_grad", PYBOOST_CALLER(2, custom::npu_moe_token_permute_with_routing_map_grad));
}
}  // namespace custom
