#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_token_unpermute_with_routing_map_grad(const ms::Tensor &unpermuted_tokens_grad, const std::optional<ms::Tensor> &unpermuted_probs_grad_opt, const ms::Tensor &sorted_indices, const ms::Tensor &routing_map, int64_t num_out_tokens, bool drop_and_pad) {
  auto unpermuted_probs_grad = unpermuted_probs_grad_opt.value_or(ms::Tensor());
  auto permuted_tokens_grad = ms::Tensor(unpermuted_tokens_grad.data_type(), std::vector<int64_t>{num_out_tokens, unpermuted_tokens_grad.shape()[1]});
  auto probs_grad = ms::Tensor(unpermuted_tokens_grad.data_type(), std::vector<int64_t>{num_out_tokens});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeTokenUnpermuteWithRoutingMapGrad");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeTokenUnpermuteWithRoutingMapGrad, unpermuted_tokens_grad, unpermuted_probs_grad_opt, sorted_indices, routing_map, num_out_tokens, drop_and_pad, permuted_tokens_grad, probs_grad));
  runner->Run({unpermuted_tokens_grad, unpermuted_probs_grad, sorted_indices, routing_map}, {permuted_tokens_grad, probs_grad});
  return {permuted_tokens_grad, probs_grad};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_token_unpermute_with_routing_map_grad", PYBOOST_CALLER(2, custom::npu_moe_token_unpermute_with_routing_map_grad));
}
}  // namespace custom
