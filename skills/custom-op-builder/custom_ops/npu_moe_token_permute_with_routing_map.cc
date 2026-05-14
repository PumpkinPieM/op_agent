#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_token_permute_with_routing_map(const ms::Tensor &tokens, const ms::Tensor &routing_map, const std::optional<ms::Tensor> &probs_opt = std::nullopt, const std::optional<int64_t> &num_out_tokens_opt = std::nullopt, bool drop_and_pad = false) {
  auto probs = probs_opt.value_or(ms::Tensor());
  auto num_out_tokens = num_out_tokens_opt.value_or(tokens.shape()[0]);
  std::vector<int64_t> token_shape = tokens.shape();
  if (!token_shape.empty()) token_shape[0] = num_out_tokens;
  auto permute_tokens = ms::Tensor(tokens.data_type(), token_shape);
  auto permute_probs = ms::Tensor(tokens.data_type(), std::vector<int64_t>{num_out_tokens});
  auto sorted_indices = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{num_out_tokens});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeTokenPermuteWithRoutingMap");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeTokenPermuteWithRoutingMap, tokens, routing_map, probs_opt, num_out_tokens, drop_and_pad, permute_tokens, permute_probs, sorted_indices));
  runner->Run({tokens, routing_map, probs}, {permute_tokens, permute_probs, sorted_indices});
  return {permute_tokens, permute_probs, sorted_indices};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_token_permute_with_routing_map", PYBOOST_CALLER(3, custom::npu_moe_token_permute_with_routing_map));
}
}  // namespace custom
