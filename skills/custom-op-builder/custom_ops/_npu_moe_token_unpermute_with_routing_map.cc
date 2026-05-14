#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> _npu_moe_token_unpermute_with_routing_map(const ms::Tensor &permuted_tokens, const ms::Tensor &sorted_indices, const std::vector<int64_t> &restore_shape, const std::optional<ms::Tensor> &probs_opt = std::nullopt, const std::optional<ms::Tensor> &routing_map_opt = std::nullopt, bool drop_and_pad = false) {
  auto probs = probs_opt.value_or(ms::Tensor());
  auto routing_map = routing_map_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(permuted_tokens.data_type(), restore_shape);
  auto out_index = ms::Tensor(sorted_indices.data_type(), sorted_indices.shape());
  auto permuted_token_id = ms::Tensor(sorted_indices.data_type(), sorted_indices.shape());
  auto permute_probs = probs_opt.has_value() ? ms::Tensor(probs_opt->data_type(), sorted_indices.shape())
                                             : ms::Tensor(permuted_tokens.data_type(), std::vector<int64_t>{0});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeTokenUnpermuteWithRoutingMap");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeTokenUnpermuteWithRoutingMap, permuted_tokens, sorted_indices,
                                          routing_map_opt, probs_opt, drop_and_pad, restore_shape, out, out_index,
                                          permuted_token_id, permute_probs));
  runner->Run({permuted_tokens, sorted_indices, probs, routing_map}, {out, out_index, permuted_token_id, permute_probs});
  return {out, out_index, permuted_token_id, permute_probs};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_moe_token_unpermute_with_routing_map", PYBOOST_CALLER(4, custom::_npu_moe_token_unpermute_with_routing_map));
}
}  // namespace custom
