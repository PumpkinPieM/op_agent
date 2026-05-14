#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_token_unpermute_with_routing_map_grad(
    const ms::Tensor &unpermuted_tokens_grad, const ms::Tensor &out_index, const ms::Tensor &permuted_token_id,
    const std::optional<ms::Tensor> &routing_map_opt, const std::optional<ms::Tensor> &permuted_tokens_opt,
    const std::optional<ms::Tensor> &probs_opt, bool drop_and_pad, const std::vector<int64_t> &restore_shape) {
  auto routing_map = routing_map_opt.value_or(ms::Tensor());
  auto permuted_tokens = permuted_tokens_opt.value_or(ms::Tensor());
  auto probs = probs_opt.value_or(ms::Tensor());
  auto permuted_tokens_grad =
    ms::Tensor(unpermuted_tokens_grad.data_type(), std::vector<int64_t>{out_index.shape()[0], unpermuted_tokens_grad.shape()[1]});
  auto probs_grad = probs_opt.has_value() ? ms::Tensor(probs_opt->data_type(), probs_opt->shape()) : ms::Tensor();
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeTokenUnpermuteWithRoutingMapGrad");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeTokenUnpermuteWithRoutingMapGrad, unpermuted_tokens_grad,
                                          out_index, permuted_token_id, routing_map_opt, permuted_tokens_opt,
                                          probs_opt, drop_and_pad, restore_shape, permuted_tokens_grad, probs_grad));
  runner->Run({unpermuted_tokens_grad, out_index, permuted_token_id, routing_map, permuted_tokens, probs},
              {permuted_tokens_grad, probs_grad});
  return {permuted_tokens_grad, probs_grad};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_token_unpermute_with_routing_map_grad", PYBOOST_CALLER(2, custom::npu_moe_token_unpermute_with_routing_map_grad));
}
}  // namespace custom
