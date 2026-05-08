#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_recurrent_gated_delta_rule_functional(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &beta, const ms::Tensor &g, const std::optional<ms::Tensor> &initial_state_opt = std::nullopt, bool output_final_state = false) {
  auto initial_state = initial_state_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(query.data_type(), query.shape());
  auto final_state = ms::Tensor(query.data_type(), query.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("RecurrentGatedDeltaRule");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRecurrentGatedDeltaRule, query, key, value, beta, g, initial_state_opt, output_final_state, out, final_state));
  runner->Run({query, key, value, beta, g, initial_state}, {out, final_state});
  return {out, final_state};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_recurrent_gated_delta_rule_functional", PYBOOST_CALLER(2, custom::npu_recurrent_gated_delta_rule_functional));
}
}  // namespace custom
