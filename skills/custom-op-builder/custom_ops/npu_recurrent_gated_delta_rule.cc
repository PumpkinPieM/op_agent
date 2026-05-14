#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_recurrent_gated_delta_rule(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                                          const ms::Tensor &state,
                                          const std::optional<ms::Tensor> &beta_opt,
                                          double scale,
                                          const std::optional<ms::Tensor> &actual_seq_lengths_opt,
                                          const std::optional<ms::Tensor> &ssm_state_indices_opt,
                                          const std::optional<ms::Tensor> &num_accepted_tokens_opt,
                                          const std::optional<ms::Tensor> &g_opt,
                                          const std::optional<ms::Tensor> &gk_opt) {
  auto beta = beta_opt.value_or(ms::Tensor());
  auto actual_seq_lengths = actual_seq_lengths_opt.value_or(ms::Tensor());
  auto ssm_state_indices = ssm_state_indices_opt.value_or(ms::Tensor());
  auto num_accepted_tokens = num_accepted_tokens_opt.value_or(ms::Tensor());
  auto g = g_opt.value_or(ms::Tensor());
  auto gk = gk_opt.value_or(ms::Tensor());
  auto state_ref = state;

  auto value_shape = value.shape();
  auto out = ms::Tensor(ms::TypeId::kNumberTypeBFloat16, value_shape);
  float scale_value = static_cast<float>(scale);

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("RecurrentGatedDeltaRule");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRecurrentGatedDeltaRule, query, key, value, beta_opt, state_ref,
                                          actual_seq_lengths_opt, ssm_state_indices_opt, g_opt, gk_opt,
                                          num_accepted_tokens_opt, scale_value, out));
  runner->Run({query, key, value, beta, state_ref, actual_seq_lengths, ssm_state_indices, g, gk, num_accepted_tokens},
              {state_ref, out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_recurrent_gated_delta_rule", PYBOOST_CALLER(1, custom::npu_recurrent_gated_delta_rule));
}
}  // namespace custom
