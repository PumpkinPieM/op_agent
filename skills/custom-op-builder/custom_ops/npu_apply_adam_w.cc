#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_apply_adam_w(const ms::Tensor &var, const ms::Tensor &m, const ms::Tensor &v,
                                         const ms::Tensor &beta1_power, const ms::Tensor &beta2_power,
                                         const ms::Tensor &lr, const ms::Tensor &weight_decay,
                                         const ms::Tensor &beta1, const ms::Tensor &beta2,
                                         const ms::Tensor &epsilon, const ms::Tensor &grad,
                                         const std::optional<ms::Tensor> &max_grad_norm_opt,
                                         const std::optional<bool> &amsgrad_opt,
                                         const std::optional<bool> &maximize_opt) {
  auto max_grad_norm_value = max_grad_norm_opt.value_or(ms::Tensor());
  auto amsgrad = amsgrad_opt.value_or(false);
  auto maximize = maximize_opt.value_or(false);
  auto var_out = var;
  auto m_out = m;
  auto v_out = v;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ApplyAdamW");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyAdamW, var_out, m_out, v_out, beta1_power, beta2_power, lr,
                                          weight_decay, beta1, beta2, epsilon, grad, max_grad_norm_opt, amsgrad,
                                          maximize));
  runner->Run({var_out, m_out, v_out, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad,
               max_grad_norm_value},
              {var_out, m_out, v_out});
  return {var_out, m_out, v_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_apply_adam_w", PYBOOST_CALLER(3, custom::npu_apply_adam_w));
}
}  // namespace custom
