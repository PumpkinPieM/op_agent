#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_apply_adam_w(double beta1_power, double beta2_power, double lr, double weight_decay, double beta1, double beta2, double epsilon, const ms::Tensor &grad, const std::optional<ms::Tensor> &max_grad_norm_opt, const std::optional<bool> &amsgrad_opt, const std::optional<bool> &maximize_opt) {
  auto max_grad_norm_value = max_grad_norm_opt.value_or(ms::Tensor());
  auto amsgrad = amsgrad_opt.value_or(false);
  auto maximize = maximize_opt.value_or(false);
  auto base_shape = grad.shape();
  auto base_dtype = grad.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ApplyAdamW");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyAdamW, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm_opt, amsgrad, maximize, out0, out1, out2));
  runner->Run({grad, max_grad_norm_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_apply_adam_w", PYBOOST_CALLER(3, custom::npu_apply_adam_w));
}
}  // namespace custom
