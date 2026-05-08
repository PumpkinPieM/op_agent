#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_group_norm_swish_grad(const ms::Tensor &grad, const ms::Tensor &input,
                                                  int64_t num_groups, const ms::Tensor &weight,
                                                  const ms::Tensor &bias, const ms::Tensor &mean,
                                                  const ms::Tensor &rstd,
                                                  const std::vector<bool> &grad_input_mask,
                                                  const std::optional<double> &swish_scale_opt = std::nullopt) {
  auto grad_x = ms::Tensor(input.data_type(), input.shape());
  auto grad_weight = ms::Tensor(weight.data_type(), weight.shape());
  auto grad_bias = ms::Tensor(bias.data_type(), bias.shape());
  auto data_format = const_cast<char *>("NCHW");
  auto swish_scale = swish_scale_opt.value_or(1.0);
  bool dgamma_required = grad_input_mask.size() > 1 ? grad_input_mask[1] : true;
  bool dbeta_required = grad_input_mask.size() > 2 ? grad_input_mask[2] : true;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupNormSwishGrad");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupNormSwishGrad, grad, mean, rstd, input, weight, bias, num_groups,
                                          data_format, swish_scale, dgamma_required, dbeta_required, grad_x,
                                          grad_weight, grad_bias));
  runner->Run({grad, mean, rstd, input, weight, bias}, {grad_x, grad_weight, grad_bias});
  return {grad_x, grad_weight, grad_bias};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_group_norm_swish_grad", PYBOOST_CALLER(3, custom::npu_group_norm_swish_grad));
}
}  // namespace custom
