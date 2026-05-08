#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_group_norm_swish(const ms::Tensor &input, int64_t num_groups, const ms::Tensor &weight,
                                             const ms::Tensor &bias,
                                             const std::optional<double> &eps_opt = std::nullopt,
                                             const std::optional<double> &swish_scale_opt = std::nullopt) {
  auto out = ms::Tensor(input.data_type(), input.shape());
  auto stat_shape = std::vector<int64_t>{input.shape().at(0), num_groups};
  auto mean = ms::Tensor(weight.data_type(), stat_shape);
  auto rstd = ms::Tensor(weight.data_type(), stat_shape);
  auto data_format = const_cast<char *>("NCHW");
  auto eps = eps_opt.value_or(1e-5);
  auto swish_scale = swish_scale_opt.value_or(1.0);
  constexpr bool activate_swish = true;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupNormSwish");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupNormSwish, input, weight, bias, num_groups, data_format, eps,
                                          activate_swish, swish_scale, out, mean, rstd));
  runner->Run({input, weight, bias}, {out, mean, rstd});
  return {out, mean, rstd};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_group_norm_swish", PYBOOST_CALLER(3, custom::npu_group_norm_swish));
}
}  // namespace custom
