#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_group_norm_silu(const ms::Tensor &input,
                                            const std::optional<ms::Tensor> &weight_opt = std::nullopt,
                                            const std::optional<ms::Tensor> &bias_opt = std::nullopt,
                                            int64_t group = 1, double eps = 1e-5) {
  auto weight = weight_opt.value_or(ms::Tensor());
  auto bias = bias_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(input.data_type(), input.shape());
  auto stat_dtype = weight_opt.has_value() ? weight_opt.value().data_type() : input.data_type();
  auto mean = ms::Tensor(stat_dtype, std::vector<int64_t>{input.shape().at(0), group});
  auto rstd = ms::Tensor(stat_dtype, std::vector<int64_t>{input.shape().at(0), group});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupNormSilu");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupNormSilu, input, weight_opt, bias_opt, group, eps, out, mean,
                                          rstd));
  runner->Run({input, weight, bias}, {out, mean, rstd});
  return {out, mean, rstd};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_group_norm_silu", PYBOOST_CALLER(3, custom::npu_group_norm_silu));
}
}  // namespace custom
