#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_gmm_alltoallv(const ms::Tensor &x1, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, const std::optional<ms::Tensor> &bias_opt = std::nullopt, const std::optional<std::vector<int64_t>> &axes_opt = std::nullopt, bool out_flag = true) {
  auto bias = bias_opt.value_or(ms::Tensor());
  auto axes = std::make_pair(axes_opt.value_or(std::vector<int64_t>{}), true);
  auto y_shape = x1.shape();
  if (!y_shape.empty() && !x2.shape().empty()) y_shape.back() = x2.shape().back();
  auto y = ms::Tensor(x1.data_type(), y_shape);
  auto gather_out = ms::Tensor(x1.data_type(), x1.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatMulAlltoAllv");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatMulAlltoAllv, x1, x2, bias_opt, hcom, world_size, axes, out_flag, y, gather_out));
  runner->Run({x1, x2, bias}, {y, gather_out});
  return {y, gather_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_gmm_alltoallv", PYBOOST_CALLER(2, custom::npu_gmm_alltoallv));
}
}  // namespace custom
