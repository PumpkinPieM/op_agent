#include <vector>
#include <optional>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_gating_top_k(const ms::Tensor &x, int64_t k,
    const std::optional<ms::Tensor> &bias_opt = std::nullopt, int64_t k_group = 1, int64_t group_count = 1,
    int64_t group_select_mode = 0, int64_t renorm = 0, int64_t norm_type = 0, bool out_flag = false,
    double routed_scaling_factor = 1.0, double eps = 1e-20) {
  auto bias = bias_opt.value_or(ms::Tensor());
  auto y = ms::Tensor(x.data_type(), std::vector<int64_t>{x.shape()[0], k});
  auto idx = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{x.shape()[0], k});
  auto out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeGatingTopK");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeGatingTopK, x, bias_opt, k, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps, y, idx, out));
  runner->Run({x, bias}, {y, idx, out});
  return {y, idx, out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_gating_top_k", PYBOOST_CALLER(3, custom::npu_moe_gating_top_k));
}
}  // namespace custom
