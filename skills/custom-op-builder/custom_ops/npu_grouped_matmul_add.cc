#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_grouped_matmul_add(const ms::Tensor &self, const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list, bool transpose_x = true, bool transpose_weight = false, int64_t group_type = 2, const std::optional<int64_t> &group_list_type_opt = std::nullopt) {
  auto group_list_type = group_list_type_opt.value_or(0);
  auto out = ms::Tensor(self.data_type(), self.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulAddV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulAddV2, x, weight, group_list, out, transpose_x,
                                          transpose_weight, group_type, group_list_type));
  runner->Run({self, x, weight, group_list}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_add", PYBOOST_CALLER(1, custom::npu_grouped_matmul_add));
}
}  // namespace custom
