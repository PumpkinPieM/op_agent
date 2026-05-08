#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_add_quant_matmul(const ms::Tensor &self, const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &x2_scale, const ms::Tensor &group_list, const std::optional<ms::Tensor> &x1_scale_opt = std::nullopt, const std::optional<int64_t> &group_list_type_opt = std::nullopt, const std::optional<std::vector<int64_t>> &group_sizes_opt = std::nullopt, const std::optional<int64_t> &x1_dtype_opt = std::nullopt, const std::optional<int64_t> &x2_dtype_opt = std::nullopt, const std::optional<int64_t> &x1_scale_dtype_opt = std::nullopt, const std::optional<int64_t> &x2_scale_dtype_opt = std::nullopt) {
  auto x1_scale = x1_scale_opt.value_or(ms::Tensor());
  auto group_list_type = group_list_type_opt.value_or(0);
  auto group_sizes = std::make_pair(group_sizes_opt.value_or(std::vector<int64_t>{}), true);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantBatchMatmulInplaceAdd");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantBatchMatmulInplaceAdd, x1, x2, x1_scale_opt, x2_scale, group_list, group_list_type, group_sizes, self));
  runner->Run({self, x1, x2, x2_scale, group_list, x1_scale}, {self});
  return {self};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_add_quant_matmul", PYBOOST_CALLER(1, custom::npu_add_quant_matmul));
}
}  // namespace custom
