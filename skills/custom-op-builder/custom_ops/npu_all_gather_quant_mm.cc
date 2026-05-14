#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_all_gather_quant_mm(const ms::Tensor &self, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt, const std::optional<ms::Tensor> &quant_scale_opt, int64_t block_size, int64_t gather_index, bool gather_output, int64_t comm_turn, const std::optional<std::vector<int64_t>> &group_sizes_opt, bool amax_output, const std::optional<int64_t> &y_dtype_opt, const std::optional<int64_t> &x1_dtype_opt, const std::optional<int64_t> &x2_dtype_opt, const std::optional<int64_t> &x1_scale_dtype_opt, const std::optional<int64_t> &x2_scale_dtype_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto x1_scale_value = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale_value = x2_scale_opt.value_or(ms::Tensor());
  auto quant_scale_value = quant_scale_opt.value_or(ms::Tensor());
  auto group_sizes = group_sizes_opt.value_or(std::vector<int64_t>{});
  auto y_dtype = y_dtype_opt.value_or(0);
  auto x1_dtype = x1_dtype_opt.value_or(0);
  auto x2_dtype = x2_dtype_opt.value_or(0);
  auto x1_scale_dtype = x1_scale_dtype_opt.value_or(0);
  auto x2_scale_dtype = x2_scale_dtype_opt.value_or(0);
  auto base_shape = self.shape();
  auto base_dtype = self.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AllGatherMatmulV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAllGatherMatmulV2, self, x2, hcom, world_size, bias_opt, x1_scale_opt, x2_scale_opt, quant_scale_opt, block_size, gather_index, gather_output, comm_turn, group_sizes, amax_output, y_dtype, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype, out0, out1, out2));
  runner->Run({self, x2, bias_value, x1_scale_value, x2_scale_value, quant_scale_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_gather_quant_mm", PYBOOST_CALLER(3, custom::npu_all_gather_quant_mm));
}
}  // namespace custom
