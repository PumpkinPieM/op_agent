#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_all_to_all_quant_matmul(const ms::Tensor &x1, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, bool all2all_out_flag, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt, const std::optional<ms::Tensor> &common_scale_opt, const std::optional<ms::Tensor> &x1_offset_opt, const std::optional<ms::Tensor> &x2_offset_opt, const std::optional<int64_t> &x1_quant_mode_opt, const std::optional<int64_t> &x2_quant_mode_opt, const std::optional<int64_t> &common_quant_mode_opt, const std::optional<std::vector<int64_t>> &group_sizes_opt, const std::optional<std::vector<int64_t>> &all2all_axes_opt, const std::optional<int64_t> &comm_quant_dtype_opt, const std::optional<int64_t> &x1_quant_dtype_opt, const std::optional<int64_t> &x1_dtype_opt, const std::optional<int64_t> &x2_dtype_opt, const std::optional<int64_t> &x1_scale_dtype_opt, const std::optional<int64_t> &x2_scale_dtype_opt, const std::optional<int64_t> &output_scale_dtype_opt, const std::optional<int64_t> &comm_scale_dtype_opt, const std::optional<int64_t> &y_dtype_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto x1_scale_value = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale_value = x2_scale_opt.value_or(ms::Tensor());
  auto common_scale_value = common_scale_opt.value_or(ms::Tensor());
  auto x1_offset_value = x1_offset_opt.value_or(ms::Tensor());
  auto x2_offset_value = x2_offset_opt.value_or(ms::Tensor());
  auto x1_quant_mode = x1_quant_mode_opt.value_or(0);
  auto x2_quant_mode = x2_quant_mode_opt.value_or(0);
  auto common_quant_mode = common_quant_mode_opt.value_or(0);
  auto group_sizes = group_sizes_opt.value_or(std::vector<int64_t>{});
  auto all2all_axes = all2all_axes_opt.value_or(std::vector<int64_t>{});
  auto comm_quant_dtype = comm_quant_dtype_opt.value_or(0);
  auto x1_quant_dtype = x1_quant_dtype_opt.value_or(0);
  auto x1_dtype = x1_dtype_opt.value_or(0);
  auto x2_dtype = x2_dtype_opt.value_or(0);
  auto x1_scale_dtype = x1_scale_dtype_opt.value_or(0);
  auto x2_scale_dtype = x2_scale_dtype_opt.value_or(0);
  auto output_scale_dtype = output_scale_dtype_opt.value_or(0);
  auto comm_scale_dtype = comm_scale_dtype_opt.value_or(0);
  auto y_dtype = y_dtype_opt.value_or(0);
  auto base_shape = x1.shape();
  auto base_dtype = x1.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AlltoAllQuantMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAlltoAllQuantMatmul, x1, x2, hcom, world_size, all2all_out_flag, bias_opt, x1_scale_opt, x2_scale_opt, common_scale_opt, x1_offset_opt, x2_offset_opt, x1_quant_mode, x2_quant_mode, common_quant_mode, group_sizes, all2all_axes, comm_quant_dtype, x1_quant_dtype, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype, output_scale_dtype, comm_scale_dtype, y_dtype, out0, out1));
  runner->Run({x1, x2, bias_value, x1_scale_value, x2_scale_value, common_scale_value, x1_offset_value, x2_offset_value}, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_to_all_quant_matmul", PYBOOST_CALLER(2, custom::npu_all_to_all_quant_matmul));
}
}  // namespace custom
