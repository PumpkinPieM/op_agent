#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_grouped_matmul_swiglu_quant_v2(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &weight_scale, const ms::Tensor &x_scale, const ms::Tensor &group_list, const std::optional<ms::Tensor> &smooth_scale_opt, const std::optional<ms::Tensor> &weight_assist_matrix_opt, const std::optional<ms::Tensor> &bias_opt, const std::optional<int64_t> &dequant_mode_opt, const std::optional<int64_t> &dequant_dtype_opt, const std::optional<int64_t> &quant_mode_opt, const std::optional<int64_t> &quant_dtype_opt, const std::optional<int64_t> &group_list_type_opt, const std::optional<std::vector<int64_t>> &tuning_config_opt, const std::optional<int64_t> &x_dtype_opt, const std::optional<int64_t> &weight_dtype_opt, const std::optional<int64_t> &weight_scale_dtype_opt, const std::optional<int64_t> &x_scale_dtype_opt) {
  auto smooth_scale_value = smooth_scale_opt.value_or(ms::Tensor());
  auto weight_assist_matrix_value = weight_assist_matrix_opt.value_or(ms::Tensor());
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto dequant_mode = dequant_mode_opt.value_or(0);
  auto dequant_dtype = dequant_dtype_opt.value_or(0);
  auto quant_mode = quant_mode_opt.value_or(0);
  auto quant_dtype = quant_dtype_opt.value_or(1);
  auto group_list_type = group_list_type_opt.value_or(0);
  auto tuning_config = tuning_config_opt.value_or(std::vector<int64_t>{});
  auto x_dtype = x_dtype_opt.value_or(0);
  auto weight_dtype = weight_dtype_opt.value_or(0);
  auto weight_scale_dtype = weight_scale_dtype_opt.value_or(0);
  auto x_scale_dtype = x_scale_dtype_opt.value_or(0);
  auto base_shape = x.shape();
  auto base_dtype = x.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulSwigluQuantV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulSwigluQuantV2, x, weight, weight_scale, x_scale, group_list, smooth_scale_opt, weight_assist_matrix_opt, bias_opt, dequant_mode, dequant_dtype, quant_mode, quant_dtype, group_list_type, tuning_config, x_dtype, weight_dtype, weight_scale_dtype, x_scale_dtype, out0, out1));
  runner->Run({x, weight, weight_scale, x_scale, group_list, smooth_scale_value, weight_assist_matrix_value, bias_value}, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_swiglu_quant_v2", PYBOOST_CALLER(2, custom::npu_grouped_matmul_swiglu_quant_v2));
}
}  // namespace custom
