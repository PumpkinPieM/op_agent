#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_grouped_matmul(const ms::Tensor &x, const ms::Tensor &weight, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &scale_opt, const std::optional<ms::Tensor> &offset_opt, const std::optional<ms::Tensor> &antiquant_scale_opt, const std::optional<ms::Tensor> &antiquant_offset_opt, const std::optional<ms::Tensor> &per_token_scale_opt, const std::optional<std::vector<int64_t>> &group_list_opt, const std::optional<ms::Tensor> &activation_input_opt, const std::optional<ms::Tensor> &activation_quant_scale_opt, const std::optional<ms::Tensor> &activation_quant_offset_opt, const std::optional<int64_t> &split_item_opt, const std::optional<int64_t> &group_type_opt, const std::optional<int64_t> &group_list_type_opt, const std::optional<int64_t> &act_type_opt, const std::optional<double> &output_dtype_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto scale_value = scale_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto antiquant_scale_value = antiquant_scale_opt.value_or(ms::Tensor());
  auto antiquant_offset_value = antiquant_offset_opt.value_or(ms::Tensor());
  auto per_token_scale_value = per_token_scale_opt.value_or(ms::Tensor());
  auto group_list = group_list_opt.value_or(std::vector<int64_t>{});
  auto activation_input_value = activation_input_opt.value_or(ms::Tensor());
  auto activation_quant_scale_value = activation_quant_scale_opt.value_or(ms::Tensor());
  auto activation_quant_offset_value = activation_quant_offset_opt.value_or(ms::Tensor());
  auto split_item = split_item_opt.value_or(0);
  auto group_type = group_type_opt.value_or(0);
  auto group_list_type = group_list_type_opt.value_or(0);
  auto act_type = act_type_opt.value_or(0);
  auto output_dtype = output_dtype_opt.value_or(0.0);
  auto base_shape = x.shape();
  auto base_dtype = x.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmul, x, weight, bias_opt, scale_opt, offset_opt, antiquant_scale_opt, antiquant_offset_opt, per_token_scale_opt, group_list, activation_input_opt, activation_quant_scale_opt, activation_quant_offset_opt, split_item, group_type, group_list_type, act_type, output_dtype, out0));
  runner->Run({x, weight, bias_value, scale_value, offset_value, antiquant_scale_value, antiquant_offset_value, per_token_scale_value, activation_input_value, activation_quant_scale_value, activation_quant_offset_value}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul", PYBOOST_CALLER(1, custom::npu_grouped_matmul));
}
}  // namespace custom
