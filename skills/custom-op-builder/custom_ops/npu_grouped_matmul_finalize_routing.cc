#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_grouped_matmul_finalize_routing(const ms::Tensor &x, const ms::Tensor &w, const ms::Tensor &group_list, const std::optional<ms::Tensor> &scale_opt, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &offset_opt, const std::optional<ms::Tensor> &pertoken_scale_opt, const std::optional<ms::Tensor> &shared_input_opt, const std::optional<ms::Tensor> &logit_opt, const std::optional<ms::Tensor> &row_index_opt, const std::optional<int64_t> &dtype_opt, const std::optional<double> &shared_input_weight_opt, const std::optional<int64_t> &shared_input_offset_opt, const std::optional<int64_t> &output_bs_opt, const std::optional<int64_t> &group_list_type_opt, const std::optional<std::vector<int64_t>> &tuning_config_opt, const std::optional<int64_t> &x_dtype_opt, const std::optional<int64_t> &w_dtype_opt, const std::optional<int64_t> &scale_dtype_opt, const std::optional<int64_t> &pertoken_scale_dtype_opt) {
  auto scale_value = scale_opt.value_or(ms::Tensor());
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto pertoken_scale_value = pertoken_scale_opt.value_or(ms::Tensor());
  auto shared_input_value = shared_input_opt.value_or(ms::Tensor());
  auto logit_value = logit_opt.value_or(ms::Tensor());
  auto row_index_value = row_index_opt.value_or(ms::Tensor());
  auto dtype = dtype_opt.value_or(0);
  auto shared_input_weight = static_cast<float>(shared_input_weight_opt.value_or(1.0));
  auto shared_input_offset = shared_input_offset_opt.value_or(0);
  auto output_bs = output_bs_opt.value_or(0);
  auto group_list_type = group_list_type_opt.value_or(1);
  (void)tuning_config_opt;
  (void)x_dtype_opt;
  (void)w_dtype_opt;
  (void)scale_dtype_opt;
  (void)pertoken_scale_dtype_opt;
  auto out_shape = x.shape();
  out_shape[0] = output_bs == 0 ? out_shape[0] : output_bs;
  out_shape[1] = w.shape().back();
  auto out0 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, out_shape);
  auto antiquant_scale = ms::Tensor();
  auto antiquant_offset = ms::Tensor();
  bool transpose_x = false;
  bool transpose_w = false;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulFinalizeRoutingV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulFinalizeRoutingV2, x, w, scale_opt, bias_opt, offset_opt,
                                          antiquant_scale, antiquant_offset, pertoken_scale_opt, group_list,
                                          shared_input_opt, logit_opt, row_index_opt, dtype, shared_input_weight,
                                          shared_input_offset, transpose_x, transpose_w, group_list_type, out0));
  runner->Run({x, w, group_list, scale_value, bias_value, offset_value, pertoken_scale_value, shared_input_value, logit_value, row_index_value}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_finalize_routing", PYBOOST_CALLER(1, custom::npu_grouped_matmul_finalize_routing));
}
}  // namespace custom
