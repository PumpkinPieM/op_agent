#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_grouped_matmul_swiglu_quant(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list, const ms::Tensor &weight_scale, const ms::Tensor &x_scale, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &offset_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto base_shape = x.shape();
  auto base_dtype = x.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulSwigluQuantWeightNZ");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulSwigluQuantWeightNZ, x, weight, group_list, weight_scale, x_scale, bias_opt, offset_opt, out0, out1, out2));
  runner->Run({x, weight, group_list, weight_scale, x_scale, bias_value, offset_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_swiglu_quant", PYBOOST_CALLER(3, custom::npu_grouped_matmul_swiglu_quant));
}
}  // namespace custom
