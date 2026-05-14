#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_dual_level_quant_matmul(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &x1_level0_scale, const ms::Tensor &x2_level0_scale, const ms::Tensor &x1_level1_scale, const ms::Tensor &x2_level1_scale, const std::optional<ms::Tensor> &bias_opt, int64_t output_dtype) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto base_shape = x1.shape();
  auto base_dtype = x1.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DualLevelQuantMatmulWeightNz");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDualLevelQuantMatmulWeightNz, x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias_opt, output_dtype, out0));
  runner->Run({x1, x2, x1_level0_scale, x2_level0_scale, x1_level1_scale, x2_level1_scale, bias_value}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_dual_level_quant_matmul", PYBOOST_CALLER(1, custom::npu_dual_level_quant_matmul));
}
}  // namespace custom
