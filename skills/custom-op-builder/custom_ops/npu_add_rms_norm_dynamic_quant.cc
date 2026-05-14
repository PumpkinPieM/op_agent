#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> ReduceLastDim(const ms::Tensor &x) {
  auto s = x.shape();
  if (!s.empty()) {
    s.pop_back();
  }
  return s;
}

bool OutputMaskEnabled(const std::vector<uint8_t> &output_mask, size_t index) {
  return output_mask.size() > index && output_mask[index] != 0;
}
}  // namespace

std::vector<ms::Tensor> npu_add_rms_norm_dynamic_quant(
    const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &gamma,
    const std::optional<ms::Tensor> &smooth_scale1_opt = std::nullopt,
    const std::optional<ms::Tensor> &smooth_scale2_opt = std::nullopt,
    const std::optional<ms::Tensor> &beta_opt = std::nullopt, double epsilon = 1e-6,
    const std::vector<uint8_t> &output_mask = std::vector<uint8_t>{1, 1},
    const std::optional<int64_t> &y_dtype_opt = std::nullopt) {
  (void)y_dtype_opt;
  auto smooth_scale1 = smooth_scale1_opt.value_or(ms::Tensor());
  auto smooth_scale2 = smooth_scale2_opt.value_or(ms::Tensor());
  auto beta = beta_opt.value_or(ms::Tensor());
  auto y1 = ms::Tensor(ms::TypeId::kNumberTypeInt8, x1.shape());
  auto y2_shape = OutputMaskEnabled(output_mask, 1) ? x1.shape() : std::vector<int64_t>{};
  auto y2 = ms::Tensor(ms::TypeId::kNumberTypeInt8, y2_shape);
  auto x_out = ms::Tensor(x1.data_type(), x1.shape());
  auto scale1 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ReduceLastDim(x1));
  auto scale2_shape = OutputMaskEnabled(output_mask, 1) ? ReduceLastDim(x1) : std::vector<int64_t>{};
  auto scale2 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, scale2_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AddRmsNormDynamicQuantV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAddRmsNormDynamicQuantV2, x1, x2, gamma, smooth_scale1_opt,
                                          smooth_scale2_opt, beta_opt, epsilon, output_mask, y1, y2, x_out, scale1,
                                          scale2));
  runner->Run({x1, x2, gamma, smooth_scale1, smooth_scale2, beta}, {y1, y2, x_out, scale1, scale2});
  return {y1, y2, x_out, scale1, scale2};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_add_rms_norm_dynamic_quant", PYBOOST_CALLER(5, custom::npu_add_rms_norm_dynamic_quant));
}

}  // namespace custom
