#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> ReduceLastDim(const ms::Tensor &x) { auto s = x.shape(); if (!s.empty()) s.pop_back(); return s; }
std::vector<int64_t> LastHalfShape(const ms::Tensor &x) { auto s = x.shape(); if (!s.empty()) s.back() /= 2; return s; }
int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

std::vector<ms::Tensor> npu_dequant_swiglu_quant(
    const ms::Tensor &x, const std::optional<ms::Tensor> &weight_scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &activation_scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &bias_opt = std::nullopt,
    const std::optional<ms::Tensor> &quant_scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &quant_offset_opt = std::nullopt,
    const std::optional<ms::Tensor> &group_index_opt = std::nullopt, bool activate_left = false,
    int64_t quant_mode = 0, const std::optional<int64_t> &dst_type_opt = std::nullopt,
    const std::optional<int64_t> &round_mode_opt = std::nullopt, const std::optional<int64_t> &activate_dim_opt = std::nullopt,
    int64_t swiglu_mode = 0, double clamp_limit = 7.0, double glu_alpha = 1.0, double glu_bias = 0.0) {
  auto weight_scale = weight_scale_opt.value_or(ms::Tensor()); auto activation_scale = activation_scale_opt.value_or(ms::Tensor());
  auto bias = bias_opt.value_or(ms::Tensor()); auto quant_scale = quant_scale_opt.value_or(ms::Tensor());
  auto quant_offset = quant_offset_opt.value_or(ms::Tensor()); auto group_index = group_index_opt.value_or(ms::Tensor());
  auto quant_mode_value = quant_mode == 0 ? std::string("static") : std::string("dynamic"); auto quant_mode_ptr = const_cast<char *>(quant_mode_value.c_str());
  auto round_mode_value = round_mode_opt.value_or(0) == 4 ? std::string("trunc") : std::string("rint"); auto round_mode_ptr = const_cast<char *>(round_mode_value.c_str());
  auto dst_type = dst_type_opt.value_or(2); auto activate_dim = activate_dim_opt.value_or(-1);
  auto y = ms::Tensor(ms::TypeId::kNumberTypeInt8, LastHalfShape(x)); auto scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ReduceLastDim(x));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DequantSwigluQuantV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDequantSwigluQuantV2, x, weight_scale_opt, activation_scale_opt, bias_opt,
                                          quant_scale_opt, quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr,
                                          dst_type, round_mode_ptr, activate_dim, swiglu_mode, clamp_limit, glu_alpha, glu_bias, y, scale));
  runner->Run({x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index}, {y, scale});
  return {y, scale};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_dequant_swiglu_quant", PYBOOST_CALLER(2, custom::npu_dequant_swiglu_quant)); }

}  // namespace custom
