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

std::vector<ms::Tensor> npu_dynamic_dual_level_mx_quant(const ms::Tensor &input,
    const std::optional<ms::Tensor> &smooth_scale_opt = std::nullopt, const std::string &round_mode = "rint") {
  auto smooth_scale = smooth_scale_opt.value_or(ms::Tensor()); auto y_shape = input.shape(); y_shape.back() /= 2;
  auto l0_shape = input.shape(); l0_shape.back() = CeilDiv(l0_shape.back(), 512);
  auto l1_shape = input.shape(); l1_shape.back() = CeilDiv(CeilDiv(l1_shape.back(), 32), 2); l1_shape.push_back(2);
  auto y = ms::Tensor(ms::TypeId::kNumberTypeUInt8, y_shape); auto l0 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, l0_shape); auto l1 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, l1_shape);
  auto round = const_cast<char *>(round_mode.c_str()); constexpr int64_t l0_block = 512; constexpr int64_t l1_block = 32;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DynamicDualLevelMxQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDynamicDualLevelMxQuant, input, smooth_scale_opt, round, l0_block, l1_block, y, l0, l1));
  runner->Run({input, smooth_scale}, {y, l0, l1}); return {y, l0, l1};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_dynamic_dual_level_mx_quant", PYBOOST_CALLER(3, custom::npu_dynamic_dual_level_mx_quant)); }

}  // namespace custom
