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

std::vector<ms::Tensor> npu_add_rms_norm_quant(
    const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &gamma, const ms::Tensor &scales1,
    const std::optional<ms::Tensor> &zero_points1_opt = std::nullopt,
    const std::optional<ms::Tensor> &beta_opt = std::nullopt,
    const std::optional<ms::Tensor> &scales2_opt = std::nullopt,
    const std::optional<ms::Tensor> &zero_points2_opt = std::nullopt, int64_t axis = -1, double epsilon = 1e-6,
    bool div_mode = true, const std::optional<int64_t> &dst_type_opt = std::nullopt) {
  (void)dst_type_opt;
  auto zp1 = zero_points1_opt.value_or(ms::Tensor()); auto beta = beta_opt.value_or(ms::Tensor());
  auto scales2 = scales2_opt.value_or(ms::Tensor()); auto zp2 = zero_points2_opt.value_or(ms::Tensor());
  auto y1 = ms::Tensor(ms::TypeId::kNumberTypeInt8, x1.shape());
  auto y2 = ms::Tensor(ms::TypeId::kNumberTypeInt8, x1.shape());
  auto x_out = ms::Tensor(x1.data_type(), x1.shape());
  auto rms_norm = ms::Tensor(x1.data_type(), x1.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AddRmsNormQuantV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAddRmsNormQuantV2, x1, x2, gamma, scales1, scales2_opt,
                                          zero_points1_opt, zero_points2_opt, beta_opt, axis, epsilon, div_mode, y1, y2,
                                          x_out, rms_norm));
  runner->Run({x1, x2, gamma, scales1, zp1, beta, scales2, zp2}, {y1, y2, x_out, rms_norm});
  return {y1, y2, x_out};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_add_rms_norm_quant", PYBOOST_CALLER(3, custom::npu_add_rms_norm_quant)); }

}  // namespace custom
