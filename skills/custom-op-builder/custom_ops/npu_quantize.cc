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

ms::Tensor npu_quantize(const ms::Tensor &self, const ms::Tensor &scales,
    const std::optional<ms::Tensor> &zero_points_opt, int64_t dtype, int64_t axis = 1, bool div_mode = true) {
  auto zero_points = zero_points_opt.value_or(ms::Tensor()); auto out = ms::Tensor(ms::TypeId::kNumberTypeInt8, self.shape());
  auto round = const_cast<char *>("round"); constexpr bool sqrt_mode = false;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AscendQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAscendQuant, self, scales, zero_points_opt, sqrt_mode, round, dtype, out));
  runner->Run({self, scales, zero_points}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_quantize", PYBOOST_CALLER(1, custom::npu_quantize)); }

}  // namespace custom
