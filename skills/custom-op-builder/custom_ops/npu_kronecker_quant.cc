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

std::vector<ms::Tensor> npu_kronecker_quant(const ms::Tensor &x, const ms::Tensor &kronecker_p1, const ms::Tensor &kronecker_p2,
    const std::optional<double> &clip_ratio_opt = std::nullopt, const std::optional<int64_t> &dst_dtype_opt = std::nullopt) {
  (void)dst_dtype_opt; auto clip_ratio = clip_ratio_opt.value_or(1.0); auto os = x.shape(); os.back() /= 8;
  auto out = ms::Tensor(ms::TypeId::kNumberTypeInt32, os); auto scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{x.shape()[0]});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlatQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlatQuant, x, kronecker_p1, kronecker_p2, clip_ratio, out, scale));
  runner->Run({x, kronecker_p1, kronecker_p2}, {out, scale}); return {out, scale};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_kronecker_quant", PYBOOST_CALLER(2, custom::npu_kronecker_quant)); }

}  // namespace custom
