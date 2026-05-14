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

ms::Tensor npu_fused_matmul(const ms::Tensor &x1, const ms::Tensor &x2,
    const std::optional<ms::Tensor> &bias_opt = std::nullopt, const std::optional<ms::Tensor> &x3_opt = std::nullopt,
    const std::string &fused_op_type = "") {
  auto bias = bias_opt.value_or(ms::Tensor()); auto x3 = x3_opt.value_or(ms::Tensor()); auto s = x1.shape(); s.back() = x2.shape().back();
  auto out = ms::Tensor(fused_op_type == "16cast32" ? ms::TypeId::kNumberTypeFloat32 : x1.data_type(), s); auto fused = const_cast<char *>(fused_op_type.c_str()); constexpr int8_t cube_math_type = 0;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedMatmul, x1, x2, bias, x3, fused, cube_math_type, out));
  runner->Run({x1, x2, bias, x3}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_fused_matmul", PYBOOST_CALLER(1, custom::npu_fused_matmul)); }

}  // namespace custom
