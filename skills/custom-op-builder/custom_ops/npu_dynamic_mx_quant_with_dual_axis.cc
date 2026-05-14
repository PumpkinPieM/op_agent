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

std::vector<ms::Tensor> npu_dynamic_mx_quant_with_dual_axis(const ms::Tensor &input,
    const std::string &round_mode = "rint", int64_t dst_type = 296, int64_t scale_alg = 0) {
  auto y1 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, input.shape()); auto y2 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, input.shape());
  auto s1 = input.shape(); s1.push_back(2); s1[s1.size()-2] = CeilDiv(CeilDiv(s1[s1.size()-2], 32), 2);
  auto s2 = input.shape(); s2.push_back(2); s2[s2.size()-3] = CeilDiv(CeilDiv(s2[s2.size()-3], 32), 2);
  auto m1 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, s1); auto m2 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, s2); auto round = const_cast<char *>(round_mode.c_str());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DynamicMxQuantWithDualAxis");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDynamicMxQuantWithDualAxis, input, round, dst_type, scale_alg, y1, m1, y2, m2));
  runner->Run({input}, {y1, m1, y2, m2}); return {y1, m1, y2, m2};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_dynamic_mx_quant_with_dual_axis", PYBOOST_CALLER(4, custom::npu_dynamic_mx_quant_with_dual_axis)); }

}  // namespace custom
