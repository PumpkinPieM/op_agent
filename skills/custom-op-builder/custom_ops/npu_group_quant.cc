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

ms::Tensor npu_group_quant(const ms::Tensor &x, const ms::Tensor &scale, const ms::Tensor &group_index,
                           const std::optional<ms::Tensor> &offset_opt = std::nullopt,
                           const std::optional<int64_t> &dst_dtype_opt = std::nullopt) {
  auto offset = offset_opt.value_or(ms::Tensor()); auto dst_type = dst_dtype_opt.value_or(2);
  auto out = ms::Tensor(ms::TypeId::kNumberTypeInt8, x.shape()); auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupQuant, x, scale, group_index, offset_opt, dst_type, out));
  runner->Run({x, scale, group_index, offset}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_group_quant", PYBOOST_CALLER(1, custom::npu_group_quant)); }

}  // namespace custom
