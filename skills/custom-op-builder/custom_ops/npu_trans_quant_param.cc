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

ms::Tensor npu_trans_quant_param(const ms::Tensor &scale, const std::optional<ms::Tensor> &offset_opt = std::nullopt,
                                    const std::optional<int64_t> &round_mode_opt = std::nullopt) {
  auto offset = offset_opt.value_or(ms::Tensor()); auto round_mode = round_mode_opt.value_or(0); auto out_shape = scale.shape(); if (offset_opt.has_value() && scale.shape().size()==1 && offset.shape()[0] > out_shape[0]) out_shape = offset.shape();
  auto out = ms::Tensor(ms::TypeId::kNumberTypeInt64, out_shape); auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("TransQuantParamV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnTransQuantParamV3, scale, offset, round_mode, out));
  runner->Run({scale, offset}, {out}); return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_trans_quant_param", PYBOOST_CALLER(1, custom::npu_trans_quant_param)); }

}  // namespace custom
