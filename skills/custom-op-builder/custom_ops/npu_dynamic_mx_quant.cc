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

std::vector<ms::Tensor> npu_dynamic_mx_quant(const ms::Tensor &input, int64_t axis = -1,
    const std::string &round_mode = "rint", int64_t dst_type = 296, int64_t block_size = 32,
    const std::optional<int64_t> &scale_alg_opt = std::nullopt, double dst_type_max = 0.0) {
  auto y = ms::Tensor(ms::TypeId::kNumberTypeUInt8, input.shape()); auto scale_shape = input.shape(); scale_shape.push_back(2);
  int64_t ax = axis < 0 ? axis + static_cast<int64_t>(input.shape().size()) : axis; scale_shape[ax] = CeilDiv(CeilDiv(scale_shape[ax], block_size), 2);
  auto mxscale = ms::Tensor(ms::TypeId::kNumberTypeUInt8, scale_shape); auto round = const_cast<char *>(round_mode.c_str()); auto scale_alg = scale_alg_opt.value_or(0);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DynamicMxQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDynamicMxQuant, input, axis, round, dst_type, block_size, scale_alg, y, mxscale));
  runner->Run({input}, {y, mxscale}); return {y, mxscale};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_dynamic_mx_quant", PYBOOST_CALLER(2, custom::npu_dynamic_mx_quant)); }

}  // namespace custom
