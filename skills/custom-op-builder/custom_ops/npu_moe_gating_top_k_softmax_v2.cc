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

std::vector<ms::Tensor> npu_moe_gating_top_k_softmax_v2(const ms::Tensor &x, int64_t k = 1,
    const std::optional<ms::Tensor> &finished_opt = std::nullopt, const std::optional<int64_t> &renorm_opt = std::nullopt,
    const std::optional<bool> &output_softmax_opt = std::nullopt) {
  auto finished = finished_opt.value_or(ms::Tensor()); auto renorm = renorm_opt.value_or(0); auto softmax_flag = output_softmax_opt.value_or(false) && renorm == 0;
  std::vector<int64_t> top_shape{x.shape()[0], k}; if (x.shape().size()==3) top_shape = {x.shape()[0], x.shape()[1], k};
  auto y = ms::Tensor(x.data_type(), top_shape); auto idx = ms::Tensor(ms::TypeId::kNumberTypeInt32, top_shape);
  auto softmax = ms::Tensor(ms::TypeId::kNumberTypeFloat32, softmax_flag ? x.shape() : std::vector<int64_t>{0});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeGatingTopKSoftmaxV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeGatingTopKSoftmaxV2, x, finished_opt, k, renorm, softmax_flag, y, idx, softmax));
  runner->Run({x, finished}, {y, idx, softmax}); return {y, idx, softmax};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_moe_gating_top_k_softmax_v2", PYBOOST_CALLER(3, custom::npu_moe_gating_top_k_softmax_v2)); }

}  // namespace custom
