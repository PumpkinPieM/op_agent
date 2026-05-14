#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> SwigluOutShape(const ms::Tensor &x, int64_t dim) {
  auto shape = x.shape();
  auto rank = static_cast<int64_t>(shape.size());
  auto axis = dim < 0 ? rank + dim : dim;
  shape.at(static_cast<size_t>(axis)) /= 2;
  return shape;
}
}  // namespace

ms::Tensor npu_clipped_swiglu(const ms::Tensor &x, const std::optional<ms::Tensor> &group_index_opt = std::nullopt,
                              int64_t dim = -1, double alpha = 1.702, double limit = 7.0, double bias = 1.0,
                              bool interleaved = true) {
  auto group_index = group_index_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(x.data_type(), SwigluOutShape(x, dim));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ClippedSwiglu");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnClippedSwiglu, x, group_index_opt, dim, alpha, limit, bias,
                                          interleaved, out));
  runner->Run({x, group_index}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_clipped_swiglu", PYBOOST_CALLER(1, custom::npu_clipped_swiglu));
}
}  // namespace custom
