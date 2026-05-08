#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> HalfDimShape(const ms::Tensor &x, int64_t dim) {
  auto shape = x.shape();
  auto rank = static_cast<int64_t>(shape.size());
  auto axis = dim < 0 ? rank + dim : dim;
  shape.at(static_cast<size_t>(axis)) /= 2;
  return shape;
}
}  // namespace

std::vector<ms::Tensor> npu_geglu(const ms::Tensor &self, int64_t dim = -1, int64_t approximate = 1,
                                  bool activate_left = false) {
  auto result = ms::Tensor(self.data_type(), HalfDimShape(self, dim));
  auto result_gelu = ms::Tensor(self.data_type(), HalfDimShape(self, dim));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GeGluV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGeGluV3, self, dim, approximate, activate_left, result, result_gelu));
  runner->Run({self}, {result, result_gelu});
  return {result, result_gelu};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_geglu", PYBOOST_CALLER(2, custom::npu_geglu));
}
}  // namespace custom
