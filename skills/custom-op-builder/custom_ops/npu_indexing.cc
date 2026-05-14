#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_indexing(const ms::Tensor &self, const std::vector<int64_t> &begin, const std::vector<int64_t> &end, const std::vector<int64_t> &strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask) {
  auto base_shape = self.shape();
  auto base_dtype = self.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("Min");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMin, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out0));
  runner->Run({self}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_indexing", PYBOOST_CALLER(1, custom::npu_indexing));
}
}  // namespace custom
