#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_batch_gather_matmul(const ms::Tensor &self, const ms::Tensor &x, const ms::Tensor &weight_b, const ms::Tensor &indices, const std::optional<ms::Tensor> &weight_a_opt, int64_t layer_idx, double scale, int64_t y_offset, int64_t y_slice_size) {
  auto weight_a_value = weight_a_opt.value_or(ms::Tensor());
  auto base_shape = self.shape();
  auto y_slice_size_value = y_slice_size == -1 ? base_shape[1] : y_slice_size;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AddLora");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAddLora, self, x, weight_b, indices, weight_a_opt, layer_idx, scale,
                                          y_offset, y_slice_size_value, self));
  runner->Run({self, x, weight_b, indices, weight_a_value}, {self});
  return {self};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_batch_gather_matmul", PYBOOST_CALLER(1, custom::npu_batch_gather_matmul));
}
}  // namespace custom
