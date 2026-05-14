#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_geglu_grad(const ms::Tensor &grad_output, const ms::Tensor &self, const ms::Tensor &gelu,
                          int64_t dim = -1, int64_t approximate = 1, bool activate_left = false) {
  auto grad_input = ms::Tensor(self.data_type(), self.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GeGluV3Backward");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGeGluV3Backward, grad_output, self, gelu, dim, approximate,
                                          activate_left, grad_input));
  runner->Run({grad_output, self, gelu}, {grad_input});
  return grad_input;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_geglu_grad", PYBOOST_CALLER(1, custom::npu_geglu_grad));
}
}  // namespace custom
