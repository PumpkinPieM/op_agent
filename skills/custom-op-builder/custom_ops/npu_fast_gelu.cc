#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_fast_gelu(const ms::Tensor &self) {
  auto out = ms::Tensor(self.data_type(), self.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FastGelu");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFastGelu, self, out));
  runner->Run({self}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fast_gelu", PYBOOST_CALLER(1, custom::npu_fast_gelu));
}
}  // namespace custom
