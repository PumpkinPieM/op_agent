#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_scatter_nd_update_(const ms::Tensor &self, const ms::Tensor &indices, const ms::Tensor &updates) {
  auto out = self;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ScatterNdUpdate");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScatterNdUpdate, out, indices, updates));
  runner->Run({out, indices, updates}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_scatter_nd_update_", PYBOOST_CALLER(1, custom::npu_scatter_nd_update_));
}
}  // namespace custom
