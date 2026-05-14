#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_scaled_masked_softmax(const ms::Tensor &x, const ms::Tensor &mask, double scale = 1.0,
                                     bool fixed_triu_mask = false) {
  auto out = ms::Tensor(x.data_type(), x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ScaledMaskedSoftmax");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScaledMaskedSoftmax, x, mask, scale, fixed_triu_mask, out));
  runner->Run({x, mask}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_scaled_masked_softmax", PYBOOST_CALLER(1, custom::npu_scaled_masked_softmax));
}
}  // namespace custom
