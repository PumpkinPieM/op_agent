#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_hans_decode(const ms::Tensor &mantissa, const ms::Tensor &fixed, const ms::Tensor &var, const ms::Tensor &pdf, bool reshuff = false) {
  auto out = ms::Tensor(mantissa.data_type(), mantissa.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("HansDecode");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnHansDecode, mantissa, fixed, var, pdf, reshuff, out));
  runner->Run({mantissa, fixed, var, pdf}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_hans_decode", PYBOOST_CALLER(1, custom::npu_hans_decode));
}
}  // namespace custom
