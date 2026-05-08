#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_hans_encode(const ms::Tensor &x, const ms::Tensor &pdf, bool reshuff = false) {
  auto mantissa = ms::Tensor(ms::TypeId::kNumberTypeInt32, x.shape());
  auto fixed = ms::Tensor(ms::TypeId::kNumberTypeInt32, x.shape());
  auto var = ms::Tensor(ms::TypeId::kNumberTypeInt32, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("HansEncode");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnHansEncode, x, pdf, reshuff, mantissa, fixed, var));
  runner->Run({x, pdf}, {mantissa, fixed, var});
  return {mantissa, fixed, var};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_hans_encode", PYBOOST_CALLER(3, custom::npu_hans_encode));
}
}  // namespace custom
