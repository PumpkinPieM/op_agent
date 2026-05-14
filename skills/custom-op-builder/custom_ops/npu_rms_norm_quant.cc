#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_rms_norm_quant(const ms::Tensor &x, const ms::Tensor &gamma, const ms::Tensor &beta,
                              const ms::Tensor &scale, const ms::Tensor &offset, double epsilon = 1e-6,
                              const std::optional<int64_t> &dst_dtype_opt = std::nullopt) {
  (void)dst_dtype_opt;
  auto y = ms::Tensor(ms::TypeId::kNumberTypeInt8, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("RmsNormQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRmsNormQuant, x, gamma, beta, scale, offset, epsilon, y));
  runner->Run({x, gamma, beta, scale, offset}, {y});
  return y;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_rms_norm_quant", PYBOOST_CALLER(1, custom::npu_rms_norm_quant));
}
}  // namespace custom
