#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> ParamReducedShape(const ms::Tensor &x, const ms::Tensor &gamma) {
  auto out = x.shape();
  auto param_dim = static_cast<int64_t>(out.size()) - static_cast<int64_t>(gamma.shape().size());
  for (size_t i = 0; i < out.size(); ++i) {
    if (static_cast<int64_t>(i) >= param_dim) {
      out[i] = 1;
    }
  }
  return out;
}
}  // namespace

std::vector<ms::Tensor> npu_gemma_rms_norm(const ms::Tensor &self, const ms::Tensor &gamma,
                                           double epsilon = 1e-6) {
  auto y = ms::Tensor(self.data_type(), self.shape());
  auto rstd = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ParamReducedShape(self, gamma));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GemmaRmsNorm");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGemmaRmsNorm, self, gamma, epsilon, y, rstd));
  runner->Run({self, gamma}, {y, rstd});
  return {y, rstd};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_gemma_rms_norm", PYBOOST_CALLER(2, custom::npu_gemma_rms_norm));
}
}  // namespace custom
