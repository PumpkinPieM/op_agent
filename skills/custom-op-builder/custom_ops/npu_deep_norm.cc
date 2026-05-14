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

std::vector<ms::Tensor> npu_deep_norm(const ms::Tensor &x, const ms::Tensor &gx, const ms::Tensor &beta,
                                      const ms::Tensor &gamma, double alpha = 0.3, double epsilon = 1e-6) {
  auto mean = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ParamReducedShape(x, gamma));
  auto rstd = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ParamReducedShape(x, gamma));
  auto y = ms::Tensor(x.data_type(), x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DeepNorm");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDeepNorm, x, gx, beta, gamma, alpha, epsilon, mean, rstd, y));
  runner->Run({x, gx, beta, gamma}, {mean, rstd, y});
  return {mean, rstd, y};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_deep_norm", PYBOOST_CALLER(3, custom::npu_deep_norm));
}
}  // namespace custom
