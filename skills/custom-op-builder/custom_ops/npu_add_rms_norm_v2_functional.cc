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

std::vector<ms::Tensor> npu_add_rms_norm_v2_functional(const ms::Tensor &x1, const ms::Tensor &x2,
                                                       const ms::Tensor &gamma, double epsilon = 1e-6) {
  auto x1_out = x1;
  auto x2_out = x2;
  auto y = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ParamReducedShape(x1, gamma));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("InplaceAddRmsNorm");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnInplaceAddRmsNorm, x1_out, x2_out, gamma, epsilon, y));
  runner->Run({x1_out, x2_out, gamma}, {y, x1_out, x2_out});
  return {y, x1_out, x2_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_add_rms_norm_v2_functional", PYBOOST_CALLER(3, custom::npu_add_rms_norm_v2_functional));
}
}  // namespace custom
