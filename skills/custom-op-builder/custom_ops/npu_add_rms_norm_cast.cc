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

std::vector<ms::Tensor> npu_add_rms_norm_cast(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &gamma,
                                              double epsilon = 1e-6) {
  auto out0 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, x1.shape());
  auto out1 = ms::Tensor(x1.data_type(), x1.shape());
  auto out2 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ParamReducedShape(x1, gamma));
  auto out3 = ms::Tensor(x1.data_type(), x1.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AddRmsNormCast");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAddRmsNormCast, x1, x2, gamma, epsilon, out0, out1, out2, out3));
  runner->Run({x1, x2, gamma}, {out0, out1, out2, out3});
  return {out0, out1, out2, out3};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_add_rms_norm_cast", PYBOOST_CALLER(4, custom::npu_add_rms_norm_cast));
}
}  // namespace custom
