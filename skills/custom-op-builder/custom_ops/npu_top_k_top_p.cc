#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_top_k_top_p(const ms::Tensor &logits, const std::optional<ms::Tensor> &p_opt = std::nullopt,
                           const std::optional<ms::Tensor> &k_opt = std::nullopt) {
  auto p = p_opt.value_or(ms::Tensor());
  auto k = k_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(logits.data_type(), logits.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ApplyTopKTopP");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyTopKTopP, logits, p_opt, k_opt, out));
  runner->Run({logits, p, k}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_top_k_top_p", PYBOOST_CALLER(1, custom::npu_top_k_top_p));
}
}  // namespace custom
