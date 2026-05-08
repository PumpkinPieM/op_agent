#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_update_expert(const ms::Tensor &expert_ids, const ms::Tensor &expert_scales, int64_t expert_num) {
  auto out = ms::Tensor(expert_ids.data_type(), expert_ids.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeUpdateExpert");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeUpdateExpert, expert_ids, expert_scales, expert_num, out));
  runner->Run({expert_ids, expert_scales}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_update_expert", PYBOOST_CALLER(1, custom::npu_moe_update_expert));
}
}  // namespace custom
