#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> _npu_distribute_barrier(const ms::Tensor &x_ref, const std::string &group, int64_t world_size, const std::optional<ms::Tensor> &time_out_opt = std::nullopt, const std::optional<ms::Tensor> &elastic_info_opt = std::nullopt) {
  auto time_out = time_out_opt.value_or(ms::Tensor());
  auto elastic_info = elastic_info_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(x_ref.data_type(), x_ref.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DistributeBarrierV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDistributeBarrierV2, x_ref, group, world_size, time_out_opt, elastic_info_opt, out));
  runner->Run({x_ref, time_out, elastic_info}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_distribute_barrier", PYBOOST_CALLER(1, custom::_npu_distribute_barrier));
}
}  // namespace custom
