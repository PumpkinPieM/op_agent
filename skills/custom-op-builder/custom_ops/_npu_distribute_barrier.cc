#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
bool HasOpApi(const char *api_name) { return mindspore::device::ascend::GetOpApiFunc(api_name) != nullptr; }
}  // namespace

std::vector<ms::Tensor> _npu_distribute_barrier(const ms::Tensor &x_ref, const std::string &group, int64_t world_size, const std::optional<ms::Tensor> &time_out_opt = std::nullopt, const std::optional<ms::Tensor> &elastic_info_opt = std::nullopt) {
  auto time_out = time_out_opt.value_or(ms::Tensor());
  auto elastic_info = elastic_info_opt.value_or(ms::Tensor());
  if (HasOpApi("aclnnDistributeBarrierV2GetWorkspaceSize")) {
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DistributeBarrierV2");
    runner->SetLaunchFunc(
      LAUNCH_ACLNN_FUNC(aclnnDistributeBarrierV2, x_ref, time_out_opt, elastic_info_opt, group, world_size));
    runner->Run({x_ref, time_out, elastic_info}, {x_ref});
    return {x_ref};
  }
  if (time_out_opt.has_value() || elastic_info_opt.has_value()) {
    throw std::runtime_error("The aclnnDistributeBarrier does not support time_out and elastic_info");
  }
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DistributeBarrier");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDistributeBarrier, x_ref, group, world_size));
  runner->Run({x_ref}, {x_ref});
  return {x_ref};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_distribute_barrier", PYBOOST_CALLER(1, custom::_npu_distribute_barrier));
}
}  // namespace custom
