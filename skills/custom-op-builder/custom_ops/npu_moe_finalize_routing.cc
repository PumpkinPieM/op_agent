#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_moe_finalize_routing(const ms::Tensor &expanded_permuted_rows, const ms::Tensor &skip1, const ms::Tensor &skip2, const ms::Tensor &bias, const ms::Tensor &scales, const ms::Tensor &expanded_src_to_dst_row, const ms::Tensor &expert_for_source_row, int64_t drop_pad_mode = 0) {
  auto out = ms::Tensor(expanded_permuted_rows.data_type(), expanded_permuted_rows.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeFinalizeRoutingV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeFinalizeRoutingV2, expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode, out));
  runner->Run({expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_finalize_routing", PYBOOST_CALLER(1, custom::npu_moe_finalize_routing));
}
}  // namespace custom
