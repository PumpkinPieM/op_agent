#include <vector>
#include <optional>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> OutputShape(const ms::Tensor &expanded_permuted_rows,
                                 const std::optional<ms::Tensor> &scales_opt,
                                 const ms::Tensor &expanded_src_to_dst_row) {
  const auto &expanded_shape = expanded_permuted_rows.shape();
  int64_t rows = expanded_src_to_dst_row.shape().empty() ? 0 : expanded_src_to_dst_row.shape()[0];
  if (scales_opt.has_value() && !scales_opt.value().shape().empty()) {
    rows = scales_opt.value().shape()[0];
  }
  const int64_t hidden = expanded_shape.size() == 3 ? expanded_shape[2] : expanded_shape[1];
  return {rows, hidden};
}
}  // namespace

std::vector<ms::Tensor> npu_moe_finalize_routing(
    const ms::Tensor &expanded_permuted_rows, const std::optional<ms::Tensor> &skip1_opt,
    const std::optional<ms::Tensor> &skip2_opt, const std::optional<ms::Tensor> &bias_opt,
    const std::optional<ms::Tensor> &scales_opt, const ms::Tensor &expanded_src_to_dst_row,
    const std::optional<ms::Tensor> &expert_for_source_row_opt, const std::optional<int64_t> &drop_pad_mode_opt) {
  auto skip1 = skip1_opt.value_or(ms::Tensor());
  auto skip2 = skip2_opt.value_or(ms::Tensor());
  auto bias = bias_opt.value_or(ms::Tensor());
  auto scales = scales_opt.value_or(ms::Tensor());
  auto expert_for_source_row = expert_for_source_row_opt.value_or(ms::Tensor());
  auto drop_pad_mode = drop_pad_mode_opt.value_or(0);
  auto out = ms::Tensor(expanded_permuted_rows.data_type(),
                        OutputShape(expanded_permuted_rows, scales_opt, expanded_src_to_dst_row));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeFinalizeRoutingV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeFinalizeRoutingV2, expanded_permuted_rows, expanded_src_to_dst_row,
                                          skip1_opt, skip2_opt, bias_opt, scales_opt, expert_for_source_row_opt,
                                          drop_pad_mode, out));
  runner->Run({expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, expert_for_source_row}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_finalize_routing", PYBOOST_CALLER(1, custom::npu_moe_finalize_routing));
}
}  // namespace custom
