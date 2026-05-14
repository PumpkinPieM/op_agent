#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_ffn(const ms::Tensor &x, const ms::Tensor &weight1, const ms::Tensor &weight2, const std::string &activation, const std::optional<std::vector<int64_t>> &expert_tokens_opt, const std::optional<std::vector<int64_t>> &expert_tokens_index_opt, const std::optional<ms::Tensor> &bias1_opt, const std::optional<ms::Tensor> &bias2_opt, const std::optional<ms::Tensor> &scale_opt, const std::optional<ms::Tensor> &offset_opt, const std::optional<ms::Tensor> &deq_scale1_opt, const std::optional<ms::Tensor> &deq_scale2_opt, const std::optional<ms::Tensor> &antiquant_scale1_opt, const std::optional<ms::Tensor> &antiquant_scale2_opt, const std::optional<ms::Tensor> &antiquant_offset1_opt, const std::optional<ms::Tensor> &antiquant_offset2_opt, const std::optional<int64_t> &inner_precise_opt, const std::optional<int64_t> &output_dtype_opt) {
  auto expert_tokens = expert_tokens_opt.value_or(std::vector<int64_t>{});
  auto expert_tokens_index = expert_tokens_index_opt.value_or(std::vector<int64_t>{});
  auto bias1_value = bias1_opt.value_or(ms::Tensor());
  auto bias2_value = bias2_opt.value_or(ms::Tensor());
  auto scale_value = scale_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto deq_scale1_value = deq_scale1_opt.value_or(ms::Tensor());
  auto deq_scale2_value = deq_scale2_opt.value_or(ms::Tensor());
  auto antiquant_scale1_value = antiquant_scale1_opt.value_or(ms::Tensor());
  auto antiquant_scale2_value = antiquant_scale2_opt.value_or(ms::Tensor());
  auto antiquant_offset1_value = antiquant_offset1_opt.value_or(ms::Tensor());
  auto antiquant_offset2_value = antiquant_offset2_opt.value_or(ms::Tensor());
  auto inner_precise = inner_precise_opt.value_or(0);
  (void)output_dtype_opt;
  auto out_shape = x.shape();
  out_shape.back() = weight2.shape().back();
  auto out0 = ms::Tensor(x.data_type(), out_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FFNV2");
  bool tokens_index_flag = expert_tokens_index_opt.has_value();
  auto expert_tokens_value = tokens_index_flag ? expert_tokens_index : expert_tokens;
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFFNV2, x, weight1, weight2, expert_tokens_value, bias1_opt, bias2_opt,
                                          scale_opt, offset_opt, deq_scale1_opt, deq_scale2_opt, antiquant_scale1_opt,
                                          antiquant_scale2_opt, antiquant_offset1_opt, antiquant_offset2_opt, activation,
                                          inner_precise, tokens_index_flag, out0));
  runner->Run({x, weight1, weight2, bias1_value, bias2_value, scale_value, offset_value, deq_scale1_value, deq_scale2_value, antiquant_scale1_value, antiquant_scale2_value, antiquant_offset1_value, antiquant_offset2_value}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_ffn", PYBOOST_CALLER(1, custom::npu_ffn));
}
}  // namespace custom
