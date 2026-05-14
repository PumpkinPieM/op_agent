#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_fusion_attention_grad_v3(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &dy, int64_t head_num, const std::string &input_layout, const std::optional<ms::Tensor> &pse_opt, const std::optional<ms::Tensor> &padding_mask_opt, const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<ms::Tensor> &softmax_max_opt, const std::optional<ms::Tensor> &softmax_sum_opt, const std::optional<ms::Tensor> &softmax_in_opt, const std::optional<ms::Tensor> &attention_in_opt, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, const std::optional<ms::Tensor> &seed_opt, const std::optional<ms::Tensor> &offset_opt, const std::optional<std::vector<int64_t>> &prefix_opt, const std::optional<ms::Tensor> &actual_seq_qlen_opt, const std::optional<ms::Tensor> &actual_seq_kvlen_opt, int64_t sparse_mode, bool gen_mask_parallel, bool sync, const std::string &softmax_layout, const std::optional<ms::Tensor> &sink_opt) {
  auto pse_value = pse_opt.value_or(ms::Tensor());
  auto padding_mask_value = padding_mask_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto softmax_max_value = softmax_max_opt.value_or(ms::Tensor());
  auto softmax_sum_value = softmax_sum_opt.value_or(ms::Tensor());
  auto softmax_in_value = softmax_in_opt.value_or(ms::Tensor());
  auto attention_in_value = attention_in_opt.value_or(ms::Tensor());
  auto seed_value = seed_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto prefix = prefix_opt.value_or(std::vector<int64_t>{});
  auto actual_seq_qlen_value = actual_seq_qlen_opt.value_or(ms::Tensor());
  auto actual_seq_kvlen_value = actual_seq_kvlen_opt.value_or(ms::Tensor());
  auto sink_value = sink_opt.value_or(ms::Tensor());
  auto base_shape = query.shape();
  auto base_dtype = query.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto out3 = ms::Tensor(base_dtype, base_shape);
  auto out4 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlashAttentionScoreGrad");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlashAttentionScoreGrad, query, key, value, dy, head_num, input_layout, pse_opt, padding_mask_opt, atten_mask_opt, softmax_max_opt, softmax_sum_opt, softmax_in_opt, attention_in_opt, scale_value, keep_prob, pre_tockens, next_tockens, inner_precise, seed_opt, offset_opt, prefix, actual_seq_qlen_opt, actual_seq_kvlen_opt, sparse_mode, gen_mask_parallel, sync, softmax_layout, sink_opt, out0, out1, out2, out3, out4));
  runner->Run({query, key, value, dy, pse_value, padding_mask_value, atten_mask_value, softmax_max_value, softmax_sum_value, softmax_in_value, attention_in_value, seed_value, offset_value, actual_seq_qlen_value, actual_seq_kvlen_value, sink_value}, {out0, out1, out2, out3, out4});
  return {out0, out1, out2, out3, out4};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fusion_attention_grad_v3", PYBOOST_CALLER(5, custom::npu_fusion_attention_grad_v3));
}
}  // namespace custom
