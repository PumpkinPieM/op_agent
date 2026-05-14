#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_fusion_attention_grad_v2(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &dy, int64_t head_num, const std::string &input_layout, const std::optional<ms::Tensor> &pse_opt, const std::optional<ms::Tensor> &padding_mask_opt, const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<ms::Tensor> &softmax_max_opt, const std::optional<ms::Tensor> &softmax_sum_opt, const std::optional<ms::Tensor> &softmax_in_opt, const std::optional<ms::Tensor> &attention_in_opt, const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt, double scale_value, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, const std::optional<std::vector<int64_t>> &prefix_opt, const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt, const std::optional<std::vector<int64_t>> &actual_seq_kvlen_opt, int64_t sparse_mode, bool gen_mask_parallel, bool sync, int64_t pse_type, const std::optional<std::vector<int64_t>> &q_start_idx_opt, const std::optional<std::vector<int64_t>> &kv_start_idx_opt, const std::string &softmax_layout, const std::optional<ms::Tensor> &sink_opt) {
  auto pse_value = pse_opt.value_or(ms::Tensor());
  auto padding_mask_value = padding_mask_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto softmax_max_value = softmax_max_opt.value_or(ms::Tensor());
  auto softmax_sum_value = softmax_sum_opt.value_or(ms::Tensor());
  auto softmax_in_value = softmax_in_opt.value_or(ms::Tensor());
  auto attention_in_value = attention_in_opt.value_or(ms::Tensor());
  auto query_rope_value = query_rope_opt.value_or(ms::Tensor());
  auto key_rope_value = key_rope_opt.value_or(ms::Tensor());
  auto prefix = prefix_opt.value_or(std::vector<int64_t>{});
  auto actual_seq_qlen = actual_seq_qlen_opt.value_or(std::vector<int64_t>{});
  auto actual_seq_kvlen = actual_seq_kvlen_opt.value_or(std::vector<int64_t>{});
  auto q_start_idx = q_start_idx_opt.value_or(std::vector<int64_t>{});
  auto kv_start_idx = kv_start_idx_opt.value_or(std::vector<int64_t>{});
  auto sink_value = sink_opt.value_or(ms::Tensor());
  auto base_shape = query.shape();
  auto base_dtype = query.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto out3 = ms::Tensor(base_dtype, base_shape);
  auto out4 = ms::Tensor(base_dtype, base_shape);
  auto out5 = ms::Tensor(base_dtype, base_shape);
  auto out6 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlashAttentionScoreGradV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlashAttentionScoreGradV2, query, key, value, dy, head_num, input_layout, pse_opt, padding_mask_opt, atten_mask_opt, softmax_max_opt, softmax_sum_opt, softmax_in_opt, attention_in_opt, query_rope_opt, key_rope_opt, scale_value, keep_prob, pre_tokens, next_tokens, inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx, softmax_layout, sink_opt, out0, out1, out2, out3, out4, out5, out6));
  runner->Run({query, key, value, dy, pse_value, padding_mask_value, atten_mask_value, softmax_max_value, softmax_sum_value, softmax_in_value, attention_in_value, query_rope_value, key_rope_value, sink_value}, {out0, out1, out2, out3, out4, out5, out6});
  return {out0, out1, out2, out3, out4, out5, out6};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fusion_attention_grad_v2", PYBOOST_CALLER(7, custom::npu_fusion_attention_grad_v2));
}
}  // namespace custom
