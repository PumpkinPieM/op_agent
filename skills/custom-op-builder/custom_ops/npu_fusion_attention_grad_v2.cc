#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
ms::Tensor OptionalLike(const std::optional<ms::Tensor> &tensor_opt, ms::TypeId dtype) {
  if (tensor_opt.has_value()) {
    return ms::Tensor(dtype, tensor_opt.value().shape());
  }
  return ms::Tensor(dtype, std::vector<int64_t>{0});
}
}  // namespace

std::vector<ms::Tensor> npu_fusion_attention_grad_v2(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &dy, int64_t head_num,
    const std::string &input_layout, const std::optional<ms::Tensor> &pse_opt,
    const std::optional<ms::Tensor> &padding_mask_opt, const std::optional<ms::Tensor> &atten_mask_opt,
    const std::optional<ms::Tensor> &softmax_max_opt, const std::optional<ms::Tensor> &softmax_sum_opt,
    const std::optional<ms::Tensor> &softmax_in_opt, const std::optional<ms::Tensor> &attention_in_opt,
    const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt, double scale_value,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, int64_t seed, int64_t offset,
    int64_t numels, const std::optional<std::vector<int64_t>> &prefix_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_kvlen_opt, int64_t sparse_mode, bool gen_mask_parallel,
    bool sync, int64_t pse_type, const std::optional<std::vector<int64_t>> &q_start_idx_opt,
    const std::optional<std::vector<int64_t>> &kv_start_idx_opt, const std::string &softmax_layout,
    const std::optional<ms::Tensor> &sink_opt) {
  auto pse_value = pse_opt.value_or(ms::Tensor());
  auto padding_mask_value = padding_mask_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto softmax_max_value = softmax_max_opt.value_or(ms::Tensor());
  auto softmax_sum_value = softmax_sum_opt.value_or(ms::Tensor());
  auto softmax_in_value = softmax_in_opt.value_or(ms::Tensor());
  auto attention_in_value = attention_in_opt.value_or(ms::Tensor());
  auto query_rope_value = query_rope_opt.value_or(ms::Tensor());
  auto key_rope_value = key_rope_opt.value_or(ms::Tensor());
  auto prefix = std::make_pair(prefix_opt.value_or(std::vector<int64_t>{}), true);
  auto q_start_idx = std::make_pair(q_start_idx_opt.value_or(std::vector<int64_t>{}), true);
  auto kv_start_idx = std::make_pair(kv_start_idx_opt.value_or(std::vector<int64_t>{}), true);

  auto dq = ms::Tensor(query.data_type(), query.shape());
  auto dk = ms::Tensor(key.data_type(), key.shape());
  auto dv = ms::Tensor(value.data_type(), value.shape());
  auto dpse = OptionalLike(pse_opt, query.data_type());
  auto dq_rope = OptionalLike(query_rope_opt, query.data_type());
  auto dk_rope = OptionalLike(key_rope_opt, key.data_type());
  auto dsink = OptionalLike(sink_opt, query.data_type());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlashAttentionScoreGradV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlashAttentionScoreGradV2, query, key, value, dy, pse_opt,
                                          std::optional<ms::Tensor>(), padding_mask_opt, atten_mask_opt,
                                          softmax_max_opt, softmax_sum_opt, softmax_in_opt, attention_in_opt, prefix,
                                          q_start_idx, kv_start_idx, scale_value, keep_prob, pre_tokens, next_tokens,
                                          head_num, input_layout, inner_precise, sparse_mode, pse_type, dq, dk, dv,
                                          dpse));
  runner->Run({query, key, value, dy, pse_value, padding_mask_value, atten_mask_value, softmax_max_value,
               softmax_sum_value, softmax_in_value, attention_in_value, query_rope_value, key_rope_value},
              {dq, dk, dv, dpse});
  return {dq, dk, dv, dpse, dq_rope, dk_rope, dsink};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fusion_attention_grad_v2", PYBOOST_CALLER(7, custom::npu_fusion_attention_grad_v2));
}
}  // namespace custom
