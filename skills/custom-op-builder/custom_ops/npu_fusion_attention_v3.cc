#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_fusion_attention_v3(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, int64_t head_num, const std::string &input_layout, const std::optional<ms::Tensor> &pse_opt, const std::optional<ms::Tensor> &padding_mask_opt, const std::optional<ms::Tensor> &atten_mask_opt, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, const std::optional<std::vector<int64_t>> &prefix_opt, const std::optional<ms::Tensor> &actual_seq_qlen_opt, const std::optional<ms::Tensor> &actual_seq_kvlen_opt, int64_t sparse_mode, bool gen_mask_parallel, bool sync, const std::string &softmax_layout, const std::optional<ms::Tensor> &sink_opt) {
  auto pse_value = pse_opt.value_or(ms::Tensor());
  auto padding_mask_value = padding_mask_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
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
  auto out5 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlashAttentionScore");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlashAttentionScore, query, key, value, head_num, input_layout, pse_opt, padding_mask_opt, atten_mask_opt, scale, keep_prob, pre_tockens, next_tockens, inner_precise, prefix, actual_seq_qlen_opt, actual_seq_kvlen_opt, sparse_mode, gen_mask_parallel, sync, softmax_layout, sink_opt, out0, out1, out2, out3, out4, out5));
  runner->Run({query, key, value, pse_value, padding_mask_value, atten_mask_value, actual_seq_qlen_value, actual_seq_kvlen_value, sink_value}, {out0, out1, out2, out3, out4, out5});
  return {out0, out1, out2, out3, out4, out5};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fusion_attention_v3", PYBOOST_CALLER(6, custom::npu_fusion_attention_v3));
}
}  // namespace custom
