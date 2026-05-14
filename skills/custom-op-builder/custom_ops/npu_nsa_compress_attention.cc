#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> AttentionOutShape(const ms::Tensor &query, const ms::Tensor &value) {
  auto shape = query.shape();
  shape.back() = value.shape().back();
  return shape;
}

std::vector<int64_t> TopkShape(const ms::Tensor &query, const ms::Tensor &key, int64_t select_block_count) {
  return {query.shape()[0], key.shape()[1], select_block_count};
}

std::vector<int64_t> SoftmaxShape(const ms::Tensor &query) {
  return {query.shape()[0], query.shape()[1], 8};
}
}  // namespace

std::vector<ms::Tensor> npu_nsa_compress_attention(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, double scale_value, int64_t head_num,
    int64_t compress_block_size, int64_t compress_stride, int64_t select_block_size, int64_t select_block_count,
    const std::optional<ms::Tensor> &topk_mask_opt, const std::optional<ms::Tensor> &atten_mask_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt,
    const std::optional<std::vector<int64_t>> &actual_cmp_seq_kvlen_opt,
    const std::optional<std::vector<int64_t>> &actual_sel_seq_kvlen_opt) {
  auto attention_out = ms::Tensor(query.data_type(), AttentionOutShape(query, value));
  auto topk_indices = ms::Tensor(ms::TypeId::kNumberTypeInt32, TopkShape(query, key, select_block_count));
  auto softmax_max = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxShape(query));
  auto softmax_sum = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxShape(query));

  auto topk_mask = topk_mask_opt.value_or(ms::Tensor());
  auto atten_mask = atten_mask_opt.value_or(ms::Tensor());
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_cmp_seq_kvlen = std::make_pair(actual_cmp_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_sel_seq_kvlen = std::make_pair(actual_sel_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);
  std::string layout = "TND";
  int64_t sparse_mode = 1;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("NsaCompressAttention");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnNsaCompressAttention, query, key, value, atten_mask_opt, topk_mask_opt,
                                          actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen, scale_value,
                                          head_num, layout, sparse_mode, compress_block_size, compress_stride,
                                          select_block_size, select_block_count, softmax_max, softmax_sum,
                                          attention_out, topk_indices));
  runner->Run({query, key, value, topk_mask, atten_mask}, {softmax_max, softmax_sum, attention_out, topk_indices});
  return {attention_out, topk_indices, softmax_max, softmax_sum};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_nsa_compress_attention", PYBOOST_CALLER(4, custom::npu_nsa_compress_attention));
}
}  // namespace custom
