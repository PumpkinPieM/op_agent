#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> AttentionInferOutShape(const ms::Tensor &query, const ms::Tensor &value,
                                            int64_t key_value_head_num, const std::string &layout) {
  const int64_t value_head_dim = value.shape()[2] / key_value_head_num;
  if (layout == "BSND") {
    return {query.shape()[0], query.shape()[1], query.shape()[2], value_head_dim};
  }
  return {query.shape()[0], query.shape()[1], value_head_dim};
}

std::vector<int64_t> AttentionInferTopkShape(const ms::Tensor &query, int64_t key_value_head_num,
                                             int64_t select_block_count, const std::string &layout) {
  if (layout == "BSND") {
    return {query.shape()[0], query.shape()[1], key_value_head_num, select_block_count};
  }
  return {query.shape()[0], key_value_head_num, select_block_count};
}
}  // namespace

std::vector<ms::Tensor> npu_nsa_compress_attention_infer(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, double scale_value, int64_t head_num,
    int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size,
    int64_t compress_block_size, int64_t compress_stride, const std::optional<std::string> &layout_opt,
    const std::optional<ms::Tensor> &atten_mask_opt, const std::optional<ms::Tensor> &block_table_opt,
    const std::optional<ms::Tensor> &topk_mask_opt, const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt,
    const std::optional<std::vector<int64_t>> &actual_cmp_seq_kvlen_opt,
    const std::optional<std::vector<int64_t>> &actual_sel_seq_kvlen_opt) {
  std::string layout = layout_opt.value_or("TND");
  auto out = ms::Tensor(query.data_type(), AttentionInferOutShape(query, value, key_value_head_num, layout));
  auto topk_indices = ms::Tensor(ms::TypeId::kNumberTypeInt32,
                                 AttentionInferTopkShape(query, key_value_head_num, select_block_count, layout));

  auto atten_mask = atten_mask_opt.value_or(ms::Tensor());
  auto block_table = block_table_opt.value_or(ms::Tensor());
  auto topk_mask = topk_mask_opt.value_or(ms::Tensor());
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_cmp_seq_kvlen = std::make_pair(actual_cmp_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_sel_seq_kvlen = std::make_pair(actual_sel_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);
  int64_t sparse_mode = 0;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("NsaCompressAttentionInfer");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnNsaCompressAttentionInfer, query, key, value, atten_mask_opt,
                                          block_table_opt, actual_seq_qlen, actual_cmp_seq_kvlen,
                                          actual_sel_seq_kvlen, topk_mask_opt, head_num, key_value_head_num,
                                          select_block_size, select_block_count, compress_block_size,
                                          compress_stride, scale_value, layout, page_block_size, sparse_mode, out,
                                          topk_indices));
  runner->Run({query, key, value, atten_mask, block_table, topk_mask}, {out, topk_indices});
  return {out, topk_indices};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_nsa_compress_attention_infer", PYBOOST_CALLER(2, custom::npu_nsa_compress_attention_infer));
}
}  // namespace custom
