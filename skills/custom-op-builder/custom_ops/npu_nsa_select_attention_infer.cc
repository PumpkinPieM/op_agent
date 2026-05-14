#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> OutputShape(const ms::Tensor &query, const ms::Tensor &value, int64_t head_num,
                                 int64_t key_value_head_num, const std::string &layout) {
  const auto query_shape = query.shape();
  const auto value_shape = value.shape();
  if (layout == "BSH" && query_shape.size() >= 3 && value_shape.size() >= 3 && key_value_head_num > 0) {
    const auto key_head_dim = value_shape[2] / key_value_head_num;
    return {query_shape[0], query_shape[1], head_num * key_head_dim};
  }
  if (layout == "TND" && query_shape.size() >= 2 && value_shape.size() >= 3 && key_value_head_num > 0) {
    const auto key_head_dim = value_shape[2] / key_value_head_num;
    return {query_shape[0], query_shape[1], key_head_dim};
  }
  if (query_shape.size() >= 3 && value_shape.size() >= 4) {
    return {query_shape[0], query_shape[1], query_shape[2], value_shape[3]};
  }
  return query_shape;
}
}  // namespace

ms::Tensor npu_nsa_select_attention_infer(const ms::Tensor & query, const ms::Tensor & key, const ms::Tensor & value, const ms::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, const std::optional<std::string> & layout_opt, const std::optional<ms::Tensor> & atten_mask_opt, const std::optional<ms::Tensor> & block_table_opt, const std::optional<std::vector<int64_t>> & actual_seq_qlen_opt, const std::optional<std::vector<int64_t>> & actual_seq_kvlen_opt) {
  auto layout=layout_opt.value_or("BSND");
  auto out=ms::Tensor(query.data_type(), OutputShape(query, value, head_num, key_value_head_num, layout));
  auto atten_mask=atten_mask_opt.value_or(ms::Tensor());
  auto block_table=block_table_opt.value_or(ms::Tensor());
  auto actual_seq_qlen=std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_kvlen=std::make_pair(actual_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);
  int64_t sparse_mode = 0;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("NsaSelectedAttentionInfer");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnNsaSelectedAttentionInfer, query, key, value, topk_indices,
                                          atten_mask_opt, block_table_opt, actual_seq_qlen, actual_seq_kvlen, layout,
                                          head_num, key_value_head_num, select_block_size, select_block_count,
                                          page_block_size, scale_value, sparse_mode, out));
  runner->Run({query,key,value,topk_indices,atten_mask,block_table}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_nsa_select_attention_infer", PYBOOST_CALLER(1, custom::npu_nsa_select_attention_infer));
}
}  // namespace custom
