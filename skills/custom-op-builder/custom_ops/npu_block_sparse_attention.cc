#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> SoftmaxLseShape(const ms::Tensor &query) {
  const auto &shape = query.shape();
  if (shape.size() == 4) {
    return {shape[0], shape[1], shape[2], 1};
  }
  return {shape[0], shape[1], 1};
}
}  // namespace

std::vector<ms::Tensor> npu_block_sparse_attention(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &block_sparse_mask, const std::vector<int64_t> &block_shape, const std::string &q_input_layout, const std::string &kv_input_layout, int64_t num_key_value_heads, double scale_value, int64_t inner_precise, const std::optional<std::vector<int64_t>> &actual_seq_lengths_opt, const std::optional<std::vector<int64_t>> &actual_seq_lengths_kv_opt, const std::optional<int64_t> &softmax_lse_flag_opt) {
  auto actual_seq_lengths = std::make_pair(actual_seq_lengths_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_lengths_kv = std::make_pair(actual_seq_lengths_kv_opt.value_or(std::vector<int64_t>{}), true);
  auto softmax_lse_flag = softmax_lse_flag_opt.value_or(0);
  auto attention_out = ms::Tensor(query.data_type(), query.shape());
  auto softmax_lse = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxLseShape(query));
  auto atten_mask = ms::Tensor();
  auto block_table = ms::Tensor();
  auto q_layout = const_cast<char *>(q_input_layout.c_str());
  auto kv_layout = const_cast<char *>(kv_input_layout.c_str());
  constexpr int64_t mask_type = 0;
  constexpr int64_t block_size = 0;
  constexpr int64_t pre_tokens = 2147483647;
  constexpr int64_t next_tokens = 2147483647;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("BlockSparseAttention");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnBlockSparseAttention, query, key, value, block_sparse_mask, atten_mask,
                                        block_shape, actual_seq_lengths, actual_seq_lengths_kv, block_table,
                                        q_layout, kv_layout, num_key_value_heads, mask_type, scale_value,
                                        inner_precise, block_size, pre_tokens, next_tokens, softmax_lse_flag,
                                        attention_out, softmax_lse));
  runner->Run({query, key, value, block_sparse_mask, atten_mask, block_table}, {attention_out, softmax_lse});
  return {attention_out, softmax_lse};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_block_sparse_attention", PYBOOST_CALLER(2, custom::npu_block_sparse_attention));
}
}  // namespace custom
