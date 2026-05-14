#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_block_sparse_attention(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &block_sparse_mask, const std::vector<int64_t> &block_shape, const std::string &q_input_layout, const std::string &kv_input_layout, int64_t num_key_value_heads, double scale_value, int64_t inner_precise, const std::optional<std::vector<int64_t>> &actual_seq_lengths_opt, const std::optional<std::vector<int64_t>> &actual_seq_lengths_kv_opt, const std::optional<int64_t> &softmax_lse_flag_opt) {
  auto actual_seq_lengths = actual_seq_lengths_opt.value_or(std::vector<int64_t>{});
  auto actual_seq_lengths_kv = actual_seq_lengths_kv_opt.value_or(std::vector<int64_t>{});
  auto softmax_lse_flag = softmax_lse_flag_opt.value_or(0);
  auto base_shape = query.shape();
  auto base_dtype = query.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("BlockSparseAttention");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnBlockSparseAttention, query, key, value, block_sparse_mask, block_shape, q_input_layout, kv_input_layout, num_key_value_heads, scale_value, inner_precise, actual_seq_lengths, actual_seq_lengths_kv, softmax_lse_flag, out0, out1));
  runner->Run({query, key, value, block_sparse_mask}, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_block_sparse_attention", PYBOOST_CALLER(2, custom::npu_block_sparse_attention));
}
}  // namespace custom
