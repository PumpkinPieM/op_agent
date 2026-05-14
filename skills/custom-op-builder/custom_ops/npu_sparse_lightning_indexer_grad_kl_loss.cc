#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {

std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query_index,
                                                                            const ms::Tensor &key_index,
                                                                            const ms::Tensor &weights) {
  auto d_query_index = ms::Tensor(query_index.data_type(), query_index.shape());
  auto d_key_index = ms::Tensor(key_index.data_type(), key_index.shape());
  auto d_weights = ms::Tensor(weights.data_type(), weights.shape());
  auto loss = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{1});
  return std::make_tuple(std::move(d_query_index), std::move(d_key_index), std::move(d_weights), std::move(loss));
}

}  // namespace

std::vector<ms::Tensor> npu_sparse_lightning_indexer_grad_kl_loss(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &query_index, const ms::Tensor &key_index,
    const ms::Tensor &weights, const ms::Tensor &sparse_indices, const ms::Tensor &softmax_max,
    const ms::Tensor &softmax_sum, double scale_value,
    const std::optional<ms::Tensor> &query_rope_opt = std::nullopt,
    const std::optional<ms::Tensor> &key_rope_opt = std::nullopt,
    const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt = std::nullopt,
    const std::optional<std::vector<int64_t>> &actual_seq_klen_opt = std::nullopt,
    const std::optional<std::string> &layout_opt = std::nullopt,
    const std::optional<int64_t> &sparse_mode_opt = std::nullopt,
    const std::optional<int64_t> &pre_tokens_opt = std::nullopt,
    const std::optional<int64_t> &next_tokens_opt = std::nullopt) {
  constexpr int64_t kMaxInt64 = 9223372036854775807LL;
  auto query_rope = query_rope_opt.value_or(ms::Tensor());
  auto key_rope = key_rope_opt.value_or(ms::Tensor());
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_klen = std::make_pair(actual_seq_klen_opt.value_or(std::vector<int64_t>{}), true);
  auto layout = layout_opt.value_or("BSND");
  auto layout_ptr = const_cast<char *>(layout.c_str());
  auto sparse_mode = sparse_mode_opt.value_or(3);
  auto pre_tokens = pre_tokens_opt.value_or(kMaxInt64);
  auto next_tokens = next_tokens_opt.value_or(kMaxInt64);
  constexpr bool deterministic = true;
  auto [d_query_index, d_key_index, d_weights, loss] = GenResultTensors(query_index, key_index, weights);

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("SparseLightningIndexerGradKLLoss");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnSparseLightningIndexerGradKLLoss, query, key, query_index, key_index,
                                          weights, sparse_indices, softmax_max, softmax_sum, query_rope_opt,
                                          key_rope_opt, actual_seq_qlen, actual_seq_klen, scale_value, layout_ptr,
                                          sparse_mode, pre_tokens, next_tokens, deterministic, d_query_index,
                                          d_key_index, d_weights, loss));
  runner->Run({query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum, query_rope,
               key_rope},
              {d_query_index, d_key_index, d_weights, loss});
  return {d_query_index, d_key_index, d_weights, loss};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_sparse_lightning_indexer_grad_kl_loss",
        PYBOOST_CALLER(4, custom::npu_sparse_lightning_indexer_grad_kl_loss));
}
}  // namespace custom
