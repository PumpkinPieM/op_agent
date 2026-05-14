#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {

std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query_index, const ms::Tensor &key_index,
                                                    const std::string &layout) {
  std::vector<int64_t> out_shape;
  if (layout == "TND") {
    out_shape = {key_index.shape().at(1), query_index.shape().at(0)};
  } else {
    out_shape = {query_index.shape().at(0), key_index.shape().at(2), query_index.shape().at(1)};
  }
  auto softmax_max_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, out_shape);
  auto softmax_sum_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, out_shape);
  return std::make_tuple(std::move(softmax_max_out), std::move(softmax_sum_out));
}

}  // namespace

std::vector<ms::Tensor> npu_dense_lightning_indexer_softmax_lse(
    const ms::Tensor &query_index, const ms::Tensor &key_index, const ms::Tensor &weights,
    const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt = std::nullopt,
    const std::optional<std::vector<int64_t>> &actual_seq_klen_opt = std::nullopt,
    const std::optional<std::string> &layout_opt = std::nullopt,
    const std::optional<int64_t> &sparse_mode_opt = std::nullopt,
    const std::optional<int64_t> &pre_tokens_opt = std::nullopt,
    const std::optional<int64_t> &next_tokens_opt = std::nullopt) {
  constexpr int64_t kMaxInt64 = 9223372036854775807LL;
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_klen = std::make_pair(actual_seq_klen_opt.value_or(std::vector<int64_t>{}), true);
  std::string layout = layout_opt.value_or("BSND");
  auto sparse_mode = sparse_mode_opt.value_or(3);
  auto pre_tokens = pre_tokens_opt.value_or(kMaxInt64);
  auto next_tokens = next_tokens_opt.value_or(kMaxInt64);
  auto [softmax_max_out, softmax_sum_out] = GenResultTensors(query_index, key_index, layout);
  auto layout_value = const_cast<char *>(layout.c_str());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DenseLightningIndexerSoftmaxLse");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDenseLightningIndexerSoftmaxLse, query_index, key_index, weights,
                                          actual_seq_qlen, actual_seq_klen, layout_value, sparse_mode, pre_tokens,
                                          next_tokens, softmax_max_out, softmax_sum_out));
  runner->Run({query_index, key_index, weights}, {softmax_max_out, softmax_sum_out});
  return {softmax_max_out, softmax_sum_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_dense_lightning_indexer_softmax_lse",
        PYBOOST_CALLER(2, custom::npu_dense_lightning_indexer_softmax_lse));
}
}  // namespace custom
