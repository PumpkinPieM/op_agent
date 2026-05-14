#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kSoftmaxLastDim = 8;

std::string UpperLayout(const std::string &layout) {
  auto value = layout;
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::toupper(c); });
  return value;
}

std::vector<int64_t> AttentionShape(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                                    int64_t head_num, const std::string &input_layout) {
  const auto &q = query.shape();
  const auto &k = key.shape();
  const auto &v = value.shape();
  auto layout = UpperLayout(input_layout);
  if (layout == "BSH") {
    int64_t d = q[2] / head_num;
    int64_t divisor = d == 0 ? 0 : k[2] / d;
    int64_t d2 = (d == 0 || k[2] == 0 || divisor == 0) ? d : v[2] / divisor;
    return {q[0], q[1], head_num * d2};
  }
  if (layout == "SBH") {
    int64_t d = q[2] / head_num;
    int64_t divisor = d == 0 ? 0 : k[2] / d;
    int64_t d2 = (d == 0 || k[2] == 0 || divisor == 0) ? d : v[2] / divisor;
    return {q[0], q[1], head_num * d2};
  }
  if (layout == "BNSD") {
    return {q[0], q[1], q[2], v[3]};
  }
  if (layout == "BSND") {
    return {q[0], q[1], q[2], v[3]};
  }
  return {q[0], q[1], v[2]};
}

std::vector<int64_t> SoftmaxShape(const ms::Tensor &query, int64_t head_num, const std::string &input_layout) {
  const auto &q = query.shape();
  auto layout = UpperLayout(input_layout);
  if (layout == "TND") {
    return {q[0], q[1], kSoftmaxLastDim};
  }
  if (layout == "SBH") {
    return {q[1], head_num, q[0], kSoftmaxLastDim};
  }
  if (layout == "BNSD") {
    return {q[0], head_num, q[2], kSoftmaxLastDim};
  }
  return {q[0], head_num, q[1], kSoftmaxLastDim};
}
}  // namespace

std::vector<ms::Tensor> npu_fusion_attention(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, int64_t head_num,
    const std::string &input_layout, const std::optional<ms::Tensor> &pse_opt,
    const std::optional<ms::Tensor> &padding_mask_opt, const std::optional<ms::Tensor> &atten_mask_opt, double scale,
    double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    const std::optional<std::vector<int64_t>> &prefix_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_kvlen_opt, int64_t sparse_mode, bool gen_mask_parallel,
    bool sync, const std::string &softmax_layout, const std::optional<ms::Tensor> &sink_opt,
    const std::optional<ms::Tensor> &dropout_mask_opt, int64_t seed, int64_t offset) {
  auto pse_value = pse_opt.value_or(ms::Tensor());
  auto padding_mask_value = padding_mask_opt.value_or(ms::Tensor());
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto dropout_mask_value = dropout_mask_opt.value_or(ms::Tensor());
  auto prefix = std::make_pair(prefix_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_kvlen = std::make_pair(actual_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);

  auto attention = ms::Tensor(query.data_type(), AttentionShape(query, key, value, head_num, input_layout));
  auto softmax_max = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxShape(query, head_num, input_layout));
  auto softmax_sum = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxShape(query, head_num, input_layout));
  auto softmax_out = ms::Tensor(query.data_type(), std::vector<int64_t>{0});

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlashAttentionScore");
  if (!actual_seq_qlen_opt.value_or(std::vector<int64_t>{}).empty() &&
      !actual_seq_kvlen_opt.value_or(std::vector<int64_t>{}).empty()) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlashAttentionVarLenScore, query, key, value, pse_opt,
                                            dropout_mask_opt, padding_mask_opt, atten_mask_opt, prefix, actual_seq_qlen,
                                            actual_seq_kvlen, scale, keep_prob, pre_tockens, next_tockens, head_num,
                                            input_layout, inner_precise, sparse_mode, softmax_max, softmax_sum,
                                            softmax_out, attention));
  } else {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlashAttentionScore, query, key, value, pse_opt, dropout_mask_opt,
                                            padding_mask_opt, atten_mask_opt, prefix, scale, keep_prob, pre_tockens,
                                            next_tockens, head_num, input_layout, inner_precise, sparse_mode,
                                            softmax_max, softmax_sum, softmax_out, attention));
  }
  runner->Run({query, key, value, pse_value, padding_mask_value, atten_mask_value, dropout_mask_value},
              {attention, softmax_max, softmax_sum, softmax_out});
  return {attention, softmax_max, softmax_sum, softmax_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fusion_attention", PYBOOST_CALLER(4, custom::npu_fusion_attention));
}
}  // namespace custom
