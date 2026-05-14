#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
using TensorList = std::vector<ms::Tensor>;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    MS_LOG(EXCEPTION) << message;
  }
}

std::pair<std::string, std::string> ParseLayout(const std::string &layout) {
  if (layout == "BSH" || layout == "BSND" || layout == "BNSD" || layout == "TND" || layout == "NTD" ||
      layout == "NSD") {
    return {layout, layout};
  }
  if (layout == "BNSD_BSND") return {"BNSD", "BSND"};
  if (layout == "BSH_BNSD") return {"BSH", "BNSD"};
  if (layout == "BSND_BNSD") return {"BSND", "BNSD"};
  if (layout == "NTD_TND") return {"NTD", "TND"};
  if (layout == "BSH_NBSD") return {"BSH", "NBSD"};
  if (layout == "BSND_NBSD") return {"BSND", "NBSD"};
  if (layout == "BNSD_NBSD") return {"BNSD", "NBSD"};
  if (layout == "TND_NTD") return {"TND", "NTD"};
  MS_LOG(EXCEPTION) << "Unsupported input_layout: " << layout;
}

std::tuple<int64_t, int64_t, int64_t, int64_t> GetBNSD(const ms::Tensor &query, const std::string &query_layout,
                                                       int64_t num_heads) {
  const auto &shape = query.shape();
  if (query_layout == "BSH") return {shape[0], num_heads, shape[1], shape[2] / num_heads};
  if (query_layout == "BSND") return {shape[0], shape[2], shape[1], shape[3]};
  if (query_layout == "BNSD") return {shape[0], shape[1], shape[2], shape[3]};
  if (query_layout == "NSD") return {1, shape[0], shape[1], shape[2]};
  MS_LOG(EXCEPTION) << "Unsupported BNSD query layout: " << query_layout;
}

std::tuple<int64_t, int64_t, int64_t> GetTND(const ms::Tensor &query, const std::string &query_layout) {
  const auto &shape = query.shape();
  if (query_layout == "TND") return {shape[0], shape[1], shape[2]};
  if (query_layout == "NTD") return {shape[1], shape[0], shape[2]};
  MS_LOG(EXCEPTION) << "Unsupported TND query layout: " << query_layout;
}

int64_t ValueD(const ms::Tensor &query, const ms::Tensor &value, const std::string &query_layout,
               const std::optional<ms::Tensor> &block_table_opt, int64_t kv_num_heads) {
  const auto &value_shape = value.shape();
  if (block_table_opt.has_value()) {
    if (value_shape.size() == 3) return value_shape[2] / kv_num_heads;
    if (value_shape.size() == 4) return value_shape[3];
    if (value_shape.size() == 5) return value_shape[2] * value_shape[4];
    MS_LOG(EXCEPTION) << "Page attention value rank must be 3, 4, or 5.";
  }
  Check(value_shape.size() == query.shape().size(), "value rank must match query rank when block_table is None.");
  if (query_layout == "BSH") return value_shape[2] / kv_num_heads;
  if (query_layout == "BSND" || query_layout == "BNSD") return value_shape[3];
  if (query_layout == "TND" || query_layout == "NTD" || query_layout == "NSD") return value_shape[2];
  MS_LOG(EXCEPTION) << "Unsupported query layout for value D: " << query_layout;
}

std::vector<int64_t> AttentionOutShape(const ms::Tensor &query, const std::string &query_layout,
                                       const std::string &out_layout, int64_t num_heads, int64_t value_d) {
  if (out_layout == "BSH") {
    auto [b, n, s, d] = GetBNSD(query, query_layout, num_heads);
    return {b, s, num_heads * value_d};
  }
  if (out_layout == "BSND") {
    auto [b, n, s, d] = GetBNSD(query, query_layout, num_heads);
    return {b, s, n, value_d == 0 ? d : value_d};
  }
  if (out_layout == "BNSD") {
    auto [b, n, s, d] = GetBNSD(query, query_layout, num_heads);
    return {b, n, s, value_d == 0 ? d : value_d};
  }
  if (out_layout == "NBSD") {
    auto [b, n, s, d] = GetBNSD(query, query_layout, num_heads);
    return {n, b, s, value_d == 0 ? d : value_d};
  }
  if (out_layout == "TND") {
    auto [t, n, d] = GetTND(query, query_layout);
    return {t, n, value_d == 0 ? d : value_d};
  }
  if (out_layout == "NTD") {
    auto [t, n, d] = GetTND(query, query_layout);
    return {n, t, value_d == 0 ? d : value_d};
  }
  if (out_layout == "NSD") {
    auto [b, n, s, d] = GetBNSD(query, query_layout, num_heads);
    return {n, s, value_d == 0 ? d : value_d};
  }
  MS_LOG(EXCEPTION) << "Unsupported output layout: " << out_layout;
}

std::vector<int64_t> LseShape(const ms::Tensor &query, const std::string &input_layout,
                              const std::string &query_layout, int64_t num_heads, bool enabled) {
  if (!enabled) return {0};
  if (input_layout == "TND" || input_layout == "NTD" || input_layout == "TND_NTD" || input_layout == "NTD_TND") {
    auto [t, n, d] = GetTND(query, query_layout);
    return {t, n, 1};
  }
  auto [b, n, s, d] = GetBNSD(query, query_layout, num_heads);
  return {b, n, s, 1};
}
}  // namespace

std::vector<ms::Tensor> npu_fused_infer_attention_score(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
    const std::optional<ms::Tensor> &pse_shift_opt, const std::optional<ms::Tensor> &atten_mask_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_lengths_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_lengths_kv_opt,
    const std::optional<ms::Tensor> &dequant_scale1_opt, const std::optional<ms::Tensor> &quant_scale1_opt,
    const std::optional<ms::Tensor> &dequant_scale2_opt, const std::optional<ms::Tensor> &quant_scale2_opt,
    const std::optional<ms::Tensor> &quant_offset2_opt, const std::optional<ms::Tensor> &antiquant_scale_opt,
    const std::optional<ms::Tensor> &antiquant_offset_opt, const std::optional<ms::Tensor> &key_antiquant_scale_opt,
    const std::optional<ms::Tensor> &key_antiquant_offset_opt,
    const std::optional<ms::Tensor> &value_antiquant_scale_opt,
    const std::optional<ms::Tensor> &value_antiquant_offset_opt, const std::optional<ms::Tensor> &block_table_opt,
    const std::optional<ms::Tensor> &query_padding_size_opt, const std::optional<ms::Tensor> &kv_padding_size_opt,
    const std::optional<ms::Tensor> &key_shared_prefix_opt, const std::optional<ms::Tensor> &value_shared_prefix_opt,
    const std::optional<std::vector<int64_t>> &actual_shared_prefix_len_opt,
    const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt,
    const std::optional<ms::Tensor> &key_rope_antiquant_scale_opt, int64_t num_heads, double scale,
    int64_t pre_tokens, int64_t next_tokens, const std::string &input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise, int64_t block_size, int64_t antiquant_mode,
    int64_t key_antiquant_mode, int64_t value_antiquant_mode, bool softmax_lse_flag) {
  Check(num_heads > 0, "num_heads must be greater than 0.");
  int64_t kv_heads = num_key_value_heads == 0 ? num_heads : num_key_value_heads;
  auto [query_layout, out_layout] = ParseLayout(input_layout);
  auto value_d = ValueD(query, value, query_layout, block_table_opt, kv_heads);
  if (value.data_type() == ms::TypeId::kNumberTypeInt32) value_d *= 8;

  auto attention_out = ms::Tensor(query.data_type(), AttentionOutShape(query, query_layout, out_layout, num_heads, value_d));
  auto softmax_lse = ms::Tensor(ms::TypeId::kNumberTypeFloat32,
                                LseShape(query, input_layout, query_layout, num_heads, softmax_lse_flag));
  TensorList key_tensors{key};
  TensorList value_tensors{value};
  auto actual_seq_lengths = std::make_pair(actual_seq_lengths_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_lengths_kv = std::make_pair(actual_seq_lengths_kv_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_shared_prefix_len =
      std::make_pair(actual_shared_prefix_len_opt.value_or(std::vector<int64_t>{}), true);

  auto pse_shift = pse_shift_opt.value_or(ms::Tensor());
  auto atten_mask = atten_mask_opt.value_or(ms::Tensor());
  auto dequant_scale1 = dequant_scale1_opt.value_or(ms::Tensor());
  auto quant_scale1 = quant_scale1_opt.value_or(ms::Tensor());
  auto dequant_scale2 = dequant_scale2_opt.value_or(ms::Tensor());
  auto quant_scale2 = quant_scale2_opt.value_or(ms::Tensor());
  auto quant_offset2 = quant_offset2_opt.value_or(ms::Tensor());
  auto antiquant_scale = antiquant_scale_opt.value_or(ms::Tensor());
  auto antiquant_offset = antiquant_offset_opt.value_or(ms::Tensor());
  auto key_antiquant_scale = key_antiquant_scale_opt.value_or(ms::Tensor());
  auto key_antiquant_offset = key_antiquant_offset_opt.value_or(ms::Tensor());
  auto value_antiquant_scale = value_antiquant_scale_opt.value_or(ms::Tensor());
  auto value_antiquant_offset = value_antiquant_offset_opt.value_or(ms::Tensor());
  auto block_table = block_table_opt.value_or(ms::Tensor());
  auto query_padding_size = query_padding_size_opt.value_or(ms::Tensor());
  auto kv_padding_size = kv_padding_size_opt.value_or(ms::Tensor());
  auto key_shared_prefix = key_shared_prefix_opt.value_or(ms::Tensor());
  auto value_shared_prefix = value_shared_prefix_opt.value_or(ms::Tensor());
  auto query_rope = query_rope_opt.value_or(ms::Tensor());
  auto key_rope = key_rope_opt.value_or(ms::Tensor());
  auto key_rope_antiquant_scale = key_rope_antiquant_scale_opt.value_or(ms::Tensor());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedInferAttentionScoreV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnFusedInferAttentionScoreV3, query, key_tensors, value_tensors, pse_shift_opt, atten_mask_opt,
      actual_seq_lengths, actual_seq_lengths_kv, dequant_scale1_opt, quant_scale1_opt, dequant_scale2_opt,
      quant_scale2_opt, quant_offset2_opt, antiquant_scale_opt, antiquant_offset_opt, block_table_opt,
      query_padding_size_opt, kv_padding_size_opt, key_antiquant_scale_opt, key_antiquant_offset_opt,
      value_antiquant_scale_opt, value_antiquant_offset_opt, key_shared_prefix_opt, value_shared_prefix_opt,
      actual_shared_prefix_len, query_rope_opt, key_rope_opt, key_rope_antiquant_scale_opt, num_heads, scale,
      pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size,
      antiquant_mode, softmax_lse_flag, key_antiquant_mode, value_antiquant_mode, attention_out, softmax_lse));
  runner->Run({query, key, value, pse_shift, atten_mask, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2,
               quant_offset2, antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset,
               value_antiquant_scale, value_antiquant_offset, block_table, query_padding_size, kv_padding_size,
               key_shared_prefix, value_shared_prefix, query_rope, key_rope, key_rope_antiquant_scale},
              {attention_out, softmax_lse});
  return {attention_out, softmax_lse};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fused_infer_attention_score", PYBOOST_CALLER(2, custom::npu_fused_infer_attention_score));
}
}  // namespace custom
