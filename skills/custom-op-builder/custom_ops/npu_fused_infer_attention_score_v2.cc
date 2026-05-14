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

std::vector<ms::Tensor> npu_fused_infer_attention_score_v2(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
    const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt,
    const std::optional<ms::Tensor> &pse_shift_opt, const std::optional<ms::Tensor> &atten_mask_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt,
    const std::optional<std::vector<int64_t>> &actual_seq_kvlen_opt, const std::optional<ms::Tensor> &block_table_opt,
    const std::optional<ms::Tensor> &dequant_scale_query_opt, const std::optional<ms::Tensor> &dequant_scale_key_opt,
    const std::optional<ms::Tensor> &dequant_offset_key_opt,
    const std::optional<ms::Tensor> &dequant_scale_value_opt,
    const std::optional<ms::Tensor> &dequant_offset_value_opt,
    const std::optional<ms::Tensor> &dequant_scale_key_rope_opt,
    const std::optional<ms::Tensor> &quant_scale_out_opt, const std::optional<ms::Tensor> &quant_offset_out_opt,
    const std::optional<ms::Tensor> &quant_scale_p_opt, const std::optional<ms::Tensor> &learnable_sink_opt,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens,
    int64_t next_tokens, const std::string &input_layout, int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode, int64_t inner_precise,
    bool return_softmax_lse, const std::optional<int64_t> &query_dtype_opt,
    const std::optional<int64_t> &key_dtype_opt, const std::optional<int64_t> &value_dtype_opt,
    const std::optional<int64_t> &query_rope_dtype_opt, const std::optional<int64_t> &key_rope_dtype_opt,
    const std::optional<int64_t> &key_shared_prefix_dtype_opt,
    const std::optional<int64_t> &value_shared_prefix_dtype_opt,
    const std::optional<int64_t> &dequant_scale_query_dtype_opt,
    const std::optional<int64_t> &dequant_scale_key_dtype_opt,
    const std::optional<int64_t> &dequant_scale_value_dtype_opt,
    const std::optional<int64_t> &dequant_scale_key_rope_dtype_opt, const std::optional<int64_t> &out_dtype_opt) {
  Check(num_query_heads > 0, "num_query_heads must be greater than 0.");
  int64_t kv_heads = num_key_value_heads == 0 ? num_query_heads : num_key_value_heads;
  auto [query_layout, out_layout] = ParseLayout(input_layout);
  auto value_d = ValueD(query, value, query_layout, block_table_opt, kv_heads);
  if (value.data_type() == ms::TypeId::kNumberTypeInt32) value_d *= 8;

  auto out_type = quant_scale_out_opt.has_value() ? ms::TypeId::kNumberTypeInt8 : query.data_type();
  auto attention_out = ms::Tensor(out_type, AttentionOutShape(query, query_layout, out_layout, num_query_heads, value_d));
  auto softmax_lse = ms::Tensor(ms::TypeId::kNumberTypeFloat32,
                                LseShape(query, input_layout, query_layout, num_query_heads, return_softmax_lse));
  TensorList key_tensors{key};
  TensorList value_tensors{value};
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
  auto actual_seq_kvlen = std::make_pair(actual_seq_kvlen_opt.value_or(std::vector<int64_t>{}), true);

  auto query_rope = query_rope_opt.value_or(ms::Tensor());
  auto key_rope = key_rope_opt.value_or(ms::Tensor());
  auto pse_shift = pse_shift_opt.value_or(ms::Tensor());
  auto atten_mask = atten_mask_opt.value_or(ms::Tensor());
  auto block_table = block_table_opt.value_or(ms::Tensor());
  auto dequant_scale_query = dequant_scale_query_opt.value_or(ms::Tensor());
  auto dequant_scale_key = dequant_scale_key_opt.value_or(ms::Tensor());
  auto dequant_offset_key = dequant_offset_key_opt.value_or(ms::Tensor());
  auto dequant_scale_value = dequant_scale_value_opt.value_or(ms::Tensor());
  auto dequant_offset_value = dequant_offset_value_opt.value_or(ms::Tensor());
  auto dequant_scale_key_rope = dequant_scale_key_rope_opt.value_or(ms::Tensor());
  auto quant_scale_out = quant_scale_out_opt.value_or(ms::Tensor());
  auto quant_offset_out = quant_offset_out_opt.value_or(ms::Tensor());
  auto quant_scale_p = quant_scale_p_opt.value_or(ms::Tensor());
  auto learnable_sink = learnable_sink_opt.value_or(ms::Tensor());

  (void)quant_scale_p_opt;
  (void)query_dtype_opt;
  (void)key_dtype_opt;
  (void)value_dtype_opt;
  (void)query_rope_dtype_opt;
  (void)key_rope_dtype_opt;
  (void)key_shared_prefix_dtype_opt;
  (void)value_shared_prefix_dtype_opt;
  (void)dequant_scale_query_dtype_opt;
  (void)dequant_scale_key_dtype_opt;
  (void)dequant_scale_value_dtype_opt;
  (void)dequant_scale_key_rope_dtype_opt;
  (void)out_dtype_opt;

  const int64_t antiquant_mode = 0;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedInferAttentionScoreV4");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnFusedInferAttentionScoreV4, query, key_tensors, value_tensors, pse_shift_opt, atten_mask_opt,
      actual_seq_qlen, actual_seq_kvlen, nullptr, nullptr, nullptr, quant_scale_out_opt, quant_offset_out_opt, nullptr,
      nullptr, block_table_opt, nullptr, nullptr, dequant_scale_key_opt, dequant_offset_key_opt,
      dequant_scale_value_opt, dequant_offset_value_opt, nullptr, nullptr, nullptr, query_rope_opt, key_rope_opt,
      dequant_scale_key_rope_opt, dequant_scale_query_opt, learnable_sink_opt, num_query_heads, softmax_scale,
      pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode,
      return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, attention_out, softmax_lse));
  runner->Run({query, key, value, query_rope, key_rope, pse_shift, atten_mask, block_table, dequant_scale_query,
               dequant_scale_key, dequant_offset_key, dequant_scale_value, dequant_offset_value,
               dequant_scale_key_rope, quant_scale_out, quant_offset_out, quant_scale_p, learnable_sink},
              {attention_out, softmax_lse});
  return {attention_out, softmax_lse};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fused_infer_attention_score_v2", PYBOOST_CALLER(2, custom::npu_fused_infer_attention_score_v2));
}
}  // namespace custom
