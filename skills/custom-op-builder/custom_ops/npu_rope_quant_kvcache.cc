#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> QueryShape(const ms::Tensor &x, const ms::Tensor &v_cache, const std::vector<int64_t> &size_splits) {
  const auto &x_shape = x.shape();
  const auto &cache_shape = v_cache.shape();
  if (x_shape.size() != 2 && x_shape.size() != 3) {
    throw std::invalid_argument("3D or 2D tensor expected for input x");
  }
  if (cache_shape.size() != 4) {
    throw std::invalid_argument("4D tensor expected for input cache");
  }
  if (size_splits.size() != 3 || size_splits[0] < 0) {
    throw std::invalid_argument("size_splits must have three non-negative q/k/v split sizes");
  }

  const int64_t batch = x_shape[0];
  const int64_t seq = x_shape.size() == 3 ? x_shape[1] : 0;
  const int64_t head_dim = cache_shape[3];
  const int64_t q_heads = head_dim == 0 ? 0 : size_splits[0] / head_dim;
  if (x_shape.size() == 3) {
    return {batch, seq, q_heads, head_dim};
  }
  return {batch, q_heads, head_dim};
}

std::vector<int64_t> KvShape(const ms::Tensor &x, const ms::Tensor &v_cache) {
  const auto &x_shape = x.shape();
  const auto &cache_shape = v_cache.shape();
  const int64_t batch = x_shape[0];
  const int64_t seq = x_shape.size() == 3 ? x_shape[1] : 0;
  const int64_t kv_heads = cache_shape[2];
  const int64_t head_dim = cache_shape[3];
  if (x_shape.size() == 3) {
    return {batch, seq, kv_heads, head_dim};
  }
  return {batch, kv_heads, head_dim};
}
}  // namespace

std::vector<ms::Tensor> npu_rope_quant_kvcache(
    const ms::Tensor &x, const ms::Tensor &cos, const ms::Tensor &sin, const ms::Tensor &k_cache,
    const ms::Tensor &v_cache, const ms::Tensor &indices, const ms::Tensor &scale_k, const ms::Tensor &scale_v,
    const std::vector<int64_t> &size_splits, const std::optional<ms::Tensor> &offset_k_opt = std::nullopt,
    const std::optional<ms::Tensor> &offset_v_opt = std::nullopt, int64_t quant_mode = 0,
    const std::optional<std::string> &input_layout_opt = std::nullopt, bool kv_output = false,
    const std::optional<std::string> &cache_mode_opt = std::nullopt) {
  auto q = ms::Tensor(cos.data_type(), QueryShape(x, v_cache, size_splits));
  auto k = ms::Tensor(cos.data_type(), kv_output ? KvShape(x, v_cache) : std::vector<int64_t>{0});
  auto v = ms::Tensor(cos.data_type(), kv_output ? KvShape(x, v_cache) : std::vector<int64_t>{0});
  auto offset_k = offset_k_opt.value_or(ms::Tensor());
  auto offset_v = offset_v_opt.value_or(ms::Tensor());
  std::optional<ms::Tensor> weight_scale_opt = std::nullopt;
  std::optional<ms::Tensor> activation_scale_opt = std::nullopt;
  std::optional<ms::Tensor> bias_opt = std::nullopt;
  auto quant_mode_str = quant_mode == 1 ? std::string("dynamic") : std::string("static");
  auto input_layout = input_layout_opt.value_or("BSND");
  auto cache_mode = cache_mode_opt.value_or("contiguous");

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DequantRopeQuantKvcache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDequantRopeQuantKvcache, x, cos, sin, k_cache, v_cache, indices,
                                          scale_k, scale_v, offset_k_opt, offset_v_opt, weight_scale_opt,
                                          activation_scale_opt, bias_opt, size_splits, quant_mode_str, input_layout,
                                          kv_output, cache_mode, q, k, v));
  runner->Run({x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, offset_k, offset_v},
              {q, k, v, k_cache, v_cache});
  return {q, k, v, k_cache, v_cache};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_rope_quant_kvcache", PYBOOST_CALLER(5, custom::npu_rope_quant_kvcache));
}
}  // namespace custom
