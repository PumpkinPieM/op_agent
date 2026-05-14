#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> QShape(const ms::Tensor &x, const ms::Tensor &v_cache, const std::vector<int64_t> &size_splits) {
  const auto &x_shape = x.shape();
  const auto &cache_shape = v_cache.shape();
  const int64_t b = x_shape[0];
  const int64_t d = cache_shape[3];
  const int64_t q_heads = d == 0 ? 0 : size_splits[0] / d;
  if (x_shape.size() == 3) {
    return {b, x_shape[1], q_heads, d};
  }
  return {b, q_heads, d};
}

std::vector<int64_t> KVShape(const ms::Tensor &x, const ms::Tensor &v_cache) {
  const auto &x_shape = x.shape();
  const auto &cache_shape = v_cache.shape();
  const int64_t b = x_shape[0];
  const int64_t kv_heads = cache_shape[2];
  const int64_t d = cache_shape[3];
  if (x_shape.size() == 3) {
    return {b, x_shape[1], kv_heads, d};
  }
  return {b, kv_heads, d};
}
}  // namespace

std::vector<ms::Tensor> npu_dequant_rope_quant_kvcache(const ms::Tensor &x, const ms::Tensor &cos, const ms::Tensor &sin, const ms::Tensor &k_cache, const ms::Tensor &v_cache, const ms::Tensor &indices, const ms::Tensor &scale_k, const ms::Tensor &scale_v, const std::vector<int64_t> &size_splits, const std::optional<ms::Tensor> &offset_k_opt, const std::optional<ms::Tensor> &offset_v_opt, const std::optional<ms::Tensor> &weight_scale_opt, const std::optional<ms::Tensor> &activation_scale_opt, const std::optional<ms::Tensor> &bias_opt, int64_t quant_mode, const std::string &input_layout, bool kv_output, const std::string &cache_mode) {
  auto offset_k_value = offset_k_opt.value_or(ms::Tensor());
  auto offset_v_value = offset_v_opt.value_or(ms::Tensor());
  auto weight_scale_value = weight_scale_opt.value_or(ms::Tensor());
  auto activation_scale_value = activation_scale_opt.value_or(ms::Tensor());
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto quant_mode_value = quant_mode == 1 ? std::string("dynamic") : std::string("static");
  auto quant_mode_ptr = const_cast<char *>(quant_mode_value.c_str());
  auto input_layout_ptr = const_cast<char *>(input_layout.c_str());
  auto cache_mode_ptr = const_cast<char *>(cache_mode.c_str());
  auto q_out = ms::Tensor(cos.data_type(), QShape(x, v_cache, size_splits));
  auto k_out = ms::Tensor(cos.data_type(), kv_output ? KVShape(x, v_cache) : std::vector<int64_t>{0});
  auto v_out = ms::Tensor(cos.data_type(), kv_output ? KVShape(x, v_cache) : std::vector<int64_t>{0});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DequantRopeQuantKvcache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDequantRopeQuantKvcache, x, cos, sin, k_cache, v_cache, indices,
                                          scale_k, scale_v, offset_k_opt, offset_v_opt, weight_scale_opt,
                                          activation_scale_opt, bias_opt, size_splits, quant_mode_ptr,
                                          input_layout_ptr, kv_output, cache_mode_ptr, q_out, k_out, v_out));
  runner->Run({x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, offset_k_value, offset_v_value,
               weight_scale_value, activation_scale_value, bias_value},
              {q_out, k_out, v_out, k_cache, v_cache});
  return {q_out, k_out, v_out, k_cache, v_cache};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_dequant_rope_quant_kvcache", PYBOOST_CALLER(5, custom::npu_dequant_rope_quant_kvcache));
}
}  // namespace custom
