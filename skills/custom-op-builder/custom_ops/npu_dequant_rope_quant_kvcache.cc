#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_dequant_rope_quant_kvcache(const ms::Tensor &x, const ms::Tensor &cos, const ms::Tensor &sin, const ms::Tensor &k_cache, const ms::Tensor &v_cache, const ms::Tensor &indices, const ms::Tensor &scale_k, const ms::Tensor &scale_v, const std::vector<int64_t> &size_splits, const std::optional<ms::Tensor> &offset_k_opt, const std::optional<ms::Tensor> &offset_v_opt, const std::optional<ms::Tensor> &weight_scale_opt, const std::optional<ms::Tensor> &activation_scale_opt, const std::optional<ms::Tensor> &bias_opt, int64_t quant_mode, const std::string &input_layout, bool kv_output, const std::string &cache_mode) {
  auto offset_k_value = offset_k_opt.value_or(ms::Tensor());
  auto offset_v_value = offset_v_opt.value_or(ms::Tensor());
  auto weight_scale_value = weight_scale_opt.value_or(ms::Tensor());
  auto activation_scale_value = activation_scale_opt.value_or(ms::Tensor());
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto base_shape = x.shape();
  auto base_dtype = x.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto out3 = ms::Tensor(base_dtype, base_shape);
  auto out4 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DequantRopeQuantKvcache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDequantRopeQuantKvcache, x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k_opt, offset_v_opt, weight_scale_opt, activation_scale_opt, bias_opt, quant_mode, input_layout, kv_output, cache_mode, out0, out1, out2, out3, out4));
  runner->Run({x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, offset_k_value, offset_v_value, weight_scale_value, activation_scale_value, bias_value}, {out0, out1, out2, out3, out4});
  return {out0, out1, out2, out3, out4};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_dequant_rope_quant_kvcache", PYBOOST_CALLER(5, custom::npu_dequant_rope_quant_kvcache));
}
}  // namespace custom
