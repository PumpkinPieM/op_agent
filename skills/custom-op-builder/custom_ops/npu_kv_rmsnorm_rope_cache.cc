#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_kv_rmsnorm_rope_cache(const ms::Tensor &kv, const ms::Tensor &gamma, const ms::Tensor &cos, const ms::Tensor &sin, const ms::Tensor &index, const ms::Tensor &k_cache, const ms::Tensor &ckv_cache, const std::optional<ms::Tensor> &k_rope_scale_opt, const std::optional<ms::Tensor> &c_kv_scale_opt, const std::optional<ms::Tensor> &k_rope_offset_opt, const std::optional<ms::Tensor> &c_kv_offset_opt, const std::optional<ms::Tensor> &v_opt, double epsilon, const std::string &cache_mode, bool is_output_kv) {
  auto k_rope_scale_value = k_rope_scale_opt.value_or(ms::Tensor());
  auto c_kv_scale_value = c_kv_scale_opt.value_or(ms::Tensor());
  auto k_rope_offset_value = k_rope_offset_opt.value_or(ms::Tensor());
  auto c_kv_offset_value = c_kv_offset_opt.value_or(ms::Tensor());
  auto v_value = v_opt.value_or(ms::Tensor());
  auto base_shape = kv.shape();
  auto base_dtype = kv.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto out3 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("KvRmsNormRopeCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale_opt, c_kv_scale_opt, k_rope_offset_opt, c_kv_offset_opt, v_opt, epsilon, cache_mode, is_output_kv, out0, out1, out2, out3));
  runner->Run({kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale_value, c_kv_scale_value, k_rope_offset_value, c_kv_offset_value, v_value}, {out0, out1, out2, out3});
  return {out0, out1, out2, out3};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_kv_rmsnorm_rope_cache", PYBOOST_CALLER(4, custom::npu_kv_rmsnorm_rope_cache));
}
}  // namespace custom
