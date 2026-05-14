#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_fused_floyd_attention(const ms::Tensor &query_ik, const ms::Tensor &key_ij, const ms::Tensor &value_ij, const ms::Tensor &key_jk, const ms::Tensor &value_jk, const std::optional<ms::Tensor> &atten_mask_opt, double scale_value) {
  auto atten_mask_value = atten_mask_opt.value_or(ms::Tensor());
  auto base_shape = query_ik.shape();
  auto base_dtype = query_ik.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedFloydAttention");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedFloydAttention, query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask_opt, scale_value, out0, out1, out2));
  runner->Run({query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fused_floyd_attention", PYBOOST_CALLER(3, custom::npu_fused_floyd_attention));
}
}  // namespace custom
