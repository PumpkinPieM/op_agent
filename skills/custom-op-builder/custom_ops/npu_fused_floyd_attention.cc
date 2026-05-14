#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> SoftmaxStatShape(const ms::Tensor &query) {
  auto shape = query.shape();
  shape.back() = 8;
  return shape;
}

}  // namespace

std::vector<ms::Tensor> npu_fused_floyd_attention(const ms::Tensor &query_ik, const ms::Tensor &key_ij, const ms::Tensor &value_ij, const ms::Tensor &key_jk, const ms::Tensor &value_jk, const ms::Tensor &atten_mask, double scale_value) {
  auto softmax_shape = SoftmaxStatShape(query_ik);
  auto out0 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, softmax_shape);
  auto out1 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, softmax_shape);
  auto out2 = ms::Tensor(query_ik.data_type(), query_ik.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedFloydAttention");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedFloydAttention, query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask, scale_value, out0, out1, out2));
  runner->Run({query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_fused_floyd_attention", PYBOOST_CALLER(3, custom::npu_fused_floyd_attention));
}
}  // namespace custom
