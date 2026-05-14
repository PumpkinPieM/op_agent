#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_deformable_conv2d(const ms::Tensor &input, const ms::Tensor &weight, const ms::Tensor &offset, const std::optional<ms::Tensor> &bias_opt, const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const std::vector<int64_t> &padding, const std::vector<int64_t> &dilation, int64_t groups, int64_t deformable_groups, bool modulated) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto base_shape = input.shape();
  auto base_dtype = input.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DeformableConv2d");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDeformableConv2d, input, weight, offset, bias_opt, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated, out0, out1));
  runner->Run({input, weight, offset, bias_value}, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_deformable_conv2d", PYBOOST_CALLER(2, custom::npu_deformable_conv2d));
}
}  // namespace custom
