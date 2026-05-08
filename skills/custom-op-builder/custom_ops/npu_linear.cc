#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_linear(const ms::Tensor &input, const ms::Tensor &weight, const std::optional<ms::Tensor> &bias_opt) {
  auto input_shape = input.shape();
  auto weight_shape = weight.shape();
  std::vector<int64_t> out_shape{input_shape[0], weight_shape[0]};
  auto out0 = ms::Tensor(input.data_type(), out_shape);
  int8_t cube_math_type = 0;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("Mm");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMm, input, weight, out0, cube_math_type));
  runner->Run({input, weight}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_linear", PYBOOST_CALLER(1, custom::npu_linear));
}
}  // namespace custom
