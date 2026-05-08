#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> GeluMulShape(const ms::Tensor &input) {
  auto shape = input.shape();
  shape.back() /= 2;
  return shape;
}
}  // namespace

ms::Tensor npu_gelu_mul(const ms::Tensor &input, const std::string &approximate = "none") {
  auto out = ms::Tensor(input.data_type(), GeluMulShape(input));
  auto approximate_value = const_cast<char *>(approximate.c_str());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GeluMul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGeluMul, input, approximate_value, out));
  runner->Run({input}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_gelu_mul", PYBOOST_CALLER(1, custom::npu_gelu_mul));
}
}  // namespace custom
