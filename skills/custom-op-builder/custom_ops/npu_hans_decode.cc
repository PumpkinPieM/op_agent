#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
int64_t NumElements(const std::vector<int64_t> &shape) {
  int64_t numel = 1;
  for (auto dim : shape) {
    numel *= dim;
  }
  return numel;
}

int64_t ElementSize(const ms::TypeId dtype) {
  if (dtype == ms::TypeId::kNumberTypeFloat32) {
    return 4;
  }
  if (dtype == ms::TypeId::kNumberTypeFloat16 || dtype == ms::TypeId::kNumberTypeBFloat16) {
    return 2;
  }
  throw std::runtime_error("npu_hans_decode: mantissa dtype must be float16, bfloat16, or float32");
}

std::vector<int64_t> OutputShape(const ms::Tensor &mantissa) {
  const auto element_size = ElementSize(mantissa.data_type());
  const auto mantissa_numel = NumElements(mantissa.shape());
  return {mantissa_numel * element_size / (element_size - 1)};
}
}  // namespace

ms::Tensor npu_hans_decode(const ms::Tensor &mantissa, const ms::Tensor &fixed, const ms::Tensor &var,
                           const ms::Tensor &pdf, bool reshuff = false) {
  auto out = ms::Tensor(mantissa.data_type(), OutputShape(mantissa));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("HansDecode");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnHansDecode, mantissa, fixed, var, pdf, reshuff, out));
  runner->Run({mantissa, fixed, var, pdf}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_hans_decode", PYBOOST_CALLER(1, custom::npu_hans_decode));
}
}  // namespace custom
