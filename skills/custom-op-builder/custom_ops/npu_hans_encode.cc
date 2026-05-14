#include <stdexcept>
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
  throw std::runtime_error("npu_hans_encode: input dtype must be float16, bfloat16, or float32");
}

int64_t CeilDiv(const int64_t value, const int64_t divisor) {
  return (value + divisor - 1) / divisor;
}
}  // namespace

std::vector<ms::Tensor> npu_hans_encode(const ms::Tensor &x, const ms::Tensor &pdf, bool statistic = false,
                                        bool reshuff = false) {
  const auto dtype = x.data_type();
  const auto element_size = ElementSize(dtype);
  const auto input_numel = NumElements(x.shape());
  const auto mantissa_numel = input_numel * (element_size - 1) / element_size;
  constexpr int64_t kMaxVectorCoreDim = 64;
  const auto compressed_bound_bytes = input_numel + input_numel / 64 + 8448 * kMaxVectorCoreDim;
  const auto compressed_bound_numel = CeilDiv(compressed_bound_bytes, element_size);

  auto mantissa = ms::Tensor(dtype, std::vector<int64_t>{mantissa_numel});
  auto fixed = ms::Tensor(dtype, std::vector<int64_t>{compressed_bound_numel});
  auto var = ms::Tensor(dtype, std::vector<int64_t>{reshuff ? input_numel : compressed_bound_numel});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("HansEncode");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnHansEncode, x, pdf, statistic, reshuff, mantissa, fixed, var));
  runner->Run({x, pdf}, {pdf, mantissa, fixed, var});
  return {pdf, mantissa, fixed, var};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_hans_encode", PYBOOST_CALLER(4, custom::npu_hans_encode));
}
}  // namespace custom
