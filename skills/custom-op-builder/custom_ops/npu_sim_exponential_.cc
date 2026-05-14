#include <cmath>
#include <stdexcept>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
int64_t NumElements(const ms::Tensor &tensor) {
  int64_t count = 1;
  for (auto dim : tensor.shape()) {
    count *= dim;
  }
  return count;
}
}  // namespace

ms::Tensor npu_sim_exponential_(const ms::Tensor &self, double lambd = 1.0) {
  if (lambd <= 0.0) {
    throw std::invalid_argument("npu_sim_exponential_ expects lambd > 0.0");
  }

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("SimThreadExponential");
  if (std::isinf(lambd)) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnInplaceZero, self));
  } else {
    constexpr int64_t kSeed = 1;
    constexpr int64_t kOffset = 0;
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnSimThreadExponential, self, NumElements(self), lambd, kSeed, kOffset));
  }
  runner->Run({self}, {self});
  return self;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_sim_exponential_", PYBOOST_CALLER(1, custom::npu_sim_exponential_));
}
}  // namespace custom
