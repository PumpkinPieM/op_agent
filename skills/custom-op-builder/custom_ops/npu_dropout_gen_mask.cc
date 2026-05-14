#include <vector>
#include <optional>
#include <string>
#include "ms_extension/all.h"

namespace custom {
namespace {
int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

ms::Tensor npu_dropout_gen_mask(const std::vector<int64_t> &size, double p,
                                   const std::optional<int64_t> &dtype_opt = std::nullopt,
                                   const std::optional<int64_t> &layout_opt = std::nullopt,
                                   const std::optional<std::string> &device_opt = std::nullopt,
                                   const std::optional<bool> &pin_memory_opt = std::nullopt) {
  (void)dtype_opt; (void)layout_opt; (void)device_opt; (void)pin_memory_opt;
  int64_t numels = 1;
  for (auto dim : size) {
    numels *= dim;
  }
  int64_t length = CeilDiv(numels, 128) * 16;
  auto shape_array = std::make_pair(std::vector<int64_t>{numels}, true);
  auto mask = ms::Tensor(ms::TypeId::kNumberTypeUInt8, std::vector<int64_t>{length});
  constexpr int64_t seed = 1;
  constexpr int64_t offset = 0;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DropoutGenMask");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDropoutGenMask, shape_array, p, seed, offset, mask));
  runner->Run({}, {mask});
  return mask;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_dropout_gen_mask", PYBOOST_CALLER(1, custom::npu_dropout_gen_mask)); }
