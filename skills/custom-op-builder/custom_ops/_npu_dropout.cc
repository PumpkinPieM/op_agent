#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> _npu_dropout(const ms::Tensor &self, double p) {
  auto out = ms::Tensor(self.data_type(), self.shape());
  int64_t numel = 1;
  for (const auto dim : self.shape()) {
    numel *= dim;
  }
  int64_t mask_len = ((numel + 127) / 128) * 16;
  std::optional<ms::Tensor> optional_noise_shape = std::nullopt;
  auto seed = static_cast<int64_t>(0);
  auto offset = static_cast<int64_t>(0);
  auto mask = ms::Tensor(ms::TypeId::kNumberTypeUInt8, std::vector<int64_t>{mask_len});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DropoutV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDropoutV3, self, optional_noise_shape, p, seed, offset, out, mask));
  runner->Run({self}, {out, mask});
  return {out, mask};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("_npu_dropout", PYBOOST_CALLER(2, custom::_npu_dropout));
}
}  // namespace custom
