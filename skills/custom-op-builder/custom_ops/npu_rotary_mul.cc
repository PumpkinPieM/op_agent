#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_rotary_mul(const ms::Tensor &self, const ms::Tensor &r1, const ms::Tensor &r2,
                          const std::string &rotary_mode = "half",
                          const std::optional<ms::Tensor> &rotate_opt = std::nullopt) {
  auto rotate = rotate_opt.value_or(ms::Tensor());
  int64_t mode = rotary_mode == "interleave" ? 1 : 0;
  auto out = ms::Tensor(self.data_type(), self.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("RotaryPositionEmbedding");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnRotaryPositionEmbedding, self, r1, r2, mode, out));
  runner->Run({self, r1, r2, rotate}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_rotary_mul", PYBOOST_CALLER(1, custom::npu_rotary_mul));
}
}  // namespace custom
