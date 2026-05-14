#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> GenOutputShape(const ms::Tensor &scale, const std::optional<ms::Tensor> &offset_opt) {
  auto out_shape = scale.shape();
  if (offset_opt.has_value() && out_shape.size() == 1) {
    const auto &offset_shape = offset_opt->shape();
    if (!offset_shape.empty() && offset_shape[0] > out_shape[0]) {
      out_shape = offset_shape;
    }
  }
  return out_shape;
}
}  // namespace

ms::Tensor npu_trans_quant_param(const ms::Tensor &scale, const std::optional<ms::Tensor> &offset_opt = std::nullopt,
                              const std::optional<int64_t> &round_mode_opt = std::nullopt) {
  auto offset = offset_opt.value_or(ms::Tensor());
  auto round_mode = round_mode_opt.value_or(0);
  if (round_mode != 0 && round_mode != 1) {
    throw std::invalid_argument("round_mode must be 0 or 1");
  }
  auto out = ms::Tensor(ms::TypeId::kNumberTypeInt64, GenOutputShape(scale, offset_opt));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("TransQuantParamV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnTransQuantParamV3, scale, offset_opt, round_mode, out));
  runner->Run({scale, offset}, {out});
  return out;
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_trans_quant_param", PYBOOST_CALLER(1, custom::npu_trans_quant_param)); }

}  // namespace custom
