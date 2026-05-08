#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_anti_quant(const ms::Tensor &x, const ms::Tensor &scale,
                          const std::optional<ms::Tensor> &offset_opt = std::nullopt,
                          const std::optional<int64_t> &dst_dtype_opt = std::nullopt,
                          const std::optional<int64_t> &src_dtype_opt = std::nullopt) {
  (void)src_dtype_opt;
  auto offset = offset_opt.value_or(ms::Tensor());
  auto dst_dtype = dst_dtype_opt.value_or(1);
  auto out = ms::Tensor(ms::TypeId::kNumberTypeFloat16, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AscendAntiQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAscendAntiQuant, x, scale, offset_opt, dst_dtype, false, out));
  runner->Run({x, scale, offset}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_anti_quant", PYBOOST_CALLER(1, custom::npu_anti_quant));
}
}  // namespace custom
