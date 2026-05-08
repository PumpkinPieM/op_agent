#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_dequant_bias(const ms::Tensor &x, const ms::Tensor &weight_scale,
                            const std::optional<ms::Tensor> &activation_scale_opt = std::nullopt,
                            const std::optional<ms::Tensor> &bias_opt = std::nullopt,
                            const std::optional<int64_t> &output_dtype_opt = std::nullopt) {
  auto activation_scale = activation_scale_opt.value_or(ms::Tensor());
  auto bias = bias_opt.value_or(ms::Tensor());
  auto output_dtype = output_dtype_opt.value_or(1);
  auto out = ms::Tensor(ms::TypeId::kNumberTypeFloat16, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DequantBias");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDequantBias, x, weight_scale, activation_scale_opt, bias_opt,
                                          output_dtype, out));
  runner->Run({x, weight_scale, activation_scale, bias}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_dequant_bias", PYBOOST_CALLER(1, custom::npu_dequant_bias));
}
}  // namespace custom
