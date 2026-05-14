#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> DequantShape(const ms::Tensor &x, const ms::Tensor &weight) {
  auto x_shape = x.shape();
  auto weight_shape = weight.shape();
  if (x_shape.size() != 2 || weight_shape.size() < 2) {
    return x_shape;
  }
  return {x_shape[0], weight_shape[0]};
}
}  // namespace

ms::Tensor npu_quant_matmul_dequant(const ms::Tensor &x, const ms::Tensor &quantized_weight,
                                    const ms::Tensor &weight_scale,
                                    const std::optional<ms::Tensor> &bias_opt,
                                    const std::optional<ms::Tensor> &x_scale_opt,
                                    const std::optional<ms::Tensor> &x_offset_opt,
                                    const std::optional<ms::Tensor> &smooth_scale_opt,
                                    const std::optional<std::string> &quant_mode_opt) {
  auto out = ms::Tensor(ms::TypeId::kNumberTypeFloat16, DequantShape(x, quantized_weight));
  auto bias = bias_opt.value_or(ms::Tensor());
  auto x_scale = x_scale_opt.value_or(ms::Tensor());
  auto x_offset = x_offset_opt.value_or(ms::Tensor());
  auto smooth_scale = smooth_scale_opt.value_or(ms::Tensor());
  auto quant_mode = quant_mode_opt.value_or("pertoken");
  bool transpose_weight = true;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulDequant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulDequant, x, quantized_weight, weight_scale, bias_opt,
                                          x_scale_opt, x_offset_opt, smooth_scale_opt, quant_mode, transpose_weight,
                                          out));
  runner->Run({x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_matmul_dequant", PYBOOST_CALLER(1, custom::npu_quant_matmul_dequant));
}
}  // namespace custom
