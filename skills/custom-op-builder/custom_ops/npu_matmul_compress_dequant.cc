#include <optional>
#include <stdexcept>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> OutputShape(const ms::Tensor &x1, const ms::Tensor &bias) {
  const auto &x1_shape = x1.shape();
  const auto &bias_shape = bias.shape();
  if (x1_shape.size() != 2) {
    throw std::runtime_error("npu_matmul_compress_dequant: x1 must be 2D");
  }
  if (bias_shape.size() != 2) {
    throw std::runtime_error("npu_matmul_compress_dequant: bias must be 2D");
  }
  return {x1_shape[0], bias_shape[1]};
}
}  // namespace

ms::Tensor npu_matmul_compress_dequant(const ms::Tensor &x1, const ms::Tensor &x2,
                                       const ms::Tensor &compress_index, const ms::Tensor &bias,
                                       const ms::Tensor &scale, const std::optional<ms::Tensor> &offsetW_opt,
                                       const std::optional<int64_t> &offsetX_opt) {
  if (offsetW_opt.has_value()) {
    throw std::runtime_error("npu_matmul_compress_dequant: offsetW currently only supports None");
  }
  if (offsetX_opt.value_or(0) != 0) {
    throw std::runtime_error("npu_matmul_compress_dequant: offsetX currently only supports 0");
  }
  if (scale.shape().size() != 2) {
    throw std::runtime_error("npu_matmul_compress_dequant: scale must be 2D");
  }
  auto out = ms::Tensor(ms::TypeId::kNumberTypeFloat16, OutputShape(x1, bias));
  auto offsetW = offsetW_opt.value_or(ms::Tensor());
  int offsetX = 0;
  std::vector<int64_t> compressInfo = {8, 8, x1.shape()[1], scale.shape().back(), 1};
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MatmulCompressDequant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulCompressDequant, x1, x2, compress_index, bias, scale,
                                          offsetW_opt, offsetX, compressInfo, out));
  runner->Run({x1, x2, compress_index, bias, scale, offsetW}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_matmul_compress_dequant", PYBOOST_CALLER(1, custom::npu_matmul_compress_dequant));
}
}  // namespace custom
