#include <optional>
#include <stdexcept>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> Int32OutShape(const ms::Tensor &x) {
  auto shape = x.shape();
  if (shape.size() != 3) {
    throw std::runtime_error("npu_kronecker_quant: x must be 3D");
  }
  if (shape.back() % 8 != 0) {
    throw std::runtime_error("npu_kronecker_quant: last dim of x must be divisible by 8 for int32 output");
  }
  shape.back() /= 8;
  return shape;
}
}  // namespace

std::vector<ms::Tensor> npu_kronecker_quant(const ms::Tensor &x, const ms::Tensor &kronecker_p1, const ms::Tensor &kronecker_p2,
                                            const std::optional<double> &clip_ratio_opt = std::nullopt,
                                            const std::optional<int64_t> &dst_dtype_opt = std::nullopt) {
  if (dst_dtype_opt.has_value() && dst_dtype_opt.value() != 3) {
    throw std::runtime_error("npu_kronecker_quant: only int32 dst_dtype is supported by this adapter");
  }
  auto clip_ratio = clip_ratio_opt.value_or(1.0);
  auto out = ms::Tensor(ms::TypeId::kNumberTypeInt32, Int32OutShape(x));
  auto scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{x.shape()[0]});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FlatQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFlatQuant, x, kronecker_p1, kronecker_p2, clip_ratio, out, scale));
  runner->Run({x, kronecker_p1, kronecker_p2}, {out, scale});
  return {out, scale};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_kronecker_quant", PYBOOST_CALLER(2, custom::npu_kronecker_quant));
}
}  // namespace custom
