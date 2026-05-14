#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> OutputShape(const ms::Tensor &x1, const ms::Tensor &x2, int64_t world_size) {
  auto x1_shape = x1.shape();
  auto x2_shape = x2.shape();
  if (x1_shape.size() != 2 || x2_shape.size() != 2 || world_size <= 0) {
    return x1_shape;
  }
  return {x1_shape[0] * world_size, x2_shape[1] / world_size};
}
}  // namespace

ms::Tensor npu_matmul_all_to_all(const ms::Tensor &x1, const ms::Tensor &x2, const std::string &hcom, int64_t world_size,
                                 const std::optional<ms::Tensor> &bias_opt = std::nullopt,
                                 const std::optional<std::vector<int64_t>> &axes_opt = std::nullopt) {
  auto bias = bias_opt.value_or(ms::Tensor());
  auto axes = std::make_pair(axes_opt.value_or(std::vector<int64_t>{-1, -2}), true);
  auto y = ms::Tensor(x1.data_type(), OutputShape(x1, x2, world_size));
  constexpr bool transpose_x1 = false;
  constexpr bool transpose_x2 = false;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MatmulAlltoAll");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulAlltoAll, x1, x2, bias_opt, axes, hcom, transpose_x1,
                                          transpose_x2, y));
  runner->Run({x1, x2, bias}, {y});
  return y;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_matmul_all_to_all", PYBOOST_CALLER(1, custom::npu_matmul_all_to_all));
}
}  // namespace custom
