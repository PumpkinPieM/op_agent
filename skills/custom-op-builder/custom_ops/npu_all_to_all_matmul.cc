#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> OutputShape(const ms::Tensor &x1, const ms::Tensor &x2, int64_t world_size) {
  const auto x1_shape = x1.shape();
  const auto x2_shape = x2.shape();
  if (x1_shape.size() != 2 || x2_shape.size() != 2 || world_size == 0) {
    return x1_shape;
  }
  return {x1_shape[0] / world_size, x2_shape[1]};
}

std::vector<int64_t> AlltoAllOutShape(const ms::Tensor &x1, int64_t world_size) {
  const auto x1_shape = x1.shape();
  if (x1_shape.size() != 2 || world_size == 0) {
    return x1_shape;
  }
  return {x1_shape[0] / world_size, x1_shape[1] * world_size};
}
}  // namespace

std::vector<ms::Tensor> npu_all_to_all_matmul(const ms::Tensor &x1, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, const std::optional<ms::Tensor> &bias_opt = std::nullopt, const std::optional<std::vector<int64_t>> &axes_opt = std::nullopt, bool out_flag = true) {
  auto bias = bias_opt.value_or(ms::Tensor());
  auto axes = std::make_pair(axes_opt.value_or(std::vector<int64_t>{-2, -1}), true);
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  auto y = ms::Tensor(x1.data_type(), OutputShape(x1, x2, world_size));
  (void)out_flag;
  auto alltoall_out = ms::Tensor(x1.data_type(), AlltoAllOutShape(x1, world_size));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AlltoAllMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAlltoAllMatmul, x1, x2, bias_opt, axes, hcom, transpose_x1,
                                          transpose_x2, y, alltoall_out));
  runner->Run({x1, x2, bias}, {y, alltoall_out});
  return {y, alltoall_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_to_all_matmul", PYBOOST_CALLER(2, custom::npu_all_to_all_matmul));
}
}  // namespace custom
