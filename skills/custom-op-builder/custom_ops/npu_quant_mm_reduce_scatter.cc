#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
ms::TypeId DTypeFromAclType(const std::optional<int64_t> &dtype, ms::TypeId fallback) {
  if (!dtype.has_value() || dtype.value() < 0) {
    return fallback;
  }
  switch (dtype.value()) {
    case 2:
      return ms::TypeId::kNumberTypeInt8;
    case 3:
      return ms::TypeId::kNumberTypeInt32;
    case 4:
      return ms::TypeId::kNumberTypeUInt8;
    case 5:
      return ms::TypeId::kNumberTypeFloat16;
    case 6:
      return ms::TypeId::kNumberTypeFloat32;
    case 27:
      return ms::TypeId::kNumberTypeBFloat16;
    default:
      return fallback;
  }
}

int64_t GroupSizeFromList(const std::optional<std::vector<int64_t>> &group_sizes_opt) {
  if (!group_sizes_opt.has_value() || group_sizes_opt->empty()) {
    return 0;
  }
  return group_sizes_opt->front();
}
}  // namespace

std::vector<ms::Tensor> npu_quant_mm_reduce_scatter(
    const ms::Tensor &self, const ms::Tensor &x2, const std::string &hcom, int64_t world_size,
    const std::optional<std::string> &reduce_op_opt, const std::optional<ms::Tensor> &bias_opt,
    const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt,
    const std::optional<ms::Tensor> &quant_scale_opt, int64_t block_size, int64_t comm_turn,
    const std::optional<std::vector<int64_t>> &group_sizes_opt, bool amax_output,
    const std::optional<int64_t> &y_dtype_opt, const std::optional<int64_t> &x1_dtype_opt,
    const std::optional<int64_t> &x2_dtype_opt, const std::optional<int64_t> &x1_scale_dtype_opt,
    const std::optional<int64_t> &x2_scale_dtype_opt) {
  (void)x1_dtype_opt;
  (void)x2_dtype_opt;
  (void)x1_scale_dtype_opt;
  (void)x2_scale_dtype_opt;

  auto self_shape = self.shape();
  auto x2_shape = x2.shape();
  std::vector<int64_t> out_shape = self_shape;
  if (self_shape.size() == 2 && x2_shape.size() == 2) {
    out_shape = {std::max<int64_t>(1, self_shape[0] / std::max<int64_t>(1, world_size)), x2_shape[1]};
  }

  auto out = ms::Tensor(DTypeFromAclType(y_dtype_opt, self.data_type()), out_shape);
  auto amax = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{1});
  auto bias = bias_opt.value_or(ms::Tensor());
  auto x1_scale = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale = x2_scale_opt.value_or(ms::Tensor());
  auto quant_scale = quant_scale_opt.value_or(ms::Tensor());
  auto reduce_op = reduce_op_opt.value_or("sum");
  int64_t stream_mode = 1;
  int64_t group_size = GroupSizeFromList(group_sizes_opt);
  const char *comm_mode = "aiv";

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MatmulReduceScatterV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulReduceScatterV2, self, x2, bias_opt, x1_scale_opt,
                                          x2_scale_opt, quant_scale_opt, block_size, hcom, reduce_op, comm_turn,
                                          stream_mode, group_size, comm_mode, out, amax));
  runner->Run({self, x2, bias, x1_scale, x2_scale, quant_scale}, {out, amax});
  return {out, amax};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_mm_reduce_scatter", PYBOOST_CALLER(2, custom::npu_quant_mm_reduce_scatter));
}
}  // namespace custom
