#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kDynPertokenQuantMode = 7;
constexpr int64_t kPerchannelQuantMode = 2;
constexpr int64_t kNonQuant = 0;
constexpr int64_t kAclUndefined = -1;
constexpr int64_t kAclFloat8E5M2 = 35;
constexpr int64_t kInt4NumsInInt32 = 8;

ms::TypeId DTypeFromOptional(const std::optional<int64_t> &dtype, ms::TypeId fallback) {
  if (!dtype.has_value() || dtype.value() < 0) {
    return fallback;
  }
  switch (dtype.value()) {
    case 1:
      return ms::TypeId::kNumberTypeInt8;
    case 3:
      return ms::TypeId::kNumberTypeInt32;
    case 5:
      return ms::TypeId::kNumberTypeFloat16;
    case 6:
      return ms::TypeId::kNumberTypeFloat32;
    case 15:
    case 27:
      return ms::TypeId::kNumberTypeBFloat16;
    default:
      return fallback;
  }
}

std::vector<int64_t> OutputShape(const ms::Tensor &x1, const ms::Tensor &x2, int64_t world_size) {
  const auto x1_shape = x1.shape();
  const auto x2_shape = x2.shape();
  if (x1_shape.size() != 2 || x2_shape.size() != 2 || world_size == 0) {
    return x1_shape;
  }
  const bool is_w4 = x2.data_type() == ms::TypeId::kNumberTypeInt32;
  return {x1_shape[0] / world_size, is_w4 ? x2_shape[1] * kInt4NumsInInt32 : x2_shape[1]};
}

std::vector<int64_t> AlltoAllOutShape(const ms::Tensor &x1, int64_t world_size) {
  const auto x1_shape = x1.shape();
  if (x1_shape.size() != 2 || world_size == 0) {
    return x1_shape;
  }
  return {x1_shape[0] / world_size, x1_shape[1] * world_size};
}

int64_t PackGroupSize(const std::optional<std::vector<int64_t>> &group_sizes_opt) {
  if (!group_sizes_opt.has_value() || group_sizes_opt.value().empty()) {
    return 0;
  }
  const auto &group_sizes = group_sizes_opt.value();
  const int64_t group_m = group_sizes.size() > 0 ? group_sizes[0] : 0;
  const int64_t group_n = group_sizes.size() > 1 ? group_sizes[1] : 0;
  const int64_t group_k = group_sizes.size() > 2 ? group_sizes[2] : 0;
  return (group_m << 32) + (group_n << 16) + group_k;
}
}  // namespace

std::vector<ms::Tensor> npu_all_to_all_quant_matmul(const ms::Tensor &x1, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, bool all2all_out_flag, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt, const std::optional<ms::Tensor> &common_scale_opt, const std::optional<ms::Tensor> &x1_offset_opt, const std::optional<ms::Tensor> &x2_offset_opt, const std::optional<int64_t> &x1_quant_mode_opt, const std::optional<int64_t> &x2_quant_mode_opt, const std::optional<int64_t> &common_quant_mode_opt, const std::optional<std::vector<int64_t>> &group_sizes_opt, const std::optional<std::vector<int64_t>> &all2all_axes_opt, const std::optional<int64_t> &comm_quant_dtype_opt, const std::optional<int64_t> &x1_quant_dtype_opt, const std::optional<int64_t> &x1_dtype_opt, const std::optional<int64_t> &x2_dtype_opt, const std::optional<int64_t> &x1_scale_dtype_opt, const std::optional<int64_t> &x2_scale_dtype_opt, const std::optional<int64_t> &output_scale_dtype_opt, const std::optional<int64_t> &comm_scale_dtype_opt, const std::optional<int64_t> &y_dtype_opt) {
  (void)x1_dtype_opt;
  (void)x2_dtype_opt;
  (void)x1_scale_dtype_opt;
  (void)x2_scale_dtype_opt;
  (void)output_scale_dtype_opt;
  (void)comm_scale_dtype_opt;

  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto x1_scale_value = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale_value = x2_scale_opt.value_or(ms::Tensor());
  auto common_scale_value = common_scale_opt.value_or(ms::Tensor());
  auto x1_offset_value = x1_offset_opt.value_or(ms::Tensor());
  auto x2_offset_value = x2_offset_opt.value_or(ms::Tensor());
  int64_t x1_quant_mode = x1_quant_mode_opt.value_or(kDynPertokenQuantMode);
  int64_t x2_quant_mode = x2_quant_mode_opt.value_or(kPerchannelQuantMode);
  int64_t common_quant_mode = common_quant_mode_opt.value_or(kNonQuant);
  auto all2all_axes = std::make_pair(all2all_axes_opt.value_or(std::vector<int64_t>{-2, -1}), true);
  int64_t comm_quant_dtype = comm_quant_dtype_opt.value_or(kAclUndefined);
  int64_t x1_quant_dtype = x1_quant_dtype_opt.value_or(kAclFloat8E5M2);
  int64_t group_size = PackGroupSize(group_sizes_opt);
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  auto out0 = ms::Tensor(DTypeFromOptional(y_dtype_opt, ms::TypeId::kNumberTypeFloat32), OutputShape(x1, x2, world_size));
  (void)all2all_out_flag;
  auto out1 = ms::Tensor(x1.data_type(), AlltoAllOutShape(x1, world_size));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AlltoAllQuantMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAlltoAllQuantMatmul, x1, x2, bias_opt, x1_scale_opt, x2_scale_opt,
                                          common_scale_opt, x1_offset_opt, x2_offset_opt, hcom, all2all_axes,
                                          x1_quant_mode, x2_quant_mode, common_quant_mode, comm_quant_dtype,
                                          x1_quant_dtype, group_size, transpose_x1, transpose_x2, out0, out1));
  runner->Run({x1, x2, bias_value, x1_scale_value, x2_scale_value, common_scale_value, x1_offset_value, x2_offset_value},
              {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_to_all_quant_matmul", PYBOOST_CALLER(2, custom::npu_all_to_all_quant_matmul));
}
}  // namespace custom
