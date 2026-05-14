#include <algorithm>
#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kInt4Pack = 8;

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
    case 27:
      return ms::TypeId::kNumberTypeBFloat16;
    default:
      return fallback;
  }
}

std::vector<int64_t> BroadcastBatch(const std::vector<int64_t> &x1_shape, const std::vector<int64_t> &x2_shape) {
  const size_t x1_batch = x1_shape.size() > 2 ? x1_shape.size() - 2 : 0;
  const size_t x2_batch = x2_shape.size() > 2 ? x2_shape.size() - 2 : 0;
  const size_t out_batch = std::max(x1_batch, x2_batch);
  std::vector<int64_t> out(out_batch, 1);
  for (size_t i = 0; i < out_batch; ++i) {
    const int64_t x1_dim = i + x1_batch >= out_batch ? x1_shape[i + x1_batch - out_batch] : 1;
    const int64_t x2_dim = i + x2_batch >= out_batch ? x2_shape[i + x2_batch - out_batch] : 1;
    out[i] = std::max(x1_dim, x2_dim);
  }
  return out;
}

std::vector<int64_t> MatmulShape(const ms::Tensor &x1, const ms::Tensor &x2) {
  const auto x1_shape = x1.shape();
  const auto x2_shape = x2.shape();
  if (x1_shape.size() < 2 || x2_shape.size() < 2) {
    return x1_shape;
  }
  auto out = BroadcastBatch(x1_shape, x2_shape);
  const bool is_a4w4 = x1.data_type() == ms::TypeId::kNumberTypeInt32 &&
                       x2.data_type() == ms::TypeId::kNumberTypeInt32;
  out.push_back(x1_shape[x1_shape.size() - 2]);
  out.push_back(is_a4w4 ? x2_shape.back() * kInt4Pack : x2_shape.back());
  return out;
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

std::vector<int64_t> TransQuantParamShape(const ms::Tensor &scale, const std::optional<ms::Tensor> &offset_opt) {
  auto scale_shape = scale.shape();
  if (scale_shape.size() == 1 && offset_opt.has_value()) {
    auto offset_shape = offset_opt.value().shape();
    if (offset_shape.size() == 1 && offset_shape[0] > scale_shape[0]) {
      return offset_shape;
    }
  }
  return scale_shape;
}
}  // namespace

ms::Tensor npu_quant_matmul(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &scale,
                            const std::optional<ms::Tensor> &offset_opt,
                            const std::optional<ms::Tensor> &pertoken_scale_opt,
                            const std::optional<ms::Tensor> &bias_opt,
                            const std::optional<int64_t> &output_dtype_opt,
                            const std::optional<int64_t> &x1_dtype_opt,
                            const std::optional<int64_t> &x2_dtype_opt,
                            const std::optional<int64_t> &pertoken_scale_dtype_opt,
                            const std::optional<int64_t> &scale_dtype_opt,
                            const std::optional<std::vector<int64_t>> &group_sizes_opt,
                            const std::optional<ms::Tensor> &y_scale_opt) {
  (void)x1_dtype_opt;
  (void)x2_dtype_opt;
  (void)pertoken_scale_dtype_opt;
  (void)scale_dtype_opt;

  auto out = ms::Tensor(DTypeFromOptional(output_dtype_opt, ms::TypeId::kNumberTypeInt8), MatmulShape(x1, x2));
  auto pertoken_scale = pertoken_scale_opt.value_or(ms::Tensor());
  auto bias = bias_opt.value_or(ms::Tensor());
  auto y_scale = y_scale_opt.value_or(ms::Tensor());
  auto offset = offset_opt.value_or(ms::Tensor());

  const bool is_a8w4 = x1.data_type() == ms::TypeId::kNumberTypeInt8 &&
                       x2.data_type() == ms::TypeId::kNumberTypeInt32;
  std::optional<ms::Tensor> x2_offset_opt = is_a8w4 ? std::nullopt : offset_opt;
  std::optional<ms::Tensor> y_offset_opt = is_a8w4 ? offset_opt : std::nullopt;
  std::optional<ms::Tensor> x1_offset_opt = std::nullopt;
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  int64_t group_size = PackGroupSize(group_sizes_opt);
  const bool output_is_bfloat16 = output_dtype_opt.has_value() && output_dtype_opt.value() == 27;
  const bool output_is_int32 = output_dtype_opt.has_value() && output_dtype_opt.value() == 3;
  const bool use_trans_quant_param = scale.data_type() == ms::TypeId::kNumberTypeFloat32 &&
                                     !pertoken_scale_opt.has_value() && !output_is_bfloat16 && !output_is_int32;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulV5");
  if (use_trans_quant_param) {
    auto quant_param = ms::Tensor(ms::TypeId::kNumberTypeInt64, TransQuantParamShape(scale, offset_opt));
    auto trans_runner = std::make_shared<ms::pynative::AclnnOpRunner>("TransQuantParamV2");
    trans_runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnTransQuantParamV2, scale, offset_opt, quant_param));
    trans_runner->Run({scale, offset}, {quant_param});
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulV5, x1, x2, pertoken_scale_opt, quant_param, y_scale_opt,
                                            x1_offset_opt, x2_offset_opt, y_offset_opt, bias_opt, transpose_x1,
                                            transpose_x2, group_size, out));
    runner->Run({x1, x2, quant_param, pertoken_scale, bias, y_scale, offset}, {out});
  } else {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulV5, x1, x2, pertoken_scale_opt, scale, y_scale_opt,
                                            x1_offset_opt, x2_offset_opt, y_offset_opt, bias_opt, transpose_x1,
                                            transpose_x2, group_size, out));
    runner->Run({x1, x2, scale, pertoken_scale, bias, y_scale, offset}, {out});
  }
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_matmul", PYBOOST_CALLER(1, custom::npu_quant_matmul));
}
}  // namespace custom
