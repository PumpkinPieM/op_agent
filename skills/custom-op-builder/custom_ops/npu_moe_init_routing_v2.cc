#include <vector>
#include <algorithm>
#include <optional>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kQuantModeStatic = 0;
constexpr int64_t kQuantModeDynamic = 1;
constexpr int64_t kQuantModeMxfp8E5m2 = 2;
constexpr int64_t kQuantModeMxfp8E4m3fn = 3;
constexpr int64_t kQuantModeHif8Cast = 6;
constexpr int64_t kQuantModeHif8PerTensor = 7;
constexpr int64_t kQuantModeHif8PerToken = 8;
constexpr int64_t kMxQuantBlockSize = 32;
constexpr int64_t kPadToEvenFactor = 2;

int64_t ExpertRangeLength(const std::vector<int64_t> &active_expert_range, int64_t expert_num) {
  if (active_expert_range.size() == 2) {
    return active_expert_range[1] - active_expert_range[0];
  }
  return expert_num;
}

ms::TypeId ExpandedXDType(const ms::Tensor &x, int64_t quant_mode) {
  if (quant_mode == kQuantModeStatic || quant_mode == kQuantModeDynamic) {
    return ms::TypeId::kNumberTypeInt8;
  }
  if (quant_mode == kQuantModeMxfp8E5m2 || quant_mode == kQuantModeMxfp8E4m3fn ||
      quant_mode == kQuantModeHif8Cast || quant_mode == kQuantModeHif8PerTensor ||
      quant_mode == kQuantModeHif8PerToken) {
    return ms::TypeId::kNumberTypeUInt8;
  }
  return x.data_type();
}

std::vector<int64_t> ExpandedXShape(const ms::Tensor &x, const ms::Tensor &expert_idx, int64_t active_num,
                                    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode) {
  const int64_t bs = x.shape()[0];
  const int64_t h = x.shape()[1];
  const int64_t k = expert_idx.shape()[1];
  if (drop_pad_mode == 1) {
    return {expert_num, expert_capacity, h};
  }
  return {active_num <= 0 ? bs * k : std::min(active_num, bs * k), h};
}

std::vector<int64_t> ExpertTokensShape(int64_t expert_num, int64_t expert_range_length,
                                       int64_t expert_tokens_num_type, bool expert_tokens_num_flag) {
  if (!expert_tokens_num_flag) {
    return {};
  }
  if (expert_tokens_num_type == 2) {
    return {expert_num, 2};
  }
  return {expert_range_length};
}

std::vector<int64_t> ExpandedScaleShape(const ms::Tensor &x, const ms::Tensor &expert_idx, int64_t active_num,
                                        int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode,
                                        int64_t quant_mode) {
  if (quant_mode == kQuantModeStatic || quant_mode == kQuantModeHif8Cast || quant_mode == kQuantModeHif8PerTensor) {
    return {};
  }
  const auto expanded_x_shape = ExpandedXShape(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode);
  const int64_t rows = drop_pad_mode == 1 ? expert_num * expert_capacity : expanded_x_shape[0];
  if (quant_mode == kQuantModeMxfp8E5m2 || quant_mode == kQuantModeMxfp8E4m3fn) {
    int64_t scale_cols = (x.shape()[1] + kMxQuantBlockSize - 1) / kMxQuantBlockSize;
    scale_cols = (scale_cols + kPadToEvenFactor - 1) / kPadToEvenFactor * kPadToEvenFactor;
    return {rows, scale_cols};
  }
  return {rows};
}
}  // namespace

std::vector<ms::Tensor> npu_moe_init_routing_v2(
    const ms::Tensor &x, const ms::Tensor &expert_idx, const std::optional<ms::Tensor> &scale_opt = std::nullopt,
    const std::optional<ms::Tensor> &offset_opt = std::nullopt, int64_t active_num = -1,
    int64_t expert_capacity = -1, int64_t expert_num = -1, int64_t drop_pad_mode = 0,
    int64_t expert_tokens_num_type = 0, bool expert_tokens_num_flag = false, int64_t quant_mode = -1,
    const std::optional<std::vector<int64_t>> &active_expert_range_opt = std::nullopt, int64_t row_idx_type = 0,
    const std::optional<int64_t> &x_dtype_opt = std::nullopt) {
  (void)x_dtype_opt;
  auto scale = scale_opt.value_or(ms::Tensor());
  auto offset = offset_opt.value_or(ms::Tensor());
  auto active_expert_range_value = active_expert_range_opt.value_or(std::vector<int64_t>{});
  auto active_expert_range = std::make_pair(active_expert_range_value, true);
  const int64_t expert_range_length = ExpertRangeLength(active_expert_range_value, expert_num);

  auto expanded_x = ms::Tensor(ExpandedXDType(x, quant_mode),
                               ExpandedXShape(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode));
  auto expanded_row_idx = ms::Tensor(ms::TypeId::kNumberTypeInt32,
                                     std::vector<int64_t>{x.shape()[0] * expert_idx.shape()[1]});
  auto expert_tokens_count_or_cumsum =
      ms::Tensor(ms::TypeId::kNumberTypeInt64,
                 ExpertTokensShape(expert_num, expert_range_length, expert_tokens_num_type, expert_tokens_num_flag));
  auto expanded_scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32,
                                   ExpandedScaleShape(x, expert_idx, active_num, expert_capacity, expert_num,
                                                      drop_pad_mode, quant_mode));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeInitRoutingV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeInitRoutingV3, x, expert_idx, scale_opt, offset_opt, active_num,
                                          expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
                                          expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type,
                                          expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
                                          expanded_scale));
  runner->Run({x, expert_idx, scale, offset},
              {expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale});
  return {expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_init_routing_v2", PYBOOST_CALLER(4, custom::npu_moe_init_routing_v2));
}
}  // namespace custom
