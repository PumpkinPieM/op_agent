#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kQuantModeNoQuant = 0;
constexpr int64_t kQuantModePerToken = 2;
constexpr int64_t kQuantModePerGroup = 3;
constexpr int64_t kQuantModeMx = 4;

bool HasOpApi(const char *api_name) { return mindspore::device::ascend::GetOpApiFunc(api_name) != nullptr; }

ms::TypeId DTypeFromOptional(const std::optional<int64_t> &dtype, ms::TypeId fallback) {
  if (!dtype.has_value() || dtype.value() < 0) {
    return fallback;
  }
  switch (dtype.value()) {
    case 0:
    case 6:
      return ms::TypeId::kNumberTypeFloat32;
    case 1:
    case 5:
      return ms::TypeId::kNumberTypeFloat16;
    case 2:
      return ms::TypeId::kNumberTypeInt8;
    case 3:
      return ms::TypeId::kNumberTypeInt32;
    case 4:
      return ms::TypeId::kNumberTypeUInt8;
    case 9:
      return ms::TypeId::kNumberTypeInt64;
    case 27:
      return ms::TypeId::kNumberTypeBFloat16;
    default:
      return fallback;
  }
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t GlobalBs(int64_t bs, int64_t ep_world_size, int64_t global_bs) {
  return global_bs == 0 ? bs * ep_world_size : global_bs;
}

std::pair<int64_t, int64_t> LocalExpertAndA(const ms::Tensor &x, const ms::Tensor &expert_ids,
                                            const std::optional<ms::Tensor> &elastic_info_opt,
                                            int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
                                            int64_t expert_shard_type, int64_t shared_expert_num,
                                            int64_t shared_expert_rank_num, int64_t global_bs) {
  const auto x_shape = x.shape();
  const auto expert_shape = expert_ids.shape();
  const int64_t bs = x_shape[0];
  const int64_t k = expert_shape[1];
  const int64_t global_bs_real = GlobalBs(bs, ep_world_size, global_bs);
  const bool shared_front = (expert_shard_type == 0);
  const bool is_shared_default = (shared_expert_num == 1 && shared_expert_rank_num == 0);
  const bool is_no_shared = (shared_expert_num == 0 && shared_expert_rank_num == 0);

  int64_t local_moe_expert_num = 1;
  int64_t a = 0;
  if (shared_front && ep_rank_id < shared_expert_rank_num) {
    local_moe_expert_num = 1;
    const int64_t max_bs = global_bs_real / ep_world_size;
    const int64_t rank_num_per_shared_expert = shared_expert_rank_num / std::max<int64_t>(1, shared_expert_num);
    const int64_t max_shared_group_num = CeilDiv(ep_world_size, rank_num_per_shared_expert);
    a = max_bs * max_shared_group_num;
  } else {
    local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
    a = global_bs_real * std::min(local_moe_expert_num, k);
  }

  if (shared_front && elastic_info_opt.has_value()) {
    if (is_shared_default || is_no_shared) {
      local_moe_expert_num =
        std::max(local_moe_expert_num, moe_expert_num / (ep_world_size - shared_expert_rank_num));
      a = global_bs_real * std::min(local_moe_expert_num, k);
    } else {
      const int64_t max_bs = global_bs_real / ep_world_size;
      const int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;
      const int64_t max_shared_group_num = CeilDiv(ep_world_size, rank_num_per_shared_expert);
      const int64_t non_shared_local = moe_expert_num / (ep_world_size - shared_expert_rank_num);
      a = std::max(max_bs * max_shared_group_num, global_bs_real * std::min(non_shared_local, k));
      local_moe_expert_num = std::max(local_moe_expert_num, non_shared_local);
    }
  }
  return {local_moe_expert_num, a};
}

std::vector<int64_t> DynamicScalesShape(const std::optional<ms::Tensor> &scales_opt, int64_t quant_mode, int64_t a,
                                        int64_t h, int64_t tp_world_size) {
  if (tp_world_size == 0) {
    return {a};
  }
  if (tp_world_size > 1) {
    return {a * tp_world_size};
  }
  if (quant_mode == kQuantModeNoQuant && scales_opt.has_value()) {
    const auto scales_shape = scales_opt.value().shape();
    return scales_shape.size() >= 2 ? std::vector<int64_t>{a, scales_shape[1]} : std::vector<int64_t>{a};
  }
  if (quant_mode == kQuantModePerToken) {
    return {a};
  }
  if (quant_mode == kQuantModePerGroup) {
    return {a, CeilDiv(h, 128)};
  }
  if (quant_mode == kQuantModeMx) {
    return {a, CeilDiv(CeilDiv(h, 32), 2) * 2};
  }
  return {a};
}

ms::TypeId DynamicScalesDType(const ms::Tensor &x, const std::optional<ms::Tensor> &scales_opt, int64_t quant_mode,
                              const std::optional<int64_t> &scales_dtype_opt) {
  if (quant_mode == kQuantModeMx) {
    return ms::TypeId::kNumberTypeUInt8;
  }
  if (quant_mode == kQuantModeNoQuant && x.data_type() != ms::TypeId::kNumberTypeBFloat16 &&
      x.data_type() != ms::TypeId::kNumberTypeFloat16 && scales_opt.has_value()) {
    return DTypeFromOptional(scales_dtype_opt, scales_opt.value().data_type());
  }
  return ms::TypeId::kNumberTypeFloat32;
}
}  // namespace

std::vector<ms::Tensor> npu_moe_distribute_dispatch_v2(
  const ms::Tensor &x, const ms::Tensor &expert_ids, const std::string &group_ep, int64_t ep_world_size,
  int64_t ep_rank_id, int64_t moe_expert_num, const std::optional<ms::Tensor> &scales_opt,
  const std::optional<ms::Tensor> &x_active_mask_opt, const std::optional<ms::Tensor> &expert_scales_opt,
  const std::optional<ms::Tensor> &elastic_info_opt, const std::optional<ms::Tensor> &performance_info_opt,
  const std::string &group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type,
  int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs,
  int64_t expert_token_nums_type, const std::string &comm_alg, int64_t zero_expert_num, int64_t copy_expert_num,
  int64_t const_expert_num, const std::optional<int64_t> &y_dtype_opt, const std::optional<int64_t> &x_dtype_opt,
  const std::optional<int64_t> &scales_dtype_opt) {
  const auto x_shape = x.shape();
  const auto expert_shape = expert_ids.shape();
  const int64_t bs = x_shape[0];
  const int64_t h = x_shape[1];
  const int64_t k = expert_shape[1];
  const int64_t global_bs_real = GlobalBs(bs, ep_world_size, global_bs);
  const auto [local_moe_expert_num, a] =
    LocalExpertAndA(x, expert_ids, elastic_info_opt, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type,
                    shared_expert_num, shared_expert_rank_num, global_bs);
  const int64_t ep_recv_count_num = expert_scales_opt.has_value()
                                      ? ep_world_size * local_moe_expert_num +
                                          2 * global_bs_real * k * (ep_world_size / 8)
                                      : ep_world_size * local_moe_expert_num *
                                          (tp_world_size == 2 ? tp_world_size : 1);

  const ms::TypeId expand_dtype =
    quant_mode == kQuantModeNoQuant ? DTypeFromOptional(y_dtype_opt, x.data_type())
                                    : DTypeFromOptional(y_dtype_opt, ms::TypeId::kNumberTypeInt8);
  auto expand_x = ms::Tensor(expand_dtype, std::vector<int64_t>{std::max(a, a * tp_world_size), h});
  auto dynamic_scales =
    ms::Tensor(DynamicScalesDType(x, scales_opt, quant_mode, scales_dtype_opt),
               DynamicScalesShape(scales_opt, quant_mode, a, h, tp_world_size));
  auto assist_info_for_combine =
    ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{std::max(bs * k, a * 128)});
  auto expert_token_nums = ms::Tensor(ms::TypeId::kNumberTypeInt64, std::vector<int64_t>{local_moe_expert_num});
  auto ep_recv_counts = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{ep_recv_count_num});
  auto tp_recv_counts = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{tp_world_size});
  auto expand_scales = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{a});

  auto scales = scales_opt.value_or(ms::Tensor());
  auto x_active_mask = x_active_mask_opt.value_or(ms::Tensor());
  auto expert_scales = expert_scales_opt.value_or(ms::Tensor());
  auto elastic_info = elastic_info_opt.value_or(ms::Tensor());
  auto performance_info = performance_info_opt.value_or(ms::Tensor());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeDistributeDispatchV2");
  if (HasOpApi("aclnnMoeDistributeDispatchV4")) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMoeDistributeDispatchV4, x, expert_ids, scales_opt, x_active_mask_opt, expert_scales_opt, elastic_info_opt,
      performance_info_opt, group_ep, ep_world_size, ep_rank_id, moe_expert_num, group_tp, tp_world_size, tp_rank_id,
      expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs_real,
      expert_token_nums_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, expand_x, dynamic_scales,
      assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales));
  } else if (HasOpApi("aclnnMoeDistributeDispatchV3")) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMoeDistributeDispatchV3, x, expert_ids, scales_opt, x_active_mask_opt, expert_scales_opt, elastic_info_opt,
      group_ep, ep_world_size, ep_rank_id, moe_expert_num, group_tp, tp_world_size, tp_rank_id, expert_shard_type,
      shared_expert_num, shared_expert_rank_num, quant_mode, global_bs_real, expert_token_nums_type, comm_alg,
      zero_expert_num, copy_expert_num, const_expert_num, expand_x, dynamic_scales, assist_info_for_combine,
      expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales));
  } else {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMoeDistributeDispatchV2, x, expert_ids, scales_opt, x_active_mask_opt, expert_scales_opt, group_ep,
      ep_world_size, ep_rank_id, moe_expert_num, group_tp, tp_world_size, tp_rank_id, expert_shard_type,
      shared_expert_num, shared_expert_rank_num, quant_mode, global_bs_real, expert_token_nums_type, comm_alg,
      expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts,
      expand_scales));
  }
  runner->Run({x, expert_ids, scales, x_active_mask, expert_scales, elastic_info, performance_info},
              {expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts,
               expand_scales});
  return {expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts,
          expand_scales};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_distribute_dispatch_v2", PYBOOST_CALLER(7, custom::npu_moe_distribute_dispatch_v2));
}
}  // namespace custom
