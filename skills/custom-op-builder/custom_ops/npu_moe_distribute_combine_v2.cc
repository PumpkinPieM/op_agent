#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
bool HasOpApi(const char *api_name) { return mindspore::device::ascend::GetOpApiFunc(api_name) != nullptr; }

ms::TypeId OutputDType(const ms::Tensor &expand_x) {
  return expand_x.data_type() == ms::TypeId::kNumberTypeInt32 ? ms::TypeId::kNumberTypeBFloat16 : expand_x.data_type();
}
}  // namespace

ms::Tensor npu_moe_distribute_combine_v2(
  const ms::Tensor &expand_x, const ms::Tensor &expert_ids, const ms::Tensor &assist_info_for_combine,
  const ms::Tensor &ep_send_counts, const ms::Tensor &expert_scales, const std::string &group_ep,
  int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num,
  const std::optional<ms::Tensor> &tp_send_counts_opt, const std::optional<ms::Tensor> &x_active_mask_opt,
  const std::optional<ms::Tensor> &expand_scales_opt, const std::optional<ms::Tensor> &shared_expert_x_opt,
  const std::optional<ms::Tensor> &elastic_info_opt, const std::optional<ms::Tensor> &ori_x_opt,
  const std::optional<ms::Tensor> &const_expert_alpha_1_opt,
  const std::optional<ms::Tensor> &const_expert_alpha_2_opt, const std::optional<ms::Tensor> &const_expert_v_opt,
  const std::optional<ms::Tensor> &performance_info_opt, const std::string &group_tp, int64_t tp_world_size,
  int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
  int64_t global_bs, int64_t comm_quant_mode, const std::string &comm_alg, int64_t zero_expert_num,
  int64_t copy_expert_num, int64_t const_expert_num) {
  const auto expand_shape = expand_x.shape();
  const auto expert_shape = expert_ids.shape();
  const int64_t global_bs_real = global_bs == 0 ? expert_shape[0] * ep_world_size : global_bs;
  auto out = ms::Tensor(OutputDType(expand_x), std::vector<int64_t>{expert_shape[0], expand_shape[1]});

  const std::optional<ms::Tensor> null_tensor = std::nullopt;
  constexpr int64_t out_dtype = 0;
  constexpr int64_t group_list_type = 0;

  auto tp_send_counts = tp_send_counts_opt.value_or(ms::Tensor());
  auto x_active_mask = x_active_mask_opt.value_or(ms::Tensor());
  auto expand_scales = expand_scales_opt.value_or(ms::Tensor());
  auto shared_expert_x = shared_expert_x_opt.value_or(ms::Tensor());
  auto elastic_info = elastic_info_opt.value_or(ms::Tensor());
  auto ori_x = ori_x_opt.value_or(ms::Tensor());
  auto const_expert_alpha_1 = const_expert_alpha_1_opt.value_or(ms::Tensor());
  auto const_expert_alpha_2 = const_expert_alpha_2_opt.value_or(ms::Tensor());
  auto const_expert_v = const_expert_v_opt.value_or(ms::Tensor());
  auto performance_info = performance_info_opt.value_or(ms::Tensor());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeDistributeCombineV2");
  if (HasOpApi("aclnnMoeDistributeCombineV4")) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMoeDistributeCombineV4, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales,
      tp_send_counts_opt, x_active_mask_opt, null_tensor, null_tensor, null_tensor, expand_scales_opt,
      shared_expert_x_opt, elastic_info_opt, ori_x_opt, const_expert_alpha_1_opt, const_expert_alpha_2_opt,
      const_expert_v_opt, performance_info_opt, group_ep, ep_world_size, ep_rank_id, moe_expert_num, group_tp,
      tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs_real,
      out_dtype, comm_quant_mode, group_list_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, out));
  } else if (HasOpApi("aclnnMoeDistributeCombineV3")) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMoeDistributeCombineV3, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales,
      tp_send_counts_opt, x_active_mask_opt, null_tensor, null_tensor, null_tensor, expand_scales_opt,
      shared_expert_x_opt, elastic_info_opt, ori_x_opt, const_expert_alpha_1_opt, const_expert_alpha_2_opt,
      const_expert_v_opt, group_ep, ep_world_size, ep_rank_id, moe_expert_num, group_tp, tp_world_size, tp_rank_id,
      expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs_real, out_dtype, comm_quant_mode,
      group_list_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, out));
  } else {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMoeDistributeCombineV2, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales,
      tp_send_counts_opt, x_active_mask_opt, null_tensor, null_tensor, null_tensor, expand_scales_opt,
      shared_expert_x_opt, group_ep, ep_world_size, ep_rank_id, moe_expert_num, group_tp, tp_world_size, tp_rank_id,
      expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs_real, out_dtype, comm_quant_mode,
      group_list_type, comm_alg, out));
  }
  runner->Run({expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, tp_send_counts,
               x_active_mask, expand_scales, shared_expert_x, elastic_info, ori_x, const_expert_alpha_1,
               const_expert_alpha_2, const_expert_v, performance_info},
              {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_distribute_combine_v2", PYBOOST_CALLER(1, custom::npu_moe_distribute_combine_v2));
}
}  // namespace custom
