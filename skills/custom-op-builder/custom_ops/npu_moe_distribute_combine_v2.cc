#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
ms::TypeId DTypeFromOptional(const std::optional<int64_t> &dtype, ms::TypeId fallback) {
  if (!dtype.has_value() || dtype.value() < 0) { return fallback; }
  switch (dtype.value()) {
    case 0: return ms::TypeId::kNumberTypeUInt8;
    case 1: return ms::TypeId::kNumberTypeInt8;
    case 2: return ms::TypeId::kNumberTypeInt16;
    case 3: return ms::TypeId::kNumberTypeInt32;
    case 4: return ms::TypeId::kNumberTypeInt64;
    case 5: return ms::TypeId::kNumberTypeFloat16;
    case 6: return ms::TypeId::kNumberTypeFloat32;
    case 27: return ms::TypeId::kNumberTypeBFloat16;
    default: return fallback;
  }
}
std::vector<int64_t> MatmulShape(const ms::Tensor &x1, const ms::Tensor &x2) {
  auto s1 = x1.shape();
  auto s2 = x2.shape();
  if (s1.size() < 2 || s2.size() < 2) { return s1; }
  std::vector<int64_t> out;
  if (s1.size() > 2) { out.insert(out.end(), s1.begin(), s1.end() - 2); }
  out.push_back(s1[s1.size() - 2]);
  out.push_back(s2.back());
  return out;
}
std::vector<int64_t> Conv2dShape(const ms::Tensor &input, const ms::Tensor &weight, const std::vector<int64_t> &strides, const std::vector<int64_t> &pads, const std::vector<int64_t> &dilations) {
  auto x=input.shape(); auto w=weight.shape(); if (x.size()!=4 || w.size()!=4) return x;
  int64_t sh=strides.empty()?1:strides[0], sw=strides.size()>1?strides[1]:sh;
  int64_t ph=pads.empty()?0:pads[0], pw=pads.size()>1?pads[1]:ph;
  int64_t dh=dilations.empty()?1:dilations[0], dw=dilations.size()>1?dilations[1]:dh;
  int64_t oh=(x[2]+2*ph-dh*(w[2]-1)-1)/sh+1; int64_t ow=(x[3]+2*pw-dw*(w[3]-1)-1)/sw+1;
  return {x[0], w[0], std::max<int64_t>(1,oh), std::max<int64_t>(1,ow)};
}
}  // namespace

ms::Tensor npu_moe_distribute_combine_v2(const ms::Tensor & expand_x, const ms::Tensor & expert_ids, const ms::Tensor & assist_info_for_combine, const ms::Tensor & ep_send_counts, const ms::Tensor & expert_scales, const std::string & group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const std::optional<ms::Tensor> & tp_send_counts_opt, const std::optional<ms::Tensor> & x_active_mask_opt, const std::optional<ms::Tensor> & expand_scales_opt, const std::optional<ms::Tensor> & shared_expert_x_opt, const std::optional<ms::Tensor> & elastic_info_opt, const std::optional<ms::Tensor> & ori_x_opt, const std::optional<ms::Tensor> & const_expert_alpha_1_opt, const std::optional<ms::Tensor> & const_expert_alpha_2_opt, const std::optional<ms::Tensor> & const_expert_v_opt, const std::optional<ms::Tensor> & performance_info_opt, const std::string & group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t comm_quant_mode, const std::string & comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num) {
  auto out=ms::Tensor(expand_x.data_type(), expand_x.shape()); auto tp_send_counts=tp_send_counts_opt.value_or(ms::Tensor()); auto x_active_mask=x_active_mask_opt.value_or(ms::Tensor()); auto expand_scales=expand_scales_opt.value_or(ms::Tensor()); auto shared_expert_x=shared_expert_x_opt.value_or(ms::Tensor()); auto elastic_info=elastic_info_opt.value_or(ms::Tensor()); auto ori_x=ori_x_opt.value_or(ms::Tensor()); auto const_expert_alpha_1=const_expert_alpha_1_opt.value_or(ms::Tensor()); auto const_expert_alpha_2=const_expert_alpha_2_opt.value_or(ms::Tensor()); auto const_expert_v=const_expert_v_opt.value_or(ms::Tensor()); auto performance_info=performance_info_opt.value_or(ms::Tensor());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeDistributeCombineV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeDistributeCombineV2, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, tp_send_counts_opt, x_active_mask_opt, expand_scales_opt, shared_expert_x_opt, elastic_info_opt, ori_x_opt, const_expert_alpha_1_opt, const_expert_alpha_2_opt, const_expert_v_opt, performance_info_opt, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, comm_quant_mode, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, out));
  runner->Run({expand_x,expert_ids,assist_info_for_combine,ep_send_counts,expert_scales,tp_send_counts,x_active_mask,expand_scales,shared_expert_x,elastic_info,ori_x,const_expert_alpha_1,const_expert_alpha_2,const_expert_v,performance_info}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_distribute_combine_v2", PYBOOST_CALLER(1, custom::npu_moe_distribute_combine_v2));
}
}  // namespace custom
