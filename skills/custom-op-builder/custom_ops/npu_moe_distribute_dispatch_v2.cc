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

std::vector<ms::Tensor> npu_moe_distribute_dispatch_v2(const ms::Tensor & x, const ms::Tensor & expert_ids, const std::string & group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const std::optional<ms::Tensor> & scales_opt, const std::optional<ms::Tensor> & x_active_mask_opt, const std::optional<ms::Tensor> & expert_scales_opt, const std::optional<ms::Tensor> & elastic_info_opt, const std::optional<ms::Tensor> & performance_info_opt, const std::string & group_tp, int64_t tp_world_size, int64_t tp_rank_id, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type, const std::string & comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num, const std::optional<int64_t> & y_dtype_opt, const std::optional<int64_t> & x_dtype_opt, const std::optional<int64_t> & scales_dtype_opt) {
  auto expand_x=ms::Tensor(DTypeFromOptional(y_dtype_opt.value_or(-1),x.data_type()), x.shape()); auto dynamic_scales=ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{1}); auto expert_token_nums=ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{moe_expert_num}); auto ep_recv_counts=ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{ep_world_size}); auto tp_recv_counts=ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{std::max<int64_t>(1,tp_world_size)}); auto expand_idx=ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{x.shape()[0]}); auto assist=ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{1}); auto scales=scales_opt.value_or(ms::Tensor()); auto x_active_mask=x_active_mask_opt.value_or(ms::Tensor()); auto expert_scales=expert_scales_opt.value_or(ms::Tensor()); auto elastic_info=elastic_info_opt.value_or(ms::Tensor()); auto performance_info=performance_info_opt.value_or(ms::Tensor());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeDistributeDispatchV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMoeDistributeDispatchV2, x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales_opt, x_active_mask_opt, expert_scales_opt, elastic_info_opt, performance_info_opt, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_alg, zero_expert_num, copy_expert_num, const_expert_num, y_dtype_opt.value_or(-1), x_dtype_opt.value_or(-1), scales_dtype_opt.value_or(-1), expand_x, dynamic_scales, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_idx, assist));
  runner->Run({x,expert_ids,scales,x_active_mask,expert_scales,elastic_info,performance_info}, {expand_x,dynamic_scales,expert_token_nums,ep_recv_counts,tp_recv_counts,expand_idx,assist});
  return {expand_x,dynamic_scales,expert_token_nums,ep_recv_counts,tp_recv_counts,expand_idx,assist};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_moe_distribute_dispatch_v2", PYBOOST_CALLER(7, custom::npu_moe_distribute_dispatch_v2));
}
}  // namespace custom
