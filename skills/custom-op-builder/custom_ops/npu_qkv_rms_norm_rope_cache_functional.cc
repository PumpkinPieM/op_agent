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

std::vector<ms::Tensor> npu_qkv_rms_norm_rope_cache_functional(const ms::Tensor & qkv, const ms::Tensor & q_gamma, const ms::Tensor & k_gamma, const ms::Tensor & cos, const ms::Tensor & sin, const ms::Tensor & index, const ms::Tensor & q_out, const ms::Tensor & k_cache, const ms::Tensor & v_cache, const std::vector<int64_t> & qkv_size, const std::vector<int64_t> & head_nums, const std::optional<ms::Tensor> & k_scale_opt, const std::optional<ms::Tensor> & v_scale_opt, const std::optional<ms::Tensor> & k_offset_opt, const std::optional<ms::Tensor> & v_offset_opt, double epsilon, const std::optional<std::string> & cache_mode_opt, bool is_output_qkv) {
  auto q=ms::Tensor(qkv.data_type(), is_output_qkv ? std::vector<int64_t>{qkv.shape()[0], head_nums[0]*qkv_size[3]} : std::vector<int64_t>{}); auto k=ms::Tensor(qkv.data_type(), is_output_qkv ? std::vector<int64_t>{qkv.shape()[0], head_nums[1]*qkv_size[3]} : std::vector<int64_t>{}); auto v=ms::Tensor(qkv.data_type(), is_output_qkv ? std::vector<int64_t>{qkv.shape()[0], head_nums[2]*qkv_size[3]} : std::vector<int64_t>{}); auto k_scale=k_scale_opt.value_or(ms::Tensor()); auto v_scale=v_scale_opt.value_or(ms::Tensor()); auto k_offset=k_offset_opt.value_or(ms::Tensor()); auto v_offset=v_offset_opt.value_or(ms::Tensor()); auto cache_mode=cache_mode_opt.value_or("PA_NZ");
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QkvRmsNormRopeCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQkvRmsNormRopeCache, qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, qkv_size, head_nums, k_scale_opt, v_scale_opt, k_offset_opt, v_offset_opt, epsilon, cache_mode, is_output_qkv, q, k, v));
  runner->Run({qkv,q_gamma,k_gamma,cos,sin,index,q_out,k_cache,v_cache,k_scale,v_scale,k_offset,v_offset}, {q,k,v});
  return {q,k,v,q_out,k_cache,v_cache};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_qkv_rms_norm_rope_cache_functional", PYBOOST_CALLER(6, custom::npu_qkv_rms_norm_rope_cache_functional));
}
}  // namespace custom
