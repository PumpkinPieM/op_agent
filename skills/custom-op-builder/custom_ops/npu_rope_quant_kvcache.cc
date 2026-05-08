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

std::vector<ms::Tensor> npu_rope_quant_kvcache(const ms::Tensor & x, const ms::Tensor & cos, const ms::Tensor & sin, const ms::Tensor & k_cache, const ms::Tensor & v_cache, const ms::Tensor & indices, const ms::Tensor & scale_k, const ms::Tensor & scale_v, const std::vector<int64_t> & size_splits, const std::optional<ms::Tensor> & offset_k_opt, const std::optional<ms::Tensor> & offset_v_opt, int64_t quant_mode, const std::optional<std::string> & input_layout_opt, bool kv_output, const std::optional<std::string> & cache_mode_opt) {
  auto q=ms::Tensor(x.data_type(), x.shape()); auto k=ms::Tensor(k_cache.data_type(), kv_output ? k_cache.shape() : std::vector<int64_t>{}); auto v=ms::Tensor(v_cache.data_type(), kv_output ? v_cache.shape() : std::vector<int64_t>{}); auto k_cache_out=ms::Tensor(k_cache.data_type(), k_cache.shape()); auto v_cache_out=ms::Tensor(v_cache.data_type(), v_cache.shape()); auto offset_k=offset_k_opt.value_or(ms::Tensor()); auto offset_v=offset_v_opt.value_or(ms::Tensor()); auto input_layout=input_layout_opt.value_or("BSND"); auto cache_mode=cache_mode_opt.value_or("contiguous");
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DequantRopeQuantKvcache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDequantRopeQuantKvcache, x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, offset_k_opt, offset_v_opt, quant_mode, input_layout, kv_output, cache_mode, q, k, v, k_cache_out, v_cache_out));
  runner->Run({x,cos,sin,k_cache,v_cache,indices,scale_k,scale_v,offset_k,offset_v}, {q,k,v,k_cache_out,v_cache_out});
  return {q,k,v,k_cache_out,v_cache_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_rope_quant_kvcache", PYBOOST_CALLER(5, custom::npu_rope_quant_kvcache));
}
}  // namespace custom
