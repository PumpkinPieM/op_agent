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

std::vector<ms::Tensor> npu_scatter_pa_kv_cache(const ms::Tensor & key, const ms::Tensor & value, const ms::Tensor & key_cache, const ms::Tensor & value_cache, const ms::Tensor & slot_mapping, const std::optional<ms::Tensor> & compress_lens_opt, const std::optional<ms::Tensor> & compress_seq_offsets_opt, const std::optional<ms::Tensor> & seq_lens_opt, const std::optional<std::string> & cache_mode_opt) {
  auto key_cache_out=ms::Tensor(key_cache.data_type(), key_cache.shape()); auto value_cache_out=ms::Tensor(value_cache.data_type(), value_cache.shape()); auto compress_lens=compress_lens_opt.value_or(ms::Tensor()); auto compress_seq_offsets=compress_seq_offsets_opt.value_or(ms::Tensor()); auto seq_lens=seq_lens_opt.value_or(ms::Tensor()); auto cache_mode=cache_mode_opt.value_or("PA_NZ");
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ScatterPaKvCache");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScatterPaKvCache, key, value, key_cache_out, value_cache_out, slot_mapping, compress_lens_opt, compress_seq_offsets_opt, seq_lens_opt, cache_mode));
  runner->Run({key,value,key_cache,value_cache,slot_mapping,compress_lens,compress_seq_offsets,seq_lens}, {key_cache_out,value_cache_out});
  return {key_cache_out,value_cache_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_scatter_pa_kv_cache", PYBOOST_CALLER(2, custom::npu_scatter_pa_kv_cache));
}
}  // namespace custom
