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

std::vector<ms::Tensor> npu_mla_prolog_v3_functional(const ms::Tensor & token_x, const ms::Tensor & weight_dq, const ms::Tensor & weight_uq_qr, const ms::Tensor & weight_uk, const ms::Tensor & weight_dkv_kr, const ms::Tensor & rmsnorm_gamma_cq, const ms::Tensor & rmsnorm_gamma_ckv, const ms::Tensor & rope_sin, const ms::Tensor & rope_cos, const ms::Tensor & kv_cache, const ms::Tensor & kr_cache, const std::optional<ms::Tensor> & cache_index_opt, const std::optional<ms::Tensor> & dequant_scale_x_opt, const std::optional<ms::Tensor> & dequant_scale_w_dq_opt, const std::optional<ms::Tensor> & dequant_scale_w_uq_qr_opt, const std::optional<ms::Tensor> & dequant_scale_w_dkv_kr_opt, const std::optional<ms::Tensor> & quant_scale_ckv_opt, const std::optional<ms::Tensor> & quant_scale_ckr_opt, const std::optional<ms::Tensor> & smooth_scales_cq_opt, const std::optional<ms::Tensor> & actual_seq_len_opt, const std::optional<ms::Tensor> & k_nope_clip_alpha_opt, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, const std::optional<std::string> & cache_mode_opt, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode, int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale, const std::optional<int64_t> & token_x_dtype_opt, const std::optional<int64_t> & weight_dq_dtype_opt, const std::optional<int64_t> & weight_uq_qr_dtype_opt, const std::optional<int64_t> & weight_dkv_kr_dtype_opt, const std::optional<int64_t> & kv_cache_dtype_opt) {
  auto q = ms::Tensor(token_x.data_type(), std::vector<int64_t>{token_x.shape()[0], weight_uq_qr.shape()[0]});
  auto kv = ms::Tensor(token_x.data_type(), std::vector<int64_t>{token_x.shape()[0], weight_uk.shape()[0]});
  auto kr = ms::Tensor(token_x.data_type(), std::vector<int64_t>{token_x.shape()[0], weight_uk.shape()[0]});
  auto cache_index = cache_index_opt.value_or(ms::Tensor()); auto dequant_scale_x = dequant_scale_x_opt.value_or(ms::Tensor()); auto dequant_scale_w_dq = dequant_scale_w_dq_opt.value_or(ms::Tensor()); auto dequant_scale_w_uq_qr = dequant_scale_w_uq_qr_opt.value_or(ms::Tensor()); auto dequant_scale_w_dkv_kr = dequant_scale_w_dkv_kr_opt.value_or(ms::Tensor()); auto quant_scale_ckv = quant_scale_ckv_opt.value_or(ms::Tensor()); auto quant_scale_ckr = quant_scale_ckr_opt.value_or(ms::Tensor()); auto smooth_scales_cq = smooth_scales_cq_opt.value_or(ms::Tensor()); auto actual_seq_len = actual_seq_len_opt.value_or(ms::Tensor()); auto k_nope_clip_alpha = k_nope_clip_alpha_opt.value_or(ms::Tensor());
  auto cache_mode = cache_mode_opt.value_or("PA_BSND");
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MlaPrologV3WeightNz");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMlaPrologV3WeightNz, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index_opt, dequant_scale_x_opt, dequant_scale_w_dq_opt, dequant_scale_w_uq_qr_opt, dequant_scale_w_dkv_kr_opt, quant_scale_ckv_opt, quant_scale_ckr_opt, smooth_scales_cq_opt, actual_seq_len_opt, k_nope_clip_alpha_opt, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, token_x_dtype_opt.value_or(-1), weight_dq_dtype_opt.value_or(-1), weight_uq_qr_dtype_opt.value_or(-1), weight_dkv_kr_dtype_opt.value_or(-1), kv_cache_dtype_opt.value_or(-1), q, kv, kr, kv_cache, kr_cache));
  runner->Run({token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len, k_nope_clip_alpha}, {q, kv, kr, kv_cache, kr_cache});
  return {q, kv, kr, kv_cache, kr_cache, kv_cache, kr_cache};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mla_prolog_v3_functional", PYBOOST_CALLER(7, custom::npu_mla_prolog_v3_functional));
}
}  // namespace custom
