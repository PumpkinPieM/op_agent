#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kDim3 = 3;
constexpr int64_t kMode1 = 1;
constexpr int64_t kMode2 = 2;
constexpr int64_t kMode3 = 3;
constexpr int64_t kMode4 = 4;
constexpr int64_t kMode5 = 5;
constexpr int64_t kFp8E4m3BlockSize = 32;

bool IsQuantQueryPath(int64_t weight_quant_mode, int64_t kv_cache_quant_mode) {
  return (weight_quant_mode == kMode2 || weight_quant_mode == kMode3 || weight_quant_mode == kMode4 ||
          weight_quant_mode == kMode5) &&
         kv_cache_quant_mode == kMode1;
}

int64_t TokenCount(const std::vector<int64_t> &token_shape) {
  return token_shape.size() == kDim3 ? token_shape[0] * token_shape[1] : token_shape[0];
}

std::vector<int64_t> QueryShape(const ms::Tensor &token_x, const ms::Tensor &weight_uk) {
  const auto token_shape = token_x.shape();
  const auto weight_uk_shape = weight_uk.shape();
  if (token_shape.size() == kDim3) {
    return {token_shape[0], token_shape[1], weight_uk_shape[0], weight_uk_shape[2]};
  }
  return {token_shape[0], weight_uk_shape[0], weight_uk_shape[2]};
}

std::vector<int64_t> QueryRopeShape(const ms::Tensor &token_x, const ms::Tensor &weight_uk, const ms::Tensor &rope_sin) {
  const auto token_shape = token_x.shape();
  const auto weight_uk_shape = weight_uk.shape();
  const auto rope_shape = rope_sin.shape();
  if (token_shape.size() == kDim3) {
    return {token_shape[0], token_shape[1], weight_uk_shape[0], rope_shape[2]};
  }
  return {token_shape[0], weight_uk_shape[0], rope_shape[1]};
}

std::vector<int64_t> QueryNormShape(const ms::Tensor &token_x, const ms::Tensor &weight_dq) {
  const auto token_shape = token_x.shape();
  if (token_shape.size() == kDim3) {
    return {token_shape[0], token_shape[1], weight_dq.shape()[1]};
  }
  return {token_shape[0], weight_dq.shape()[1]};
}

std::vector<int64_t> DequantScaleQNopeShape(const ms::Tensor &token_x, const ms::Tensor &weight_uk,
                                            bool quant_query_path) {
  if (!quant_query_path) {
    return {0};
  }
  return {TokenCount(token_x.shape()), weight_uk.shape()[0], 1};
}

std::vector<int64_t> DequantScaleQNormShape(const ms::Tensor &token_x, const ms::Tensor &weight_dq,
                                            int64_t weight_quant_mode) {
  const int64_t token_count = TokenCount(token_x.shape());
  if (weight_quant_mode == kMode3) {
    return {token_count, weight_dq.shape()[1] / kFp8E4m3BlockSize};
  }
  return {token_count, 1};
}

void SetNzStorage(const ms::Tensor &tensor) {
  const std::string nz_format = "FRACTAL_NZ";
  tensor.set_format(nz_format);
  auto nd_shape = tensor.shape();
  auto nz_shape =
      mindspore::trans::DeviceShapeTransfer().GetDeviceShapeByFormat(nd_shape, nz_format, tensor.data_type());

  constexpr int64_t kStrideBase = 1;
  constexpr int kStrideOffset = 2;
  auto strides = nd_shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(kStrideBase);
  for (int i = static_cast<int>(strides.size()) - kStrideOffset; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  auto storage_info = std::make_shared<mindspore::TensorStorageInfo>(nd_shape, strides, nz_shape, strides, true);
  MS_EXCEPTION_IF_NULL(tensor.tensor());
  MS_EXCEPTION_IF_NULL(tensor.tensor()->device_address());
  tensor.tensor()->set_storage_info(storage_info);
}
}  // namespace

std::vector<ms::Tensor> npu_mla_prolog_v3(
    const ms::Tensor &token_x, const ms::Tensor &weight_dq, const ms::Tensor &weight_uq_qr,
    const ms::Tensor &weight_uk, const ms::Tensor &weight_dkv_kr, const ms::Tensor &rmsnorm_gamma_cq,
    const ms::Tensor &rmsnorm_gamma_ckv, const ms::Tensor &rope_sin, const ms::Tensor &rope_cos,
    const ms::Tensor &kv_cache, const ms::Tensor &kr_cache, const std::optional<ms::Tensor> &cache_index_opt,
    const std::optional<ms::Tensor> &dequant_scale_x_opt,
    const std::optional<ms::Tensor> &dequant_scale_w_dq_opt,
    const std::optional<ms::Tensor> &dequant_scale_w_uq_qr_opt,
    const std::optional<ms::Tensor> &dequant_scale_w_dkv_kr_opt,
    const std::optional<ms::Tensor> &quant_scale_ckv_opt,
    const std::optional<ms::Tensor> &quant_scale_ckr_opt,
    const std::optional<ms::Tensor> &smooth_scales_cq_opt,
    const std::optional<ms::Tensor> &actual_seq_len_opt,
    const std::optional<ms::Tensor> &k_nope_clip_alpha_opt, double rmsnorm_epsilon_cq,
    double rmsnorm_epsilon_ckv, const std::optional<std::string> &cache_mode_opt, bool query_norm_flag,
    int64_t weight_quant_mode, int64_t kv_cache_quant_mode, int64_t query_quant_mode, int64_t ckvkr_repo_mode,
    int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale,
    const std::optional<int64_t> &token_x_dtype_opt, const std::optional<int64_t> &weight_dq_dtype_opt,
    const std::optional<int64_t> &weight_uq_qr_dtype_opt, const std::optional<int64_t> &weight_dkv_kr_dtype_opt,
    const std::optional<int64_t> &kv_cache_dtype_opt) {
  const bool quant_query_path = IsQuantQueryPath(weight_quant_mode, kv_cache_quant_mode);
  auto query = ms::Tensor(quant_query_path ? token_x.data_type() : ms::TypeId::kNumberTypeBFloat16,
                          QueryShape(token_x, weight_uk));
  auto query_rope = ms::Tensor(ms::TypeId::kNumberTypeBFloat16, QueryRopeShape(token_x, weight_uk, rope_sin));
  auto dequant_scale_q_nope =
      ms::Tensor(ms::TypeId::kNumberTypeFloat32, DequantScaleQNopeShape(token_x, weight_uk, quant_query_path));
  auto query_norm = ms::Tensor(weight_uq_qr.data_type(), query_norm_flag ? QueryNormShape(token_x, weight_dq)
                                                                         : std::vector<int64_t>{0});
  const bool need_dequant_scale_q_norm =
      query_norm_flag &&
      (weight_quant_mode == kMode1 || weight_quant_mode == kMode2 || weight_quant_mode == kMode3 ||
       weight_quant_mode == kMode4 || weight_quant_mode == kMode5);
  auto dequant_scale_q_norm =
      ms::Tensor(ms::TypeId::kNumberTypeFloat32,
                 need_dequant_scale_q_norm ? DequantScaleQNormShape(token_x, weight_dq, weight_quant_mode)
                                           : std::vector<int64_t>{0});
  std::optional<ms::Tensor> dequant_scale_q_nope_out_opt =
      quant_query_path ? std::optional<ms::Tensor>(dequant_scale_q_nope) : std::nullopt;
  std::optional<ms::Tensor> query_norm_out_opt =
      query_norm_flag ? std::optional<ms::Tensor>(query_norm) : std::nullopt;
  std::optional<ms::Tensor> dequant_scale_q_norm_out_opt =
      need_dequant_scale_q_norm ? std::optional<ms::Tensor>(dequant_scale_q_norm) : std::nullopt;

  auto cache_index = cache_index_opt.value_or(ms::Tensor());
  auto dequant_scale_x = dequant_scale_x_opt.value_or(ms::Tensor());
  auto dequant_scale_w_dq = dequant_scale_w_dq_opt.value_or(ms::Tensor());
  auto dequant_scale_w_uq_qr = dequant_scale_w_uq_qr_opt.value_or(ms::Tensor());
  auto dequant_scale_w_dkv_kr = dequant_scale_w_dkv_kr_opt.value_or(ms::Tensor());
  auto quant_scale_ckv = quant_scale_ckv_opt.value_or(ms::Tensor());
  auto quant_scale_ckr = quant_scale_ckr_opt.value_or(ms::Tensor());
  auto smooth_scales_cq = smooth_scales_cq_opt.value_or(ms::Tensor());
  auto actual_seq_len = actual_seq_len_opt.value_or(ms::Tensor());
  auto k_nope_clip_alpha = k_nope_clip_alpha_opt.value_or(ms::Tensor());
  auto cache_mode = cache_mode_opt.value_or("PA_BSND");

  SetNzStorage(weight_dq);
  SetNzStorage(weight_uq_qr);
  SetNzStorage(weight_dkv_kr);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MlaPrologV3WeightNz");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
      aclnnMlaPrologV3WeightNz, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
      rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index_opt, dequant_scale_x_opt,
      dequant_scale_w_dq_opt, dequant_scale_w_uq_qr_opt, dequant_scale_w_dkv_kr_opt, quant_scale_ckv_opt,
      quant_scale_ckr_opt, smooth_scales_cq_opt, actual_seq_len_opt, k_nope_clip_alpha_opt, rmsnorm_epsilon_cq,
      rmsnorm_epsilon_ckv, cache_mode, weight_quant_mode, kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode,
      quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query, query_rope, dequant_scale_q_nope_out_opt,
      query_norm_out_opt, dequant_scale_q_norm_out_opt));
  runner->Run({token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
               rope_sin, rope_cos, kv_cache, kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq,
               dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq,
               actual_seq_len, k_nope_clip_alpha},
              {query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache, kr_cache});
  return {query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mla_prolog_v3", PYBOOST_CALLER(5, custom::npu_mla_prolog_v3));
}
}  // namespace custom
