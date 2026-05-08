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

ms::Tensor npu_quant_matmul_all_to_all(const ms::Tensor & x1, const ms::Tensor & x2, const std::string & hcom, int64_t world_size, const std::optional<ms::Tensor> & bias_opt, const std::optional<ms::Tensor> & x1_scale_opt, const std::optional<ms::Tensor> & x2_scale_opt, const std::optional<ms::Tensor> & common_scale_opt, const std::optional<ms::Tensor> & x1_offset_opt, const std::optional<ms::Tensor> & x2_offset_opt, const std::optional<int64_t> & x1_quant_mode_opt, const std::optional<int64_t> & x2_quant_mode_opt, const std::optional<int64_t> & common_quant_mode_opt, const std::optional<std::vector<int64_t>> & group_sizes_opt, const std::optional<std::vector<int64_t>> & all2all_axes_opt, const std::optional<int64_t> & comm_quant_dtype_opt, const std::optional<int64_t> & x1_dtype_opt, const std::optional<int64_t> & x2_dtype_opt, const std::optional<int64_t> & x1_scale_dtype_opt, const std::optional<int64_t> & x2_scale_dtype_opt, const std::optional<int64_t> & output_scale_dtype_opt, const std::optional<int64_t> & comm_scale_dtype_opt, const std::optional<int64_t> & y_dtype_opt) {
  auto out=ms::Tensor(DTypeFromOptional(y_dtype_opt.value_or(-1), ms::TypeId::kNumberTypeFloat16), MatmulShape(x1,x2)); auto bias=bias_opt.value_or(ms::Tensor()); auto x1_scale=x1_scale_opt.value_or(ms::Tensor()); auto x2_scale=x2_scale_opt.value_or(ms::Tensor()); auto common_scale=common_scale_opt.value_or(ms::Tensor()); auto x1_offset=x1_offset_opt.value_or(ms::Tensor()); auto x2_offset=x2_offset_opt.value_or(ms::Tensor()); auto group_sizes=std::make_pair(group_sizes_opt.value_or(std::vector<int64_t>{}), true); auto all2all_axes=std::make_pair(all2all_axes_opt.value_or(std::vector<int64_t>{}), true);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulAlltoAll");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulAlltoAll, x1, x2, hcom, world_size, bias_opt, x1_scale_opt, x2_scale_opt, common_scale_opt, x1_offset_opt, x2_offset_opt, x1_quant_mode_opt.value_or(0), x2_quant_mode_opt.value_or(0), common_quant_mode_opt.value_or(0), group_sizes, all2all_axes, comm_quant_dtype_opt.value_or(-1), x1_dtype_opt.value_or(-1), x2_dtype_opt.value_or(-1), x1_scale_dtype_opt.value_or(-1), x2_scale_dtype_opt.value_or(-1), output_scale_dtype_opt.value_or(-1), comm_scale_dtype_opt.value_or(-1), y_dtype_opt.value_or(-1), out));
  runner->Run({x1,x2,bias,x1_scale,x2_scale,common_scale,x1_offset,x2_offset}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_matmul_all_to_all", PYBOOST_CALLER(1, custom::npu_quant_matmul_all_to_all));
}
}  // namespace custom
