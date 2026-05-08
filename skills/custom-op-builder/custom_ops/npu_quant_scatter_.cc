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

ms::Tensor npu_quant_scatter_(const ms::Tensor & self, const ms::Tensor & indices, const ms::Tensor & updates, const ms::Tensor & quant_scales, const std::optional<ms::Tensor> & quant_zero_points_opt, int64_t axis, int64_t quant_axis, const std::string & reduce, const std::optional<int64_t> & dst_type_opt, const std::optional<std::string> & round_mode_opt) {
  auto out=ms::Tensor(DTypeFromOptional(dst_type_opt.value_or(-1),self.data_type()), self.shape()); auto quant_zero_points=quant_zero_points_opt.value_or(ms::Tensor());
  int64_t reduction = 1;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("InplaceQuantScatter");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnInplaceQuantScatter, out, indices, updates, quant_scales, quant_zero_points_opt, axis, quant_axis, reduction));
  runner->Run({self,indices,updates,quant_scales,quant_zero_points}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_scatter_", PYBOOST_CALLER(1, custom::npu_quant_scatter_));
}
}  // namespace custom
