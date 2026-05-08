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

ms::Tensor npu_mm_all_reduce_base(const ms::Tensor & x1, const ms::Tensor & x2, const std::string & hcom, const std::optional<std::string> & reduce_op_opt, const std::optional<ms::Tensor> & bias_opt, const std::optional<ms::Tensor> & antiquant_scale_opt, const std::optional<ms::Tensor> & antiquant_offset_opt, const std::optional<ms::Tensor> & x3_opt, const std::optional<ms::Tensor> & dequant_scale_opt, const std::optional<ms::Tensor> & pertoken_scale_opt, const std::optional<ms::Tensor> & comm_quant_scale_1_opt, const std::optional<ms::Tensor> & comm_quant_scale_2_opt, int64_t antiquant_group_size, int64_t comm_turn, const std::optional<std::vector<int64_t>> & group_sizes_opt, const std::optional<int64_t> & y_dtype_opt, const std::optional<int64_t> & x1_dtype_opt, const std::optional<int64_t> & x2_dtype_opt, const std::optional<int64_t> & dequant_scale_dtype_opt, const std::optional<int64_t> & pertoken_scale_dtype_opt, int64_t comm_quant_mode) {
  auto out = ms::Tensor(DTypeFromOptional(y_dtype_opt.value_or(-1), x1.data_type()), MatmulShape(x1, x2));
  auto bias=bias_opt.value_or(ms::Tensor()); auto antiquant_scale=antiquant_scale_opt.value_or(ms::Tensor()); auto antiquant_offset=antiquant_offset_opt.value_or(ms::Tensor()); auto x3=x3_opt.value_or(ms::Tensor()); auto dequant_scale=dequant_scale_opt.value_or(ms::Tensor()); auto pertoken_scale=pertoken_scale_opt.value_or(ms::Tensor()); auto comm_quant_scale_1=comm_quant_scale_1_opt.value_or(ms::Tensor()); auto comm_quant_scale_2=comm_quant_scale_2_opt.value_or(ms::Tensor()); auto reduce_op=reduce_op_opt.value_or("sum"); auto group_sizes=std::make_pair(group_sizes_opt.value_or(std::vector<int64_t>{}), true);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MatmulAllReduce");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulAllReduce, x1, x2, hcom, reduce_op, bias_opt, antiquant_scale_opt, antiquant_offset_opt, x3_opt, dequant_scale_opt, pertoken_scale_opt, comm_quant_scale_1_opt, comm_quant_scale_2_opt, antiquant_group_size, comm_turn, group_sizes, y_dtype_opt.value_or(-1), comm_quant_mode, out));
  runner->Run({x1,x2,bias,antiquant_scale,antiquant_offset,x3,dequant_scale,pertoken_scale,comm_quant_scale_1,comm_quant_scale_2}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mm_all_reduce_base", PYBOOST_CALLER(1, custom::npu_mm_all_reduce_base));
}
}  // namespace custom
