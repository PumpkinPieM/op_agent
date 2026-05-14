#include <algorithm>
#include <optional>
#include <string>
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
}  // namespace

ms::Tensor npu_mm_reduce_scatter_base(const ms::Tensor & self, const ms::Tensor & x2, const std::string & hcom, int64_t world_size, const std::optional<std::string> & reduce_op_opt, const std::optional<ms::Tensor> & bias_opt, const std::optional<ms::Tensor> & x1_scale_opt, const std::optional<ms::Tensor> & x2_scale_opt, int64_t comm_turn, const std::optional<int64_t> & output_dtype_opt, const std::optional<std::string> & comm_mode_opt) {
  auto shape = MatmulShape(self, x2);
  if (!shape.empty()) {
    shape[0] = std::max<int64_t>(1, shape[0] / std::max<int64_t>(1, world_size));
  }
  const bool has_quant = x2_scale_opt.has_value();
  auto out_dtype = self.data_type();
  if (has_quant) {
    out_dtype = x2_scale_opt.value().data_type() == ms::TypeId::kNumberTypeInt64
                    ? ms::TypeId::kNumberTypeFloat16
                    : DTypeFromOptional(output_dtype_opt, ms::TypeId::kNumberTypeBFloat16);
  }
  auto out=ms::Tensor(out_dtype, shape);
  auto bias=bias_opt.value_or(ms::Tensor());
  auto x1_scale=x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale=x2_scale_opt.value_or(ms::Tensor());
  auto reduce_op=reduce_op_opt.value_or("sum");
  auto comm_mode=comm_mode_opt.value_or("ai_cpu");
  int64_t stream_mode = 1;
  if (comm_mode == "ai_cpu") {
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MatmulReduceScatter");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulReduceScatter, self, x2, bias_opt, hcom, reduce_op,
                                            comm_turn, stream_mode, out));
    runner->Run({self,x2,bias}, {out});
  } else {
    std::optional<ms::Tensor> quant_scale_opt = std::nullopt;
    auto quant_scale = quant_scale_opt.value_or(ms::Tensor());
    int64_t block_size = 0;
    int64_t group_size = 0;
    std::optional<ms::Tensor> amax_out_opt = std::nullopt;
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MatmulReduceScatterV2");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulReduceScatterV2, self, x2, bias_opt, x1_scale_opt,
                                            x2_scale_opt, quant_scale_opt, block_size, hcom, reduce_op, comm_turn,
                                            stream_mode, group_size, comm_mode, out, amax_out_opt));
    runner->Run({self,x2,bias,x1_scale,x2_scale,quant_scale}, {out});
  }
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mm_reduce_scatter_base", PYBOOST_CALLER(1, custom::npu_mm_reduce_scatter_base));
}
}  // namespace custom
