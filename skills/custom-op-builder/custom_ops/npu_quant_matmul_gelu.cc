#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> MatmulShape(const ms::Tensor &x1, const ms::Tensor &x2) {
  auto s1 = x1.shape();
  auto s2 = x2.shape();
  if (s1.size() < 2 || s2.size() < 2) {
    return s1;
  }
  std::vector<int64_t> out;
  if (s1.size() > 2) {
    out.insert(out.end(), s1.begin(), s1.end() - 2);
  }
  out.push_back(s1[s1.size() - 2]);
  out.push_back(s2.back());
  return out;
}

ms::TypeId OutputDType(const ms::Tensor &x2_scale, const std::optional<ms::Tensor> &bias_opt) {
  if (x2_scale.data_type() == ms::TypeId::kNumberTypeBFloat16 ||
      (bias_opt.has_value() && bias_opt.value().data_type() == ms::TypeId::kNumberTypeBFloat16)) {
    return ms::TypeId::kNumberTypeBFloat16;
  }
  return ms::TypeId::kNumberTypeFloat16;
}
}  // namespace

ms::Tensor npu_quant_matmul_gelu(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &x1_scale,
                                 const ms::Tensor &x2_scale, const std::optional<ms::Tensor> &bias_opt,
                                 const std::optional<std::string> &approximate_opt) {
  auto out = ms::Tensor(OutputDType(x2_scale, bias_opt), MatmulShape(x1, x2));
  auto bias = bias_opt.value_or(ms::Tensor());
  auto approximate = approximate_opt.value_or("gelu_erf");
  std::optional<ms::Tensor> empty_tensor_opt = std::nullopt;
  int64_t group_size = 0;

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("FusedQuantMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedQuantMatmul, x1, x2, x1_scale, x2_scale, empty_tensor_opt,
                                          empty_tensor_opt, empty_tensor_opt, empty_tensor_opt, bias_opt,
                                          empty_tensor_opt, approximate, group_size, out));
  runner->Run({x1, x2, x1_scale, x2_scale, bias}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_matmul_gelu", PYBOOST_CALLER(1, custom::npu_quant_matmul_gelu));
}
}  // namespace custom
