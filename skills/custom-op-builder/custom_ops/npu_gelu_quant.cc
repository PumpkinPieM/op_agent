#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> DynamicScaleShape(const ms::Tensor &x) {
  std::vector<int64_t> out;
  const auto &shape = x.shape();
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    out.push_back(shape[i]);
  }
  return out;
}
}  // namespace

std::vector<ms::Tensor> npu_gelu_quant(const ms::Tensor &self,
                                       const std::optional<ms::Tensor> &input_scale_opt = std::nullopt,
                                       const std::optional<ms::Tensor> &input_offset_opt = std::nullopt,
                                       const std::string &approximate = "none",
                                       const std::string &quant_mode = "dynamic",
                                       const std::optional<int64_t> &dst_type_opt = std::nullopt,
                                       const std::string &round_mode = "rint") {
  auto input_scale = input_scale_opt.value_or(ms::Tensor());
  auto input_offset = input_offset_opt.value_or(ms::Tensor());
  auto y = ms::Tensor(ms::TypeId::kNumberTypeInt8, self.shape());
  auto out_scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, DynamicScaleShape(self));
  auto approximate_value = const_cast<char *>(approximate.c_str());
  auto quant_mode_value = const_cast<char *>(quant_mode.c_str());
  auto round_mode_value = const_cast<char *>(round_mode.c_str());
  auto dst_type = dst_type_opt.value_or(2);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GeluQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGeluQuant, self, input_scale_opt, input_offset_opt, approximate_value,
                                          quant_mode_value, round_mode_value, dst_type, y, out_scale));
  runner->Run({self, input_scale, input_offset}, {y, out_scale});
  return {y, out_scale};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_gelu_quant", PYBOOST_CALLER(2, custom::npu_gelu_quant));
}
}  // namespace custom
