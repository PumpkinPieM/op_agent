#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> ScaleShape(const ms::Tensor &input, const std::string &quant_mode) {
  const auto &shape = input.shape();
  if (quant_mode == "pertensor") {
    return {1};
  }
  std::vector<int64_t> out;
  if (quant_mode == "perchannel") {
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
      out.push_back(shape[i]);
    }
    out.push_back(shape.back());
  } else {
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      out.push_back(shape[i]);
    }
  }
  return out;
}
}  // namespace

std::vector<ms::Tensor> npu_dynamic_quant(const ms::Tensor &input,
                                          const std::optional<ms::Tensor> &smooth_scales_opt = std::nullopt,
                                          const std::optional<ms::Tensor> &group_index_opt = std::nullopt,
                                          const std::optional<int64_t> &dst_type_opt = std::nullopt,
                                          const std::string &quant_mode = "pertoken",
                                          double dst_type_max = 0.0) {
  auto smooth_scales = smooth_scales_opt.value_or(ms::Tensor());
  auto group_index = group_index_opt.value_or(ms::Tensor());
  auto dst_type = dst_type_opt.value_or(2);
  auto y = ms::Tensor(ms::TypeId::kNumberTypeInt8, input.shape());
  auto scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, ScaleShape(input, quant_mode));
  auto quant_mode_value = const_cast<char *>(quant_mode.c_str());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DynamicQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDynamicQuant, input, smooth_scales_opt, y, scale));
  runner->Run({input, smooth_scales, group_index}, {y, scale});
  (void)quant_mode_value;
  (void)dst_type_max;
  return {y, scale};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_dynamic_quant", PYBOOST_CALLER(2, custom::npu_dynamic_quant));
}
}  // namespace custom
