#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {

std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &x, int64_t dst_type) {
  auto y_shape = x.shape();
  y_shape.back() = dst_type == 29 ? y_shape.back() / 4 : y_shape.back() / 2;
  auto scale_shape = x.shape();
  scale_shape.pop_back();
  auto y = ms::Tensor(ms::TypeId::kNumberTypeInt8, y_shape);
  auto scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, scale_shape);
  return std::make_tuple(std::move(y), std::move(scale));
}

}  // namespace

std::vector<ms::Tensor> npu_swiglu_quant(const ms::Tensor &x,
                                         const std::optional<ms::Tensor> &smooth_scales_opt = std::nullopt,
                                         const std::optional<ms::Tensor> &offsets_opt = std::nullopt,
                                         const std::optional<ms::Tensor> &group_index_opt = std::nullopt,
                                         bool activate_left = false, int64_t quant_mode = 0,
                                         int64_t group_list_type = 0,
                                         const std::optional<int64_t> &dst_type_opt = std::nullopt) {
  auto smooth_scales = smooth_scales_opt.value_or(ms::Tensor());
  auto offsets = offsets_opt.value_or(ms::Tensor());
  auto group_index = group_index_opt.value_or(ms::Tensor());
  auto dst_type = dst_type_opt.value_or(2);
  auto quant_mode_value = quant_mode == 0 ? std::string("static") : std::string("dynamic");
  auto quant_mode_ptr = const_cast<char *>(quant_mode_value.c_str());
  auto [y, scale] = GenResultTensors(x, dst_type);

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("SwiGluQuantV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnSwiGluQuantV2, x, smooth_scales_opt, offsets_opt, group_index_opt,
                                          activate_left, quant_mode_ptr, group_list_type, dst_type, y, scale));
  runner->Run({x, smooth_scales, offsets, group_index}, {y, scale});
  return {y, scale};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_swiglu_quant", PYBOOST_CALLER(2, custom::npu_swiglu_quant));
}
}  // namespace custom
