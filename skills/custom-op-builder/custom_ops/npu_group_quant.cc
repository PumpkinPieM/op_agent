#include <vector>
#include <optional>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> GroupQuantOutShape(const ms::Tensor &x, int64_t dst_type) {
  auto shape = x.shape();
  if (dst_type == 3 || dst_type == 29) {
    shape.back() /= 8;
  }
  return shape;
}

ms::TypeId GroupQuantOutDtype(int64_t dst_type) {
  if (dst_type == 3 || dst_type == 29) {
    return ms::TypeId::kNumberTypeInt32;
  }
  return ms::TypeId::kNumberTypeInt8;
}
}  // namespace

ms::Tensor npu_group_quant(const ms::Tensor &x, const ms::Tensor &scale, const ms::Tensor &group_index,
                           const std::optional<ms::Tensor> &offset_opt = std::nullopt,
                           const std::optional<int64_t> &dst_dtype_opt = std::nullopt) {
  auto offset = offset_opt.value_or(ms::Tensor());
  int64_t dst_type = dst_dtype_opt.value_or(2);
  auto out = ms::Tensor(GroupQuantOutDtype(dst_type), GroupQuantOutShape(x, dst_type));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupQuant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupQuant, x, scale, group_index, offset_opt, dst_type, out));
  runner->Run({x, scale, group_index, offset}, {out});
  return out;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_group_quant", PYBOOST_CALLER(1, custom::npu_group_quant)); }
