#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
constexpr int64_t kTorchByte = 0;
constexpr int64_t kTorchChar = 1;
constexpr int64_t kTorchInt = 3;
constexpr int64_t kTorchQInt8 = 12;
constexpr int64_t kTorchQUInt8 = 13;
constexpr int64_t kTorchQInt32 = 14;
constexpr int64_t kTorchQUInt4x2 = 18;

int32_t AclTypeFromTorchDType(int64_t dtype) {
  switch (dtype) {
    case kTorchByte:
    case kTorchQUInt8:
      return 4;  // ACL_UINT8
    case kTorchChar:
    case kTorchQInt8:
      return 2;  // ACL_INT8
    case kTorchInt:
    case kTorchQInt32:
    case kTorchQUInt4x2:
      return 3;  // ACL_INT32
    default:
      return 2;
  }
}

ms::TypeId TypeIdFromTorchDType(int64_t dtype) {
  switch (dtype) {
    case kTorchByte:
    case kTorchQUInt8:
      return ms::TypeId::kNumberTypeUInt8;
    case kTorchInt:
    case kTorchQInt32:
    case kTorchQUInt4x2:
      return ms::TypeId::kNumberTypeInt32;
    case kTorchChar:
    case kTorchQInt8:
    default:
      return ms::TypeId::kNumberTypeInt8;
  }
}

std::vector<int64_t> OutputShape(const ms::Tensor &self, int64_t dtype, bool div_mode) {
  auto shape = self.shape();
  if (!div_mode && dtype == kTorchQUInt4x2 && !shape.empty()) {
    shape.back() /= 8;
  }
  return shape;
}
}  // namespace

ms::Tensor npu_quantize(const ms::Tensor &self, const ms::Tensor &scales,
                        const std::optional<ms::Tensor> &zero_points_opt, int64_t dtype, int64_t axis = 1,
                        bool div_mode = true) {
  auto zero_points = zero_points_opt.value_or(ms::Tensor());
  auto out = ms::Tensor(TypeIdFromTorchDType(dtype), OutputShape(self, dtype, div_mode));
  int32_t acl_dtype = AclTypeFromTorchDType(dtype);

  auto runner =
      std::make_shared<ms::pynative::AclnnOpRunner>(div_mode ? "Quantize" : "AscendQuantV3");
  if (div_mode) {
    int32_t axis_value = static_cast<int32_t>(axis);
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantize, self, scales, zero_points_opt, acl_dtype, axis_value, out));
  } else {
    constexpr bool sqrt_mode = false;
    const char *round_mode = "round";
    int32_t axis_value = static_cast<int32_t>(axis < -1 ? axis : -1);
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAscendQuantV3, self, scales, zero_points_opt, sqrt_mode, round_mode,
                                            acl_dtype, axis_value, out));
  }
  runner->Run({self, scales, zero_points}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quantize", PYBOOST_CALLER(1, custom::npu_quantize));
}
}  // namespace custom
