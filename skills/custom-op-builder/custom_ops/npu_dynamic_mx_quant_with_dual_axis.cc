#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

constexpr int64_t kTorchFloat8E5M2 = 23;
constexpr int64_t kTorchFloat8E4M3FN = 24;
constexpr int64_t kTorchNpuFloat4E2M1 = 296;
constexpr int64_t kTorchNpuFloat4E1M2 = 297;
constexpr int64_t kAclFloat8E5M2 = 35;
constexpr int64_t kAclFloat8E4M3FN = 36;
constexpr int64_t kAclFloat4E2M1 = 40;
constexpr int64_t kAclFloat4E1M2 = 41;

int64_t ToAclDstType(int64_t dst_type) {
  switch (dst_type) {
    case kTorchFloat8E5M2:
      return kAclFloat8E5M2;
    case kTorchFloat8E4M3FN:
      return kAclFloat8E4M3FN;
    case kTorchNpuFloat4E2M1:
      return kAclFloat4E2M1;
    case kTorchNpuFloat4E1M2:
      return kAclFloat4E1M2;
    default:
      return dst_type;
  }
}

ms::TypeId OutputDType(int64_t acl_dst_type) {
  if (acl_dst_type == kAclFloat8E5M2) {
    return ms::TypeId::kNumberTypeFloat8E5M2;
  }
  if (acl_dst_type == kAclFloat8E4M3FN) {
    return ms::TypeId::kNumberTypeFloat8E4M3FN;
  }
  return ms::TypeId::kNumberTypeUInt8;
}

std::vector<int64_t> OutputShape(const ms::Tensor &input, int64_t acl_dst_type) {
  auto shape = input.shape();
  if ((acl_dst_type == kAclFloat4E2M1 || acl_dst_type == kAclFloat4E1M2) && !shape.empty()) {
    shape.back() /= 2;
  }
  return shape;
}

std::vector<int64_t> MxScale1Shape(const ms::Tensor &input) {
  auto shape = input.shape();
  shape.push_back(2);
  shape[shape.size() - 2] = CeilDiv(CeilDiv(shape[shape.size() - 2], 32), 2);
  return shape;
}

std::vector<int64_t> MxScale2Shape(const ms::Tensor &input) {
  auto shape = input.shape();
  shape.push_back(2);
  shape[shape.size() - 3] = CeilDiv(CeilDiv(shape[shape.size() - 3], 32), 2);
  return shape;
}
}  // namespace

std::vector<ms::Tensor> npu_dynamic_mx_quant_with_dual_axis(const ms::Tensor &input,
    const std::string &round_mode = "rint", int64_t dst_type = 296, int64_t scale_alg = 0) {
  auto acl_dst_type = ToAclDstType(dst_type);
  auto output_dtype = OutputDType(acl_dst_type);
  auto output_shape = OutputShape(input, acl_dst_type);
  auto y1 = ms::Tensor(output_dtype, output_shape);
  auto y2 = ms::Tensor(output_dtype, output_shape);
  auto m1 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, MxScale1Shape(input));
  auto m2 = ms::Tensor(ms::TypeId::kNumberTypeUInt8, MxScale2Shape(input));
  auto round = const_cast<char *>(round_mode.c_str());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("DynamicMxQuantWithDualAxis");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDynamicMxQuantWithDualAxis, input, round, acl_dst_type, scale_alg, y1, m1, y2, m2));
  runner->Run({input}, {y1, m1, y2, m2}); return {y1, m1, y2, m2};
}
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_dynamic_mx_quant_with_dual_axis", PYBOOST_CALLER(4, custom::npu_dynamic_mx_quant_with_dual_axis)); }

}  // namespace custom
