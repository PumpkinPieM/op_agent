#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
ms::TypeId DTypeFromAclType(const std::optional<int64_t> &dtype, ms::TypeId fallback) {
  if (!dtype.has_value() || dtype.value() < 0) {
    return fallback;
  }
  switch (dtype.value()) {
    case 5:
      return ms::TypeId::kNumberTypeFloat16;
    case 6:
      return ms::TypeId::kNumberTypeFloat32;
    case 27:
      return ms::TypeId::kNumberTypeBFloat16;
    default:
      return fallback;
  }
}

std::vector<int64_t> MatmulShape(const ms::Tensor &self, const ms::Tensor &x2, int64_t world_size,
                                 int64_t gather_index) {
  const auto &self_shape = self.shape();
  const auto &x2_shape = x2.shape();
  int64_t rows = gather_index == 0 ? self_shape[0] * world_size : self_shape[0];
  return {rows, x2_shape[1]};
}

std::vector<int64_t> GatherShape(const ms::Tensor &self, const ms::Tensor &x2, int64_t world_size,
                                 int64_t gather_index, bool gather_output) {
  if (!gather_output) {
    return {0};
  }
  const auto &self_shape = self.shape();
  const auto &x2_shape = x2.shape();
  if (gather_index == 0) {
    return {self_shape[0] * world_size, self_shape[1]};
  }
  return {x2_shape[0] * world_size, x2_shape[1]};
}
}  // namespace

std::vector<ms::Tensor> npu_all_gather_base_mm(const ms::Tensor &self, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt, int64_t gather_index, bool gather_output, int64_t comm_turn, const std::optional<int64_t> &output_dtype_opt, const std::optional<std::string> &comm_mode_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto x1_scale_value = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale_value = x2_scale_opt.value_or(ms::Tensor());
  auto comm_mode = comm_mode_opt.value_or("ai_cpu");
  int64_t stream_mode = 1;
  const bool has_quant = x2_scale_opt.has_value();
  auto out_dtype = self.data_type();
  if (has_quant) {
    out_dtype = x2_scale_value.data_type() == ms::TypeId::kNumberTypeInt64
                    ? ms::TypeId::kNumberTypeFloat16
                    : DTypeFromAclType(output_dtype_opt, ms::TypeId::kNumberTypeBFloat16);
  }
  auto out0 = ms::Tensor(out_dtype, MatmulShape(self, x2, world_size, gather_index));
  auto out1 = ms::Tensor(self.data_type(), GatherShape(self, x2, world_size, gather_index, gather_output));

  if (comm_mode == "ai_cpu") {
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AllGatherMatmul");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAllGatherMatmul, self, x2, bias_opt, hcom, gather_index, comm_turn,
                                            stream_mode, out0, out1));
    runner->Run({self, x2, bias_value, x1_scale_value, x2_scale_value}, {out0, out1});
    return {out0, out1};
  }

  std::optional<ms::Tensor> quant_scale_opt = std::nullopt;
  auto quant_scale_value = ms::Tensor();
  int64_t block_size = 0;
  int64_t group_size = 0;
  auto amax = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{0});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AllGatherMatmulV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAllGatherMatmulV2, self, x2, bias_opt, x1_scale_opt, x2_scale_opt,
                                          quant_scale_opt, block_size, hcom, gather_index, comm_turn, stream_mode,
                                          group_size, comm_mode, out0, out1, amax));
  runner->Run({self, x2, bias_value, x1_scale_value, x2_scale_value, quant_scale_value}, {out0, out1, amax});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_gather_base_mm", PYBOOST_CALLER(2, custom::npu_all_gather_base_mm));
}
}  // namespace custom
