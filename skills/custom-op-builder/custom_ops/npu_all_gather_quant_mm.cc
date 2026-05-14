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
    case 2:
      return ms::TypeId::kNumberTypeInt8;
    case 3:
      return ms::TypeId::kNumberTypeInt32;
    case 4:
      return ms::TypeId::kNumberTypeUInt8;
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

int64_t GroupSizeFromList(const std::optional<std::vector<int64_t>> &group_sizes_opt) {
  if (!group_sizes_opt.has_value() || group_sizes_opt->empty()) {
    return 0;
  }
  return group_sizes_opt->front();
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

std::vector<ms::Tensor> npu_all_gather_quant_mm(const ms::Tensor &self, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt, const std::optional<ms::Tensor> &quant_scale_opt, int64_t block_size, int64_t gather_index, bool gather_output, int64_t comm_turn, const std::optional<std::vector<int64_t>> &group_sizes_opt, bool amax_output, const std::optional<int64_t> &y_dtype_opt, const std::optional<int64_t> &x1_dtype_opt, const std::optional<int64_t> &x2_dtype_opt, const std::optional<int64_t> &x1_scale_dtype_opt, const std::optional<int64_t> &x2_scale_dtype_opt) {
  (void)x1_dtype_opt;
  (void)x2_dtype_opt;
  (void)x1_scale_dtype_opt;
  (void)x2_scale_dtype_opt;

  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto x1_scale_value = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale_value = x2_scale_opt.value_or(ms::Tensor());
  auto quant_scale_value = quant_scale_opt.value_or(ms::Tensor());
  auto out0 = ms::Tensor(DTypeFromAclType(y_dtype_opt, self.data_type()), MatmulShape(self, x2, world_size, gather_index));
  auto out1 = ms::Tensor(gather_index == 0 ? self.data_type() : x2.data_type(),
                         GatherShape(self, x2, world_size, gather_index, gather_output));
  auto out2 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, amax_output ? std::vector<int64_t>{1} : std::vector<int64_t>{0});
  int64_t stream_mode = 1;
  int64_t group_size = GroupSizeFromList(group_sizes_opt);
  const char *comm_mode = "aiv";

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AllGatherMatmulV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAllGatherMatmulV2, self, x2, bias_opt, x1_scale_opt, x2_scale_opt,
                                          quant_scale_opt, block_size, hcom, gather_index, comm_turn, stream_mode,
                                          group_size, comm_mode, out0, out1, out2));
  runner->Run({self, x2, bias_value, x1_scale_value, x2_scale_value, quant_scale_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_gather_quant_mm", PYBOOST_CALLER(3, custom::npu_all_gather_quant_mm));
}
}  // namespace custom
