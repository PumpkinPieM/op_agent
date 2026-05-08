#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_all_gather_base_mm(const ms::Tensor &self, const ms::Tensor &x2, const std::string &hcom, int64_t world_size, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &x1_scale_opt, const std::optional<ms::Tensor> &x2_scale_opt, int64_t gather_index, bool gather_output, int64_t comm_turn, const std::optional<int64_t> &output_dtype_opt, const std::optional<std::string> &comm_mode_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto x1_scale_value = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale_value = x2_scale_opt.value_or(ms::Tensor());
  (void)x1_scale_value;
  (void)x2_scale_value;
  (void)output_dtype_opt;
  auto comm_mode = comm_mode_opt.value_or("ai_cpu");
  (void)comm_mode;
  auto self_shape = self.shape();
  auto x2_shape = x2.shape();
  auto mm_rows = gather_index == 0 ? self_shape[0] * world_size : self_shape[0];
  auto out0 = ms::Tensor(self.data_type(), std::vector<int64_t>{mm_rows, x2_shape[1]});
  auto gather_rows = (gather_index == 0 ? self_shape[0] : x2_shape[0]) * world_size;
  auto gather_cols = gather_index == 0 ? self_shape[1] : x2_shape[1];
  auto out1 = gather_output ? ms::Tensor(self.data_type(), std::vector<int64_t>{gather_rows, gather_cols})
                            : ms::Tensor(self.data_type(), std::vector<int64_t>{0});
  int64_t stream_mode = 1;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AllGatherMatmul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAllGatherMatmul, self, x2, bias_opt, hcom, gather_index, comm_turn,
                                          stream_mode, out0, out1));
  runner->Run({self, x2, bias_value, x1_scale_value, x2_scale_value}, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_all_gather_base_mm", PYBOOST_CALLER(2, custom::npu_all_gather_base_mm));
}
}  // namespace custom
