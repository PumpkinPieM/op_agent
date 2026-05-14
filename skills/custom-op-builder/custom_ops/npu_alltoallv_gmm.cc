#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_alltoallv_gmm(const ms::Tensor &gmm_x, const ms::Tensor &gmm_weight, const std::string &hcom, int64_t ep_world_size, const std::vector<int64_t> &send_counts, const std::vector<int64_t> &recv_counts, const std::optional<ms::Tensor> &send_counts_tensor_opt, const std::optional<ms::Tensor> &recv_counts_tensor_opt, const std::optional<ms::Tensor> &mm_x_opt, const std::optional<ms::Tensor> &mm_weight_opt, bool trans_gmm_weight, bool trans_mm_weight, bool permute_out_flag) {
  auto send_counts_tensor_value = send_counts_tensor_opt.value_or(ms::Tensor());
  auto recv_counts_tensor_value = recv_counts_tensor_opt.value_or(ms::Tensor());
  auto mm_x_value = mm_x_opt.value_or(ms::Tensor());
  auto mm_weight_value = mm_weight_opt.value_or(ms::Tensor());
  auto base_shape = gmm_x.shape();
  auto base_dtype = gmm_x.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AlltoAllvGroupedMatMul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAlltoAllvGroupedMatMul, gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, send_counts_tensor_opt, recv_counts_tensor_opt, mm_x_opt, mm_weight_opt, trans_gmm_weight, trans_mm_weight, permute_out_flag, out0, out1, out2));
  runner->Run({gmm_x, gmm_weight, send_counts_tensor_value, recv_counts_tensor_value, mm_x_value, mm_weight_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_alltoallv_gmm", PYBOOST_CALLER(3, custom::npu_alltoallv_gmm));
}
}  // namespace custom
