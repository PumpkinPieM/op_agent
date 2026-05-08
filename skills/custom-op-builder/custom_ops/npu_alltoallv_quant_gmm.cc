#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_alltoallv_quant_gmm(const ms::Tensor &gmm_x, const ms::Tensor &gmm_weight, const ms::Tensor &gmm_x_scale, const ms::Tensor &gmm_weight_scale, const std::string &hcom, int64_t ep_world_size, const std::vector<int64_t> &send_counts, const std::vector<int64_t> &recv_counts, int64_t gmm_y_dtype, const std::optional<ms::Tensor> &send_counts_tensor_opt, const std::optional<ms::Tensor> &recv_counts_tensor_opt, const std::optional<ms::Tensor> &mm_x_opt, const std::optional<ms::Tensor> &mm_weight_opt, const std::optional<ms::Tensor> &mm_x_scale_opt, const std::optional<ms::Tensor> &mm_weight_scale_opt, const std::optional<int64_t> &gmm_x_quant_mode_opt, const std::optional<int64_t> &gmm_weight_quant_mode_opt, const std::optional<int64_t> &mm_x_quant_mode_opt, const std::optional<int64_t> &mm_weight_quant_mode_opt, bool permute_out_flag, const std::optional<std::vector<int64_t>> &group_size_opt, const std::optional<int64_t> &gmm_x_dtype_opt, const std::optional<int64_t> &gmm_weight_dtype_opt, const std::optional<int64_t> &gmm_x_scale_dtype_opt, const std::optional<int64_t> &gmm_weight_scale_dtype_opt, const std::optional<int64_t> &mm_x_dtype_opt, const std::optional<int64_t> &mm_weight_dtype_opt, const std::optional<int64_t> &mm_x_scale_dtype_opt, const std::optional<int64_t> &mm_weight_scale_dtype_opt, const std::optional<int64_t> &mm_y_dtype_opt) {
  auto send_counts_tensor_value = send_counts_tensor_opt.value_or(ms::Tensor());
  auto recv_counts_tensor_value = recv_counts_tensor_opt.value_or(ms::Tensor());
  auto mm_x_value = mm_x_opt.value_or(ms::Tensor());
  auto mm_weight_value = mm_weight_opt.value_or(ms::Tensor());
  auto mm_x_scale_value = mm_x_scale_opt.value_or(ms::Tensor());
  auto mm_weight_scale_value = mm_weight_scale_opt.value_or(ms::Tensor());
  auto gmm_x_quant_mode = gmm_x_quant_mode_opt.value_or(0);
  auto gmm_weight_quant_mode = gmm_weight_quant_mode_opt.value_or(0);
  auto mm_x_quant_mode = mm_x_quant_mode_opt.value_or(0);
  auto mm_weight_quant_mode = mm_weight_quant_mode_opt.value_or(0);
  auto group_size = group_size_opt.value_or(std::vector<int64_t>{});
  auto gmm_x_dtype = gmm_x_dtype_opt.value_or(0);
  auto gmm_weight_dtype = gmm_weight_dtype_opt.value_or(0);
  auto gmm_x_scale_dtype = gmm_x_scale_dtype_opt.value_or(0);
  auto gmm_weight_scale_dtype = gmm_weight_scale_dtype_opt.value_or(0);
  auto mm_x_dtype = mm_x_dtype_opt.value_or(0);
  auto mm_weight_dtype = mm_weight_dtype_opt.value_or(0);
  auto mm_x_scale_dtype = mm_x_scale_dtype_opt.value_or(0);
  auto mm_weight_scale_dtype = mm_weight_scale_dtype_opt.value_or(0);
  auto mm_y_dtype = mm_y_dtype_opt.value_or(0);
  auto base_shape = gmm_x.shape();
  auto base_dtype = gmm_x.data_type();
  auto out0 = ms::Tensor(base_dtype, base_shape);
  auto out1 = ms::Tensor(base_dtype, base_shape);
  auto out2 = ms::Tensor(base_dtype, base_shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("AlltoAllvQuantGroupedMatMul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAlltoAllvQuantGroupedMatMul, gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, hcom, ep_world_size, send_counts, recv_counts, gmm_y_dtype, send_counts_tensor_opt, recv_counts_tensor_opt, mm_x_opt, mm_weight_opt, mm_x_scale_opt, mm_weight_scale_opt, gmm_x_quant_mode, gmm_weight_quant_mode, mm_x_quant_mode, mm_weight_quant_mode, permute_out_flag, group_size, gmm_x_dtype, gmm_weight_dtype, gmm_x_scale_dtype, gmm_weight_scale_dtype, mm_x_dtype, mm_weight_dtype, mm_x_scale_dtype, mm_weight_scale_dtype, mm_y_dtype, out0, out1, out2));
  runner->Run({gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, send_counts_tensor_value, recv_counts_tensor_value, mm_x_value, mm_weight_value, mm_x_scale_value, mm_weight_scale_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_alltoallv_quant_gmm", PYBOOST_CALLER(3, custom::npu_alltoallv_quant_gmm));
}
}  // namespace custom
