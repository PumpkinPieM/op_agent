#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
bool HasOpApi(const char *api_name) { return mindspore::device::ascend::GetOpApiFunc(api_name) != nullptr; }

void SetNzStorage(const ms::Tensor &tensor) {
  const std::string nz_format = "FRACTAL_NZ";
  tensor.set_format(nz_format);
  auto nd_shape = tensor.shape();
  auto nz_shape =
    mindspore::trans::DeviceShapeTransfer().GetDeviceShapeByFormat(nd_shape, nz_format, tensor.data_type());

  constexpr int64_t kStrideBase = 1;
  constexpr int kStrideOffset = 2;
  auto strides = nd_shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(kStrideBase);
  for (int i = static_cast<int>(strides.size()) - kStrideOffset; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  auto storage_info = std::make_shared<mindspore::TensorStorageInfo>(nd_shape, strides, nz_shape, strides, true);
  MS_EXCEPTION_IF_NULL(tensor.tensor());
  MS_EXCEPTION_IF_NULL(tensor.tensor()->device_address());
  tensor.tensor()->set_storage_info(storage_info);
}
}  // namespace

std::vector<ms::Tensor> npu_grouped_matmul_finalize_routing(const ms::Tensor &x, const ms::Tensor &w, const ms::Tensor &group_list, const std::optional<ms::Tensor> &scale_opt, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &offset_opt, const std::optional<ms::Tensor> &pertoken_scale_opt, const std::optional<ms::Tensor> &shared_input_opt, const std::optional<ms::Tensor> &logit_opt, const std::optional<ms::Tensor> &row_index_opt, const std::optional<int64_t> &dtype_opt, const std::optional<double> &shared_input_weight_opt, const std::optional<int64_t> &shared_input_offset_opt, const std::optional<int64_t> &output_bs_opt, const std::optional<int64_t> &group_list_type_opt, const std::optional<std::vector<int64_t>> &tuning_config_opt, const std::optional<int64_t> &x_dtype_opt, const std::optional<int64_t> &w_dtype_opt, const std::optional<int64_t> &scale_dtype_opt, const std::optional<int64_t> &pertoken_scale_dtype_opt) {
  auto scale_value = scale_opt.value_or(ms::Tensor());
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto pertoken_scale_value = pertoken_scale_opt.value_or(ms::Tensor());
  auto shared_input_value = shared_input_opt.value_or(ms::Tensor());
  auto logit_value = logit_opt.value_or(ms::Tensor());
  auto row_index_value = row_index_opt.value_or(ms::Tensor());
  auto dtype = dtype_opt.value_or(0);
  auto shared_input_weight = static_cast<float>(shared_input_weight_opt.value_or(1.0));
  auto shared_input_offset = shared_input_offset_opt.value_or(0);
  auto output_bs = output_bs_opt.value_or(0);
  auto group_list_type = group_list_type_opt.value_or(1);
  (void)tuning_config_opt;
  (void)x_dtype_opt;
  (void)w_dtype_opt;
  (void)scale_dtype_opt;
  (void)pertoken_scale_dtype_opt;
  auto out_shape = x.shape();
  out_shape[0] = output_bs == 0 ? out_shape[0] : output_bs;
  out_shape[1] = w.shape().back();
  auto out0 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, out_shape);
  std::optional<ms::Tensor> antiquant_scale_opt = std::nullopt;
  std::optional<ms::Tensor> antiquant_offset_opt = std::nullopt;
  bool transpose_x = false;
  bool transpose_w = false;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulFinalizeRoutingWeightNz");
  if (HasOpApi("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize")) {
    auto tuning_config = std::make_pair(tuning_config_opt.value_or(std::vector<int64_t>{}), true);
    runner->SetLaunchFunc([x, w, scale_opt, bias_opt, offset_opt, antiquant_scale_opt, antiquant_offset_opt,
                           pertoken_scale_opt, group_list, shared_input_opt, logit_opt, row_index_opt, dtype,
                           shared_input_weight, shared_input_offset, transpose_x, transpose_w, group_list_type,
                           tuning_config, out0](auto dev_ctx, auto stream_id) {
      SetNzStorage(w);
      LAUNCH_ACLNN(aclnnGroupedMatmulFinalizeRoutingWeightNzV2, dev_ctx, stream_id, ms::pynative::Arg(x), ms::pynative::Arg(w),
                   ms::pynative::Arg(scale_opt), ms::pynative::Arg(bias_opt),
                   ms::pynative::Arg(offset_opt), ms::pynative::Arg(antiquant_scale_opt),
                   ms::pynative::Arg(antiquant_offset_opt), ms::pynative::Arg(pertoken_scale_opt),
                   ms::pynative::Arg(group_list), ms::pynative::Arg(shared_input_opt), ms::pynative::Arg(logit_opt),
                   ms::pynative::Arg(row_index_opt), ms::pynative::Arg(dtype), ms::pynative::Arg(shared_input_weight),
                   ms::pynative::Arg(shared_input_offset), ms::pynative::Arg(transpose_x),
                   ms::pynative::Arg(transpose_w), ms::pynative::Arg(group_list_type), ms::pynative::Arg(tuning_config),
                   ms::pynative::Arg(out0));
    });
  } else {
    runner->SetLaunchFunc([x, w, scale_opt, bias_opt, pertoken_scale_opt, group_list, shared_input_opt, logit_opt,
                           row_index_opt, dtype, shared_input_weight, shared_input_offset, transpose_x, transpose_w,
                           group_list_type, out0](auto dev_ctx, auto stream_id) {
      SetNzStorage(w);
      LAUNCH_ACLNN(aclnnGroupedMatmulFinalizeRoutingWeightNz, dev_ctx, stream_id, ms::pynative::Arg(x),
                   ms::pynative::Arg(w), ms::pynative::Arg(scale_opt), ms::pynative::Arg(bias_opt),
                   ms::pynative::Arg(pertoken_scale_opt), ms::pynative::Arg(group_list),
                   ms::pynative::Arg(shared_input_opt), ms::pynative::Arg(logit_opt), ms::pynative::Arg(row_index_opt),
                   ms::pynative::Arg(dtype), ms::pynative::Arg(shared_input_weight),
                   ms::pynative::Arg(shared_input_offset), ms::pynative::Arg(transpose_x),
                   ms::pynative::Arg(transpose_w), ms::pynative::Arg(group_list_type), ms::pynative::Arg(out0));
    });
  }
  runner->Run({x, w, group_list, scale_value, bias_value, offset_value, pertoken_scale_value, shared_input_value, logit_value, row_index_value}, {out0});
  return {out0};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_finalize_routing", PYBOOST_CALLER(1, custom::npu_grouped_matmul_finalize_routing));
}
}  // namespace custom
