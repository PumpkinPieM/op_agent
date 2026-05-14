#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
using TensorList = std::vector<ms::Tensor>;

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

std::vector<int64_t> OutputShape(const ms::Tensor &x, const ms::Tensor &weight) {
  const auto &x_shape = x.shape();
  const auto &weight_shape = weight.shape();
  const int64_t m = x_shape.empty() ? 0 : x_shape[0];
  int64_t n = 0;
  if (weight_shape.size() == 5) {
    constexpr int64_t kWeightNzA8W8LastDim = 32;
    constexpr int64_t kWeightNzA4W4LastDim = 64;
    n = weight_shape[4] == kWeightNzA8W8LastDim ? weight_shape[1] : weight_shape[1] * kWeightNzA4W4LastDim;
  } else {
    n = weight_shape.size() > 2 ? weight_shape[2] : (weight_shape.empty() ? 0 : weight_shape.back());
  }
  return {m, n / 2};
}
}  // namespace

std::vector<ms::Tensor> npu_grouped_matmul_swiglu_quant_v2(const ms::Tensor &x, const TensorList &weight, const TensorList &weight_scale, const ms::Tensor &x_scale, const ms::Tensor &group_list, const std::optional<ms::Tensor> &smooth_scale_opt, const std::optional<TensorList> &weight_assist_matrix_opt, const std::optional<ms::Tensor> &bias_opt, const std::optional<int64_t> &dequant_mode_opt, const std::optional<int64_t> &dequant_dtype_opt, const std::optional<int64_t> &quant_mode_opt, const std::optional<int64_t> &quant_dtype_opt, const std::optional<int64_t> &group_list_type_opt, const std::optional<std::vector<int64_t>> &tuning_config_opt, const std::optional<int64_t> &x_dtype_opt, const std::optional<int64_t> &weight_dtype_opt, const std::optional<int64_t> &weight_scale_dtype_opt, const std::optional<int64_t> &x_scale_dtype_opt) {
  auto smooth_scale_value = smooth_scale_opt.value_or(ms::Tensor());
  auto weight_assist_matrix_value = weight_assist_matrix_opt.value_or(TensorList{});
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto dequant_mode = dequant_mode_opt.value_or(0);
  int64_t dequant_dtype = dequant_dtype_opt.value_or(0);
  if (dequant_dtype == 6) {
    dequant_dtype = 0;
  } else if (dequant_dtype == 5) {
    dequant_dtype = 1;
  } else if (dequant_dtype == 15) {
    dequant_dtype = 27;
  }
  auto quant_mode = quant_mode_opt.value_or(0);
  auto group_list_type = group_list_type_opt.value_or(0);
  auto tuning_config = std::make_pair(tuning_config_opt.value_or(std::vector<int64_t>{}), true);
  (void)quant_dtype_opt;
  (void)x_dtype_opt;
  (void)weight_dtype_opt;
  (void)weight_scale_dtype_opt;
  (void)x_scale_dtype_opt;
  auto out0 = ms::Tensor(ms::TypeId::kNumberTypeInt8, OutputShape(x, weight[0]));
  auto out1 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{x.shape()[0]});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulSwigluQuantWeightNzV2");
  if (HasOpApi("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize")) {
    runner->SetLaunchFunc([x, weight, weight_scale, weight_assist_matrix_value, bias_opt, x_scale, smooth_scale_opt,
                           group_list, dequant_mode, dequant_dtype, quant_mode, group_list_type, tuning_config, out0,
                           out1](auto dev_ctx, auto stream_id) {
      SetNzStorage(weight[0]);
      LAUNCH_ACLNN(aclnnGroupedMatmulSwigluQuantWeightNzV2, dev_ctx, stream_id, ms::pynative::Arg(x),
                   ms::pynative::Arg(weight), ms::pynative::Arg(weight_scale),
                   ms::pynative::Arg(weight_assist_matrix_value), ms::pynative::Arg(bias_opt),
                   ms::pynative::Arg(x_scale), ms::pynative::Arg(smooth_scale_opt), ms::pynative::Arg(group_list),
                   ms::pynative::Arg(dequant_mode), ms::pynative::Arg(dequant_dtype), ms::pynative::Arg(quant_mode),
                   ms::pynative::Arg(group_list_type), ms::pynative::Arg(tuning_config), ms::pynative::Arg(out0),
                   ms::pynative::Arg(out1));
    });
  } else {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulSwigluQuantV2, x, weight, weight_scale,
                                            weight_assist_matrix_value, bias_opt, x_scale, smooth_scale_opt, group_list,
                                            dequant_mode, dequant_dtype, quant_mode, group_list_type, tuning_config,
                                            out0, out1));
  }
  std::vector<ms::Tensor> inputs = {x, x_scale, group_list, smooth_scale_value, bias_value};
  inputs.insert(inputs.end(), weight.begin(), weight.end());
  inputs.insert(inputs.end(), weight_scale.begin(), weight_scale.end());
  inputs.insert(inputs.end(), weight_assist_matrix_value.begin(), weight_assist_matrix_value.end());
  runner->Run(inputs, {out0, out1});
  return {out0, out1};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_swiglu_quant_v2", PYBOOST_CALLER(2, custom::npu_grouped_matmul_swiglu_quant_v2));
}
}  // namespace custom
