#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
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
  const int64_t n = weight_shape.size() > 2 ? weight_shape[2] : (weight_shape.empty() ? 0 : weight_shape.back());
  return {m, n / 2};
}
}  // namespace

std::vector<ms::Tensor> npu_grouped_matmul_swiglu_quant(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list, const ms::Tensor &weight_scale, const ms::Tensor &x_scale, const std::optional<ms::Tensor> &bias_opt, const std::optional<ms::Tensor> &offset_opt) {
  auto bias_value = bias_opt.value_or(ms::Tensor());
  auto offset_value = offset_opt.value_or(ms::Tensor());
  auto out0 = ms::Tensor(ms::TypeId::kNumberTypeInt8, OutputShape(x, weight));
  auto out1 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{x.shape()[0]});
  auto out2 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{});
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulSwigluQuantWeightNZ");
  runner->SetLaunchFunc([x, weight, bias_opt, offset_opt, weight_scale, x_scale, group_list, out0, out1, out2](
                          auto dev_ctx, auto stream_id) {
    SetNzStorage(weight);
    LAUNCH_ACLNN(aclnnGroupedMatmulSwigluQuantWeightNZ, dev_ctx, stream_id, ms::pynative::Arg(x),
                 ms::pynative::Arg(weight), ms::pynative::Arg(bias_opt), ms::pynative::Arg(offset_opt),
                 ms::pynative::Arg(weight_scale), ms::pynative::Arg(x_scale), ms::pynative::Arg(group_list),
                 ms::pynative::Arg(out0), ms::pynative::Arg(out1), ms::pynative::Arg(out2));
  });
  runner->Run({x, weight, group_list, weight_scale, x_scale, bias_value, offset_value}, {out0, out1, out2});
  return {out0, out1, out2};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul_swiglu_quant", PYBOOST_CALLER(3, custom::npu_grouped_matmul_swiglu_quant));
}
}  // namespace custom
