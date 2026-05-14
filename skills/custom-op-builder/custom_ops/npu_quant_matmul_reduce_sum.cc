#include <optional>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
std::vector<int64_t> ReduceSumShape(const ms::Tensor &x1, const ms::Tensor &x2) {
  auto x1_shape = x1.shape();
  auto x2_shape = x2.shape();
  if (x1_shape.size() < 3 || x2_shape.size() < 3) {
    return x1_shape;
  }
  return {x1_shape[1], x2_shape.back()};
}

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

ms::Tensor npu_quant_matmul_reduce_sum(const ms::Tensor &x1, const ms::Tensor &x2,
                                       const std::optional<ms::Tensor> &x1_scale_opt,
                                       const std::optional<ms::Tensor> &x2_scale_opt) {
  auto out = ms::Tensor(ms::TypeId::kNumberTypeBFloat16, ReduceSumShape(x1, x2));
  auto x1_scale = x1_scale_opt.value_or(ms::Tensor());
  auto x2_scale = x2_scale_opt.value_or(ms::Tensor());
  std::optional<ms::Tensor> empty_tensor_opt = std::nullopt;
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  int64_t group_size = -1;
  auto dims = std::make_pair(std::vector<int64_t>{0}, true);
  bool keep_dims = false;

  SetNzStorage(x2);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulReduceSumWeightNz");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulReduceSumWeightNz, x1, x2, x1_scale_opt, x2_scale_opt,
                                          empty_tensor_opt, empty_tensor_opt, empty_tensor_opt, empty_tensor_opt,
                                          empty_tensor_opt, transpose_x1, transpose_x2, group_size, dims, keep_dims,
                                          out));
  runner->Run({x1, x2, x1_scale, x2_scale}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_quant_matmul_reduce_sum", PYBOOST_CALLER(1, custom::npu_quant_matmul_reduce_sum));
}
}  // namespace custom
