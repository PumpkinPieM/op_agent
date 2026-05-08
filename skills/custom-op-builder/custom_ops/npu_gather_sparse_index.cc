#include <vector>
#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_gather_sparse_index(const ms::Tensor &input, const ms::Tensor &index) {
  auto out_shape = index.shape();
  const auto &input_shape = input.shape();
  out_shape.insert(out_shape.end(), input_shape.begin() + 1, input_shape.end());
  auto out = ms::Tensor(input.data_type(), out_shape);
  constexpr int64_t dim = 0;
  constexpr int64_t batch_dims = 0;
  constexpr int64_t mode = 1;
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GatherV3");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGatherV3, input, dim, index, batch_dims, mode, out));
  runner->Run({input, index}, {out});
  return out;
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_gather_sparse_index", PYBOOST_CALLER(1, custom::npu_gather_sparse_index));
}
}  // namespace custom
