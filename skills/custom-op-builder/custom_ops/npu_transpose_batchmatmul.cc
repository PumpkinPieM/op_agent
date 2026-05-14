#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"

namespace custom {

std::vector<ms::Tensor> npu_transpose_batchmatmul(const ms::Tensor &x, const ms::Tensor &weight, const std::optional<ms::Tensor> &bias_opt = std::nullopt, bool perm_x1 = false, bool perm_x2 = false) {
  auto bias = bias_opt.value_or(ms::Tensor());
  auto shape = x.shape();
  if (!shape.empty() && !weight.shape().empty()) shape.back() = weight.shape().back();
  auto out = ms::Tensor(x.data_type(), shape);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("TransposeBatchMatMul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnTransposeBatchMatMul, x, weight, bias_opt, perm_x1, perm_x2, out));
  runner->Run({x, weight, bias}, {out});
  return {out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_transpose_batchmatmul", PYBOOST_CALLER(1, custom::npu_transpose_batchmatmul));
}
}  // namespace custom
